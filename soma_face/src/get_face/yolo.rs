use crate::core::common_utils::{non_maximum_suppression, Bbox};
use crate::core::onnx_backend::{Inference, InferenceResult, OnnxModel};
use anyhow::{Error, Result};
use image::imageops;
use image::{DynamicImage, GenericImageView};
use ndarray::{s, Array, ArrayBase, Axis, Dim, OwnedRepr};
use ort::{inputs, CPUExecutionProvider, GraphOptimizationLevel, Session};

#[cfg(feature = "gpuinference")]
use ort::TensorRTExecutionProvider;

fn scale_wh(w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
    let r = (w1 / w0).min(h1 / h0);
    (r, (w0 * r).round(), (h0 * r).round())
}
/// preprocess for yolo detections.
pub fn preprocess_face_f32_yolo(
    image_source: &DynamicImage,
    target_size: i32,
) -> Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>, Error> {
    let mut preproc: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> =
        Array::ones((1, 3, target_size as usize, target_size as usize));
    preproc.fill(144.0 / 255.0);
    let (_, w_new, h_new) = scale_wh(
        image_source.width() as f32,
        image_source.height() as f32,
        target_size as f32,
        target_size as f32,
    );
    let img = image_source.resize_exact(w_new as u32, h_new as u32, imageops::FilterType::Triangle);
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        preproc[[0, 0, y, x]] = r as f32 / 255.0;
        preproc[[0, 1, y, x]] = g as f32 / 255.0;
        preproc[[0, 2, y, x]] = b as f32 / 255.0;
    }
    Ok(preproc)
}

pub struct GetFaceYolo {
    pub onnx_model: OnnxModel,
    pub width: i32,
    pub height: i32,
}

impl GetFaceYolo {
    pub fn new(
        model_path: &str,
        width: i32,
        height: i32,
        use_fp16: bool,
    ) -> Result<GetFaceYolo, Error> {
        let get_face = GetFaceYolo {
            onnx_model: GetFaceYolo::load(model_path, use_fp16),
            width,
            height,
        };
        // run warmup
        //
        #[cfg(feature = "gpuinference")]
        let warmup = {
            println!("running warmup");
            let img = image::open("test_image.jpg").unwrap();
            for i in 0..5 {
                let _ = get_face.forward(&img, 0.1);
            }
            println!("warmup done!");
            println!("TensorRT engine loaded!");
        };
        Ok(get_face)
    }
}

impl Inference for GetFaceYolo {
    fn forward(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
    ) -> Result<InferenceResult, Error> {
        let preprocess_image = preprocess_face_f32_yolo(input_image, self.width.to_owned())?;
        let inference = self
            .onnx_model
            .model
            .run(inputs!["images" => preprocess_image.view()]?)?;
        let _raw_output = inference["output0"]
            .try_extract_tensor::<f32>()?
            .view()
            .t()
            .into_owned();
        let (_, w_new, h_new) = scale_wh(
            input_image.width() as f32,
            input_image.height() as f32,
            self.width as f32,
            self.height as f32,
        );
        let mut bbox_vec: Vec<Bbox> = vec![];
        for i in 0.._raw_output.len_of(Axis(0)) {
            let row = _raw_output.slice(s![i, .., ..]);
            let confidence = row[[4, 0]];
            if &confidence >= &confidence_threshold {
                let x = row[[0, 0]];
                let y = row[[1, 0]];
                let w = row[[2, 0]];
                let h = row[[3, 0]];

                let x1 = x - w / 2.0;
                let y1 = y - h / 2.0;
                let x2 = x + w / 2.0;
                let y2 = y + h / 2.0;
            
            let mut kpss = Vec::new();
            for j in (5.._raw_output.len_of(Axis(1))).step_by(3) {
                let kp_x = row[[j, 0]];
                let kp_y = row[[j + 1, 0]];
                kpss.push(vec![kp_x, kp_y]);
            }
                let bbox = Bbox::new(x1, y1, x2, y2, confidence, kpss).apply_image_scale(
                    input_image,
                    w_new,
                    h_new,
                );
                bbox_vec.push(bbox);
            }
        }
        Ok(InferenceResult::FaceDetection(non_maximum_suppression(
            bbox_vec, 0.5,
        )))
    }

    fn load(model_path: &str, fp16: bool) -> OnnxModel {
        // cuda or no cuda
        // TODO: add warmup function if running on CUDA
        //
        #[cfg(not(feature = "gpuinference"))]
        let execution_providers = [CPUExecutionProvider::default().build()];
        #[cfg(feature = "gpuinference")]
        let execution_providers = [TensorRTExecutionProvider::default()
            .with_device_id(0)
            .with_timing_cache(true)
            .with_engine_cache(true)
            .with_engine_cache_path("/tmp")
            .build()];

        let model: ort::Session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_execution_providers(execution_providers)
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();
        let onnx = OnnxModel {
            model,
            is_fp16: fp16,
        };
        onnx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;
    use std::path::Path;

    #[test]
    fn test_get_face_yolo() -> Result<(), Error> {
        // Create a mock image
        let width = 800;
        let height = 600;
        let mut img = RgbImage::new(width, height);
        // Fill the image with a solid color (e.g., white)
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]);
        }
        let dynamic_image = image::open("/home/hbdesk/Downloads/cwigl.jpeg").unwrap();

        // Create a mock ONNX model
        // Note: You'll need to replace this with a path to a real ONNX model file
        let model_path = Path::new("./models/yolov8n-face.onnx");

        // Initialize GetFaceYolo
        let face_detector = GetFaceYolo::new(
            model_path.to_str().unwrap(),
            width as i32,
            height as i32,
            false,
        )?;

        // Run inference
        let confidence_threshold = 0.5;
        let result = face_detector.forward(&dynamic_image, confidence_threshold)?;

        // Check the result
        match result {
            InferenceResult::FaceDetection(bboxes) => {
                // Add assertions based on expected behavior
                // For example:
                assert!(!bboxes.is_empty(), "Expected at least one face detection");

                // Check properties of the first bbox
                let first_bbox = &bboxes[0];
                assert!(
                    first_bbox.confidence >= confidence_threshold,
                    "Confidence should be above threshold"
                );
                assert!(
                    first_bbox.x1 >= 0.0 && first_bbox.x1 < width as f32,
                    "X1 should be within image bounds"
                );
                assert!(
                    first_bbox.y1 >= 0.0 && first_bbox.y1 < height as f32,
                    "Y1 should be within image bounds"
                );
                assert!(
                    first_bbox.x2 > first_bbox.x1 && first_bbox.x2 <= width as f32,
                    "X2 should be greater than X1 and within image bounds"
                );
                assert!(
                    first_bbox.y2 > first_bbox.y1 && first_bbox.y2 <= height as f32,
                    "Y2 should be greater than Y1 and within image bounds"
                );
            }
            _ => panic!("Expected FaceDetection result"),
        }

        Ok(())
    }

    #[test]
    fn test_preprocess_face_f32_yolo() -> Result<(), Error> {
        // Create a mock image
        let width = 800;
        let height = 600;
        let mut img = RgbImage::new(width, height);
        // Fill the image with a solid color (e.g., white)
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]);
        }
        let dynamic_image = DynamicImage::ImageRgb8(img);

        // Run preprocessing
        let preprocessed = preprocess_face_f32_yolo(&dynamic_image)?;

        // Check preprocessed image properties
        assert_eq!(
            preprocessed.shape(),
            &[1, 3, 640, 640],
            "Preprocessed image should have shape [1, 3, 640, 640]"
        );

        // Check some values in the preprocessed image
        // The background should be filled with 144.0/255.0
        // assert!((preprocessed[[0, 0, 0, 0]] - 144.0/255.0).abs() < 1e-6, "Background should be filled with 144.0/255.0");

        // The center of the image should be white (1.0, 1.0, 1.0)
        let center_x = 320;
        let center_y = 240;
        assert!(
            (preprocessed[[0, 0, center_y, center_x]] - 1.0).abs() < 1e-6,
            "Center pixel should be white (R channel)"
        );
        assert!(
            (preprocessed[[0, 1, center_y, center_x]] - 1.0).abs() < 1e-6,
            "Center pixel should be white (G channel)"
        );
        assert!(
            (preprocessed[[0, 2, center_y, center_x]] - 1.0).abs() < 1e-6,
            "Center pixel should be white (B channel)"
        );

        Ok(())
    }
}
