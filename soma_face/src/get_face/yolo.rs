use soma_core::common_utils::{non_maximum_suppression, Bbox};
use soma_core::onnx_backend::{Inference, InferenceResult, OnnxModel};
use anyhow::{Error, Result};
use image::imageops;
use image::{DynamicImage, GenericImageView};
use ndarray::{s, Array, ArrayBase, Axis, Dim, IxDyn, OwnedRepr};
use ort::{
    inputs, CPUExecutionProvider, ExecutionProvider, GraphOptimizationLevel, Session,
    SessionOutputs,
};

#[cfg(feature="tractinference")]
use tract_data::tensor::Tensor;

fn scale_wh(w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
    let r = (w1 / w0).min(h1 / h0);
    (r, (w0 * r).round(), (h0 * r).round())
}

#[cfg(feature="tractinference")]
pub fn preprocess_face_f32_yolo(
    image_source: &DynamicImage
) -> Result<tract_data::tensor::Tensor>{
    let width = input_image.width();
    let height = input_image.height();
    let scale = 640.0 / width.max(height) as f32;
    let new_width = (width as f32 * scale) as u32;
    let new_height = (height as f32 * scale) as u32;
    let resized = image::imageops::resize(&raw_image.to_rgb8(), new_width, new_height, image::imageops::FilterType::Triangle);
    let mut padded = image::RgbImage::new(640, 640);
    image::imageops::replace(&mut padded, &resized, (640 - new_width as i64) / 2, (640 - new_height as i64) / 2);
    
    tract_ndarray::Array4::from_shape_fn((1, 3, 640, 640), |(_, c, y, x)| {
        padded.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
    })
    .into()
}

#[cfg(not(feature="tractinference"))]
pub fn preprocess_face_f32_yolo(
    image_source: &DynamicImage,
) -> Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>, Error> {
    let mut preproc: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> = Array::ones((1, 3, 640, 640));

    // TODO: refactor to funciton arg 
    let (_, w_new, h_new) = scale_wh(image_source.width() as f32,
                                      image_source.height() as f32, 
                                    640.0,640.0);
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
    #[cfg(not(feature="tractinference"))]
    pub fn new(
        model_path: &str,
        width: i32,
        height: i32,
        use_fp16: bool,
    ) -> Result<GetFaceYolo, Error> {
        Ok(GetFaceYolo {
            onnx_model: GetFaceYolo::load(model_path, use_fp16),
            width: width,
            height: height,
        })
    }

    #[cfg(feature="tractinference")]
    pub fn new(
        model_path: &str,
        width: i32,
        height: i32,
        use_fp16: bool,
    ) -> Result<GetFaceYolo, Error> {
        Ok(GetFaceYolo {
            onnx_model: GetFaceYolo::load(model_path, use_fp16),
            width: width,
            height: height,
        })
    }
}

impl Inference for GetFaceYolo {
    #[cfg(feature="tractinference")]
    fn forward(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32
    ) -> Result<InferenceResult, Error> {
        let preprocess_image = preprocess_face_f32_yolo(input_image);

    }

    #[cfg(not(feature="tractinference"))]
    fn forward(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
    ) -> Result<InferenceResult, Error> {
        let preprocess_image = preprocess_face_f32_yolo(input_image)?;
        let inference = self
            .onnx_model
            .model
            .run(inputs!["images" => preprocess_image.view()]?)?;
        let _raw_output = inference["output0"]
            .try_extract_tensor::<f32>()?
            .view()
            .t()
            .into_owned();
        let (_, w_new, h_new) = scale_wh(input_image.width() as f32,
                              input_image.height() as f32, 
                            640.0,640.0);
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
                let bbox = Bbox::new(x1, y1, x2, y2, confidence, vec![vec![]]).apply_image_scale(
                    &input_image,
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
    #[cfg(not(feature="tractinference"))]
    fn load(model_path: &str, fp16: bool) -> OnnxModel {
        let model: ort::Session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();
        OnnxModel {
            model: model,
            is_fp16: fp16,
        }
    }
    
    #[cfg(feature="tractinference")]
    fn load(model_path: &str, fp16: bool) -> OnnxModel {
        let model = tract_onnx::onnx()
            .model_for_path(model_path).unwrap()
            .with_input_fact(0, f32::fact([1,3,640,640]).into())?
            .into_optimized().unwrap()
            .into_runnable().unwrap();
        OnnxModel {
            model,
            fp16
        }
    }
}
