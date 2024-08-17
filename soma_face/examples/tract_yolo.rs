use soma_face::core::common_utils::{non_maximum_suppression, Bbox};
use soma_face::core::onnx_backend::{Inference, OnnxModel};
use soma_face::get_face::retinaface::GetFaceRetinaface;
use soma_face::get_face::yolo::{preprocess_face_f32_yolo, GetFaceYolo};
use tract_onnx::prelude::*;
use tract_ndarray::{Array, s};
use anyhow::{Result, Error};
use ort::inputs;
use clap::{Parser};

#[derive(Parser)]
struct CliArgs {
    #[arg(long)]
    input_image: String
}


fn main() -> Result<(), Error>{
    println!("fuck");
    let args = CliArgs::parse();

    let model = tract_onnx::onnx()
        .model_for_path("./models/yolov8n-face.onnx")?
        .with_input_fact(0, f32::fact([1,3,640,640]).into())?
        .into_optimized()?
        .into_runnable()?;
    let rimg = image::open(&args.input_image)?.to_rgb8();
    let ort_rimg = image::open(&args.input_image)?;
    let resized = image::imageops::resize(&rimg, 640, 640, image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 640, 640), |(_, c, y, x)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into();

    let ort_image = preprocess_face_f32_yolo(&ort_rimg)?; 
    for i in 0..5 {
        let tract_time = std::time::Instant::now();
        let result = model.run(tvec!(image.to_owned().into()))?;
        println!("tract_time {:?}",tract_time.elapsed());
    }

    // let _raw_output = result[0].to_array_view::<f32>()?.view().t().into_owned();    
    // let mut bbox_vec: Vec<Bbox> = vec![];
    // for i in 0.._raw_output.len_of(tract_ndarray::Axis(0)) {
    //         let row = _raw_output.slice(s![i, .., ..]);
    //         let confidence = row[[4, 0]];
    //         if &confidence >= &0.01 {
    //             let x = row[[0, 0]];
    //             let y = row[[1, 0]];
    //             let w = row[[2, 0]];
    //             let h = row[[3, 0]];

    //             let x1 = x - w / 2.0;
    //             let y1 = y - h / 2.0;
    //             let x2 = x + w / 2.0;
    //             let y2 = y + h / 2.0;
    //             let bbox = Bbox::new(x1, y1, x2, y2, confidence, vec![vec![]]).apply_image_scale(
    //                 &ort_rimg,
    //                 640.0,
    //                 640.0,
    //             );
    //             bbox_vec.push(bbox);
    //         }
    // }
    for i in 0..5 {
        let onnx_time = std::time::Instant::now();    
        let yolo = GetFaceYolo::new("./models/yolov8n-face.onnx", 640,640,false)?;
        println!("onnx time {:?}", onnx_time.elapsed());
    }

    // let res = yolo.onnx_model.model.run(inputs!["images" => ort_image.view()]?)?;
    // let _raw_output_ort = res["output0"]
    //         .try_extract_tensor::<f32>()?
    //         .view()
    //         .t()
    //         .into_owned();
    // println!("res2 :{:?}", _raw_output_ort);
    // println!("ouptut {:?}", _raw_output.t());
    // println!("bboxes : {:?}", bbox_vec);
    Ok(())
}



/* pub fn preprocess_face_f32_yolo(
    image_source: &DynamicImage,
) -> Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>, Error> {
    let img = image_source.resize_exact(640, 640, imageops::FilterType::Triangle);
    let mut preproc: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> = Array::ones((1, 3, 640, 640));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        preproc[[0, 0, y, x]] = r as f32 / 255.0;
        preproc[[0, 1, y, x]] = g as f32 / 255.0;
        preproc[[0, 2, y, x]] = b as f32 / 255.0;
    }
    Ok(preproc)

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
                    640.0,
                    640.0,
                );
                bbox_vec.push(bbox);
            }
        }
        Ok(InferenceResult::FaceDetection(non_maximum_suppression(
            bbox_vec, 0.5,
        )))
    }

} */
