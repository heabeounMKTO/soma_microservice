use soma_core::common_utils::image_to_ndarray;
use soma_core::onnx_backend::{Inference, InferenceResult, OnnxModel};
use anyhow::{Error, Result};
use image::imageops;
use image::DynamicImage;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use ort::{inputs, CPUExecutionProvider, GraphOptimizationLevel, Session};

pub struct GetFaceVecArcFace {
    pub onnx_model: OnnxModel,
}

impl GetFaceVecArcFace {
    pub fn new(model_path: &str) -> Result<GetFaceVecArcFace, Error> {
        Ok(GetFaceVecArcFace {
            onnx_model: GetFaceVecArcFace::load(model_path, false),
        })
    }
}

impl Inference for GetFaceVecArcFace {
    fn load(model_path: &str, fp16: bool) -> OnnxModel {
        let model: ort::Session = ort::Session::builder()
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
    /// note: confidence is literally not used in this context.
    fn forward(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
    ) -> Result<InferenceResult, Error> {
        let _ = confidence_threshold;
        let preprocess_image = preprocess_arcface(input_image)?;
        let inference = self
            .onnx_model
            .model
            .run(inputs!["data" => preprocess_image.view()]?)?;
        let results = inference["fc1"].try_extract_tensor::<f32>()?.into_owned();
        Ok(InferenceResult::FaceEmbedding(results.into_raw_vec()))
    }
}

fn preprocess_arcface(
    image_source: &DynamicImage,
) -> Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>, Error> {
    let img = image_source.resize_exact(112, 112, imageops::FilterType::Triangle);

    let mut ndarray_image = image_to_ndarray(&img);
    ndarray_image -= 127.5;
    ndarray_image *= 1.0 / 128.0;
    let _final = ndarray_image.permuted_axes((2, 0, 1));
    Ok(_final.insert_axis(ndarray::Axis(0)))
}

pub fn load_arcface(model_path: &str, fp16: bool) -> OnnxModel {
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
