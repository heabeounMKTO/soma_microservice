use crate::common_utils::Bbox;
use anyhow::{Error, Result};
use image::DynamicImage;
use serde::{Deserialize, Serialize};

#[cfg(feature="smol")]
use tract_onnx::prelude::*;

#[cfg(feature="smol")]
use tract_core::model::typed::RunnableModel;

/// FaceDetection is a `Vec<Bbox>` (for bbox detection of faces)
///
///
/// FaceEmbedding is a Vec[f32; 512] face vector (latent)
#[derive(Debug, Serialize, Deserialize)]
pub enum InferenceResult {
    FaceDetection(Vec<Bbox>),
    FaceEmbedding(Vec<f32>),
}

/// half precision not implemented !
/// (probably never will be)
///
#[cfg(not(feature="smol"))]
#[derive(Debug)]
pub struct OnnxModel {
    pub model: ort::Session,
    pub is_fp16: bool,
}

#[cfg(feature="smol")]
#[derive(Debug)]
pub struct OnnxModel {
    pub model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    pub is_fp16: bool,
}



impl OnnxModel {
    #[cfg(not(feature="smol"))]
    pub fn new(model: ort::Session, is_fp16: bool) -> Result<OnnxModel, Error> {
        Ok(OnnxModel { model, is_fp16 })
    }

    #[cfg(feature="smol")]
    pub fn new(model:  SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>, is_fp16: bool) -> Result<OnnxModel, Error> {
        Ok(OnnxModel { model, is_fp16 })
    }
}

pub trait Inference {
    /// runs a forward pass,
    /// outputs [InferenceResult]
    ///
    fn forward(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
    ) -> Result<InferenceResult, Error>;

    /// load or panic bih
    fn load(model_path: &str, fp16: bool) -> OnnxModel;
}
