use crate::core::common_utils::Bbox;
use anyhow::{Error, Result};
use image::DynamicImage;
use serde::{Deserialize, Serialize};

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
#[derive(Debug)]
pub struct OnnxModel {
    pub model: ort::Session,
    pub is_fp16: bool,
}

impl OnnxModel {
    pub fn new(model: ort::Session, is_fp16: bool) -> Result<OnnxModel, Error> {
        Ok(OnnxModel { model, is_fp16 })
    }
}

pub trait Inference {
    /// runs a forward pass,
    /// outputs [InferenceResult]
    fn forward(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
    ) -> Result<InferenceResult, Error>;

    /// load or panic bih
    fn load(model_path: &str, fp16: bool) -> OnnxModel;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ort::{GraphOptimizationLevel, SessionBuilder};
    use std::path::Path;

    // Mock implementation of Inference trait for testing
    struct MockInference;

    impl Inference for MockInference {
        fn forward(
            &self,
            _input_image: &DynamicImage,
            _confidence_threshold: f32,
        ) -> Result<InferenceResult, Error> {
            Ok(InferenceResult::FaceDetection(vec![Bbox::new(
                0.0,
                0.0,
                100.0,
                100.0,
                0.9,
                vec![],
            )]))
        }

        fn load(_model_path: &str, fp16: bool) -> OnnxModel {
            let session = SessionBuilder::new()
                .unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level1)
                .unwrap()
                .with_intra_threads(1)
                .unwrap()
                .commit_from_file(Path::new("./models/yolov8n-face.onnx"))
                .unwrap();

            OnnxModel {
                model: session,
                is_fp16: fp16,
            }
        }
    }

    #[test]
    fn test_onnx_model_creation() -> Result<(), Error> {
        let session = SessionBuilder::new()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .unwrap()
            .with_intra_threads(1)
            .unwrap()
            .commit_from_file(Path::new("./models/yolov8n-face.onnx"))
            .unwrap();

        let model = OnnxModel::new(session, false)?;

        assert!(!model.is_fp16);
        Ok(())
    }

    #[test]
    fn test_inference_trait() {
        let mock_inference = MockInference;
        let mock_image = DynamicImage::new_rgb8(100, 100);

        let result = mock_inference.forward(&mock_image, 0.5);
        assert!(result.is_ok());

        match result.unwrap() {
            InferenceResult::FaceDetection(bboxes) => {
                assert_eq!(bboxes.len(), 1);
                assert_eq!(bboxes[0].x1, 0.0);
                assert_eq!(bboxes[0].y1, 0.0);
                assert_eq!(bboxes[0].x2, 100.0);
                assert_eq!(bboxes[0].y2, 100.0);
                assert_eq!(bboxes[0].confidence, 0.9);
            }
            _ => panic!("Expected FaceDetection result"),
        }

        let model = MockInference::load("mock_path.onnx", true);
        assert!(model.is_fp16);
    }

    #[test]
    fn test_inference_result_serialization() {
        let face_detection =
            InferenceResult::FaceDetection(vec![Bbox::new(0.0, 0.0, 100.0, 100.0, 0.9, vec![])]);
        let face_embedding = InferenceResult::FaceEmbedding(vec![0.1; 512]);

        let serialized_detection = serde_json::to_string(&face_detection).unwrap();
        let serialized_embedding = serde_json::to_string(&face_embedding).unwrap();

        let deserialized_detection: InferenceResult =
            serde_json::from_str(&serialized_detection).unwrap();
        let deserialized_embedding: InferenceResult =
            serde_json::from_str(&serialized_embedding).unwrap();

        match deserialized_detection {
            InferenceResult::FaceDetection(bboxes) => {
                assert_eq!(bboxes.len(), 1);
                assert_eq!(bboxes[0].x1, 0.0);
                assert_eq!(bboxes[0].y1, 0.0);
                assert_eq!(bboxes[0].x2, 100.0);
                assert_eq!(bboxes[0].y2, 100.0);
                assert_eq!(bboxes[0].confidence, 0.9);
            }
            _ => panic!("Expected FaceDetection result"),
        }

        match deserialized_embedding {
            InferenceResult::FaceEmbedding(embedding) => {
                assert_eq!(embedding.len(), 512);
                assert!(embedding.iter().all(|&x| (x - 0.1).abs() < 1e-6));
            }
            _ => panic!("Expected FaceEmbedding result"),
        }
    }
}
