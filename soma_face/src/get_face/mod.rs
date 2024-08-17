//! abstraction for running face detection models.
pub mod retinaface;
pub mod yolo;

// use crate::core::{
//     common_utils::{sort_conf_bbox, Bbox},
//     onnx_backend::{Inference, InferenceResult},
// };
use soma_core::{
    common_utils::{sort_conf_bbox, Bbox},
    onnx_backend::{Inference, InferenceResult}
};
use anyhow::{Error, Result};
use image::DynamicImage;
use retinaface::GetFaceRetinaface;
use yolo::GetFaceYolo;

/// Abstraction for getting faces from pictures by using various models (YOLO, RetinaFace)
pub struct FaceExtractor {
    pub width: i32,
    pub height: i32,
    pub retina_model: GetFaceRetinaface,
    pub yolo_model: GetFaceYolo,
}

impl FaceExtractor {
    /// either all of them loads or none of them loads !
    pub fn new(
        retina_path: &str,
        yolo_path: &str,
        width: i32,
        height: i32,
    ) -> Result<FaceExtractor, Error> {
        let load_retina_model = GetFaceRetinaface::new(retina_path, width, height, false).unwrap();
        let load_yolo_model = GetFaceYolo::new(yolo_path, width, height, false).unwrap();
        Ok(FaceExtractor {
            width,
            height,
            retina_model: load_retina_model,
            yolo_model: load_yolo_model,
        })
    }
    fn forward_retinaface(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
    ) -> Result<Vec<Bbox>, Error> {
        let fwd = self
            .retina_model
            .forward(input_image, confidence_threshold)?;
        match fwd {
            InferenceResult::FaceDetection(results) => Ok(results),
            _ => unreachable!("invalid InferenceResult output"),
        }
    }

    fn forward_yolo(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
    ) -> Result<Vec<Bbox>, Error> {
        let fwd = self.yolo_model.forward(input_image, confidence_threshold)?;
        match fwd {
            InferenceResult::FaceDetection(results) => Ok(results),
            _ => unreachable!("invalid InferenceResult output"),
        }
    }

    pub fn extract_face_from_image(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
    ) -> Result<Vec<Bbox>, Error> {
        let yolo_results = match self.yolo_model.forward(input_image, confidence_threshold)? {
            InferenceResult::FaceDetection(res) => res,
            _ => unreachable!("invalid `InferenceResult`"),
        };
        let retina_results = match self
            .retina_model
            .forward(input_image, confidence_threshold)?
        {
            InferenceResult::FaceDetection(res) => res,
            _ => unreachable!("invalid `InferenceResult`"),
        };

        // combines results from all the inferece and takes the one with the highest confidence
        let mut combined_res = yolo_results;
        combined_res.extend(retina_results.clone());
        if combined_res.len() > 0 {
            combined_res = sort_conf_bbox(&mut combined_res);
            Ok(vec![combined_res[0].to_owned()])
        } else {
            Ok(vec![])
        }
    }
}
