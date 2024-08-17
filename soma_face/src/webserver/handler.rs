use soma_core::common_utils::{sort_conf_bbox, Bbox};
use crate::webserver::service;
use actix_multipart::form::tempfile::TempFile;
use actix_multipart::form::text::Text;
use actix_multipart::form::MultipartForm;
use actix_multipart::Multipart;
use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::{SwaggerUi, Url};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema)]
pub struct FaceResponse {
    pub confidence: f32,
    pub height: i32,
    pub width: i32,
}

/// returns width and height from bbox
fn get_wh(input_bbox: &Bbox) -> (i32, i32) {
    let width = input_bbox.x2 - input_bbox.x1;
    let height = input_bbox.y2 - input_bbox.y1;
    (width as i32, height as i32)
}

impl FaceResponse {
    /// takes in a vec of bbox
    pub fn from_bbox_vec(input_vec: &Vec<Bbox>) -> Vec<FaceResponse> {
        // let mut faces: Vec<FaceResponse> = vec![];
        let faces = input_vec
            .clone()
            .into_iter()
            .map(|x| {
                let conf = x.confidence.to_owned();
                let (w, h) = get_wh(&x);
                let resp = FaceResponse {
                    confidence: conf,
                    width: w,
                    height: h,
                };
                resp
            })
            .collect();
        faces
    }
}

#[derive(Debug, MultipartForm, ToSchema)]
pub struct GetFaceVecRequest {
    #[schema(value_type = String, example = "face.png")]
    pub input: TempFile,

    #[schema(value_type = bool, example = "true", required=false)]
    pub aligned: Text<bool>,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct GetFaceVecResponse {
    #[schema(value_type = Vec<f32>, example = "[-0.231,...,-0.42]")]
    pub data: Vec<f32>,
}

#[derive(Debug, MultipartForm, ToSchema)]
pub struct GetFaceRequest {
    #[schema(value_type = String, example = "face.png")]
    pub input: TempFile,
}

#[derive(Serialize, Deserialize, Debug, ToSchema)]
pub struct GetFaceResponse {
    #[schema(value_type = String, example = "[ confidence: 0.64324556, width: 124, height: 41 ]")]
    pub data: Vec<FaceResponse>,
    #[schema(value_type = String, example = "success")]
    pub message: String,
}

#[derive(Serialize, Deserialize, Debug, ToSchema)]
pub struct GetFaceResponseNone {
    #[schema(value_type = String, example = "no detections were found, please try with a better image")]
    pub message: String,
}

#[derive(MultipartForm, Debug, ToSchema)]
pub struct GetLargestFaceRequest {
    #[schema(value_type = String, example = "face.png")]
    pub input: TempFile,
}

#[derive(Serialize, Deserialize, Debug, ToSchema)]
pub struct GetLargestFaceResponse {
    pub coords: FaceResponse,
    pub cropped_face: String,
}

impl GetLargestFaceResponse {
    pub fn new(coords: FaceResponse, cropped_face: String) -> GetLargestFaceResponse {
        GetLargestFaceResponse {
            coords,
            cropped_face,
        }
    }
}
