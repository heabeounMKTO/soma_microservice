use serde::{Deserialize, Serialize};
use anyhow::{Result, Error};
use soma_core::common_utils::{sort_conf_bbox, Bbox};

#[derive(Debug, Serialize, Deserialize)]
pub struct FaceResponse {
    confidence: f32,
    height: i32,
    width: i32,
}

/// returns width and height from bbox 
fn get_wh(input_bbox: &Bbox) -> (i32, i32) {
    let width = input_bbox.x2 - input_bbox.x1;
    let height = input_bbox.y2 - input_bbox.y1;
    (width as i32, height as i32)
}

impl FaceResponse {
    /// takes in a vec of bbox
    pub fn from_bbox_vec(input_vec: &Vec<Bbox>) -> FaceResponse {
        let _highest: Bbox = input_vec[0].to_owned(); 
        let (w, h) = get_wh(&_highest); 
        FaceResponse{
            confidence: _highest.confidence,
            width: w,
            height: h 
        }
    }
}

