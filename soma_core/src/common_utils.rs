use anyhow::{Error, Result};
use image::DynamicImage;
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::cmp::{PartialEq, PartialOrd};

/// kpss = face keypoints
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, PartialOrd)]
pub struct Bbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub kpss: Vec<Vec<f32>>,
}

impl Bbox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, confidence: f32, kpss: Vec<Vec<f32>>) -> Bbox {
        Bbox {
            x1,
            y1,
            x2,
            y2,
            confidence,
            kpss,
        }
    }
    pub fn apply_image_scale(
        &mut self,
        original_image: &DynamicImage,
        x_scale: f32,
        y_scale: f32,
    ) -> Bbox {
        let normalized_x1 = self.x1 / x_scale;
        let normalized_x2 = self.x2 / x_scale;
        let normalized_y1 = self.y1 / y_scale;
        let normalized_y2 = self.y2 / y_scale;

        let cart_x1 = original_image.width() as f32 * normalized_x1;
        let cart_x2 = original_image.width() as f32 * normalized_x2;
        let cart_y1 = original_image.height() as f32 * normalized_y1;
        let cart_y2 = original_image.height() as f32 * normalized_y2;

        Bbox {
            x1: cart_x1,
            y1: cart_y1,
            x2: cart_x2,
            y2: cart_y2,
            confidence: self.confidence,
            kpss: self.kpss.to_owned(),
        }
    }
    pub fn crop_bbox(&self, original_image: &DynamicImage) -> Result<DynamicImage, Error> {
        let bbox_width = (self.x2 - self.x1) as u32;
        let bbox_height = (self.y2 - self.y1) as u32;
        Ok(original_image.to_owned().crop_imm(
            self.x1 as u32,
            self.y1 as u32,
            bbox_width,
            bbox_height,
        ))
    }
}

pub fn image_to_ndarray(img: &DynamicImage) -> Array3<f32> {
    let height = img.height();
    let width = img.width();
    let img_buffer = img.to_rgb8();
    Array3::from_shape_vec((height as usize, width as usize, 3), img_buffer.into_raw())
        .expect("Error converting image to ndarray")
        .mapv(|x| x as f32)
}

/// sorts all bboxes based on confidence, from higher to lower
pub fn sort_conf_bbox(input_bbox: &mut Vec<Bbox>) -> Vec<Bbox> {
    input_bbox.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    input_bbox.to_vec()
}
/// you should NOT use this function if there is only one Bbox
/// (for obvious reasons -_- (THERE'S ONLY ONE , ITS THE BIGGEST ONE COME ON))
pub fn get_largest_bbox(bboxes: Vec<Bbox>) -> Bbox {
    bboxes
        .into_iter()
        .max_by(|a, b| {
            let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
            let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
            area_a
                .partial_cmp(&area_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
}
pub fn non_maximum_suppression(mut boxes: Vec<Bbox>, iou_threshold: f32) -> Vec<Bbox> {
    // Sort boxes by confidence score in descending order
    boxes.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(Ordering::Equal)
    });

    let mut keep = Vec::new();

    while !boxes.is_empty() {
        // Keep the box with the highest confidence score
        let current = boxes.remove(0);
        keep.push(current.clone());

        // Remove boxes with IoU higher than the threshold
        boxes.retain(|box_| calculate_iou(&current, box_) <= iou_threshold);
    }

    keep
}

fn calculate_iou(box1: &Bbox, box2: &Bbox) -> f32 {
    let x1 = box1.x1.max(box2.x1);
    let y1 = box1.y1.max(box2.y1);
    let x2 = box1.x2.min(box2.x2);
    let y2 = box1.y2.min(box2.y2);

    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    let area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    let union = area1 + area2 - intersection;

    intersection / union
}
