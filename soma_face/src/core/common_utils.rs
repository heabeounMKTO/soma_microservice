use anyhow::{Error, Result};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use image::imageops::{rotate180, rotate180_in_place, rotate90, rotate90_in};
use image::{DynamicImage, RgbaImage, Rgba, GenericImageView, imageops::FilterType};
use nalgebra::{Matrix3, Vector2};
use image::{DynamicImage,GenericImageView, ImageBuffer, Rgb};
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

    /// please run apply_image_scale first
    /// or else u will get wrong coords
    pub fn to_xywh(&self) -> Bbox {
        let cx = (self.x1 + self.x2) / 2.0;
        let cy = (self.y1 + self.y2) / 2.0;
        let width = self.x2 - self.x1;
        let height = self.y2 - self.y1;
        Bbox {
            x1: cx,
            y1: cy,
            x2: width,
            y2: height,
            confidence: self.confidence,
            kpss: self.kpss.clone(),
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
    pub fn crop_and_align_face(&self, image: &DynamicImage, desired_size: (u32, u32)) -> Result<DynamicImage,Error> {
        let left_eye = &self.kpss[0];
        let right_eye = &self.kpss[1];

        // Calculate the angle for rotation
        let dy = right_eye[1] - left_eye[1];
        let dx = right_eye[0] - left_eye[0];
        let angle = dy.atan2(dx) * 180.0 / std::f32::consts::PI;

        // Crop the face region
        let face_image = image.to_owned().crop(
            self.x1 as u32,
            self.y1 as u32,
            (self.x2 - self.x1) as u32,
            (self.y2 - self.y1) as u32,
        );

        // Rotate the face to align eyes horizontally
        let rotated_face = rotate_about_center(
            &face_image.to_rgb8(),
            angle.to_radians(),
            Interpolation::Bilinear,
            Rgb([0, 0, 0]),
        );

        // Calculate the eye center in the rotated image
        let eye_center_x = (left_eye[0] + right_eye[0]) / 2.0 - self.x1;
        let eye_center_y = (left_eye[1] + right_eye[1]) / 2.0 - self.y1;

        // Calculate the crop region
        let crop_width = desired_size.0;
        let crop_height = desired_size.1;
        let crop_x = (eye_center_x - crop_width as f32 / 2.0).max(0.0) as u32;
        let crop_y = (eye_center_y - crop_height as f32 / 3.0).max(0.0) as u32;

        // Perform final crop
        let mut final_image = ImageBuffer::new(crop_width, crop_height);
        for (x, y, pixel) in final_image.enumerate_pixels_mut() {
            let src_x = crop_x + x;
            let src_y = crop_y + y;
            if src_x < rotated_face.width() && src_y < rotated_face.height() {
                *pixel = *rotated_face.get_pixel(src_x, src_y);
            }
        }

        Ok(DynamicImage::ImageRgb8(final_image))
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

pub fn sort_conf_area_ratio_bbox(input_bbox: &mut Vec<Bbox>) -> Vec<Bbox> {
    input_bbox.sort_by(|a, b| {
        let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        let ratio_a = a.confidence / area_a;
        let ratio_b = b.confidence / area_b;
        ratio_b.partial_cmp(&ratio_a).unwrap()
    });
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

/// does rotation based on index original orientation
///
/// no rotation , 90, 180, 270 for 0,1,2,3 respectively
///
/// meant to be used when retrying detections.
pub fn rotate_image_with_index(
    image: &DynamicImage,
    rotation_index: i32,
) -> Result<DynamicImage, Error> {
    // copy the image so that you dont mess with the original image
    let mod_img = image.to_owned();
    match rotation_index {
        0 => {
            println!("returning original image");
            Ok(image.to_owned())
        }
        1 => {
            println!("doing 90deg rotation on image");
            Ok(mod_img.rotate90())
        }
        2 => {
            println!("doing 180deg rotation on image");
            Ok(mod_img.rotate180())
        }
        3 => {
            println!("doing 270deg rotation on image");
            Ok(mod_img.rotate270())
        }
        _ => {
            // returns original image if some stupid index is applied..
            // * THIS IS A SAFE GUARD *
            println!("returning original image");
            Ok(image.to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgb, RgbImage};

    #[test]
    fn test_bbox_creation_and_methods() {
        let bbox = Bbox::new(10.0, 20.0, 110.0, 120.0, 0.9, vec![vec![30.0, 40.0]]);

        assert_eq!(bbox.x1, 10.0);
        assert_eq!(bbox.y1, 20.0);
        assert_eq!(bbox.x2, 110.0);
        assert_eq!(bbox.y2, 120.0);
        assert_eq!(bbox.confidence, 0.9);
        assert_eq!(bbox.kpss, vec![vec![30.0, 40.0]]);

        // Test to_xywh
        let xywh = bbox.to_xywh();
        assert_relative_eq!(xywh.x1, 60.0); // center x
        assert_relative_eq!(xywh.y1, 70.0); // center y
        assert_relative_eq!(xywh.x2, 100.0); // width
        assert_relative_eq!(xywh.y2, 100.0); // height
    }

    #[test]
    fn test_bbox_crop() -> Result<(), Error> {
        let bbox = Bbox::new(10.0, 20.0, 110.0, 120.0, 0.9, vec![]);

        // Create an ImageBuffer first
        let mut img_buffer: RgbImage = ImageBuffer::new(200, 200);

        // Fill the image with a color
        for (x, y, pixel) in img_buffer.enumerate_pixels_mut() {
            *pixel = Rgb([x as u8, y as u8, 100]);
        }

        // Convert ImageBuffer to DynamicImage
        let image = DynamicImage::ImageRgb8(img_buffer);

        let cropped = bbox.crop_bbox(&image)?;

        assert_eq!(cropped.dimensions(), (100, 100));

        // Check the top-left pixel
        let top_left = cropped.get_pixel(0, 0);
        assert_eq!(top_left[0], 10); // R
        assert_eq!(top_left[1], 20); // G
        assert_eq!(top_left[2], 100); // B

        // Check the bottom-right pixel
        let bottom_right = cropped.get_pixel(99, 99);
        assert_eq!(bottom_right[0], 109); // R
        assert_eq!(bottom_right[1], 119); // G
        assert_eq!(bottom_right[2], 100); // B

        Ok(())
    }

    #[test]
    fn test_image_to_ndarray() {
        let image = DynamicImage::new_rgb8(2, 2);
        let array = image_to_ndarray(&image);

        assert_eq!(array.shape(), &[2, 2, 3]);
    }

    #[test]
    fn test_sort_conf_bbox() {
        let mut bboxes = vec![
            Bbox::new(0.0, 0.0, 1.0, 1.0, 0.5, vec![]),
            Bbox::new(0.0, 0.0, 1.0, 1.0, 0.9, vec![]),
            Bbox::new(0.0, 0.0, 1.0, 1.0, 0.7, vec![]),
        ];

        let sorted = sort_conf_bbox(&mut bboxes);
        assert_eq!(sorted[0].confidence, 0.9);
        assert_eq!(sorted[1].confidence, 0.7);
        assert_eq!(sorted[2].confidence, 0.5);
    }

    #[test]
    fn test_get_largest_bbox() {
        let bboxes = vec![
            Bbox::new(0.0, 0.0, 1.0, 1.0, 0.5, vec![]),
            Bbox::new(0.0, 0.0, 2.0, 2.0, 0.7, vec![]),
            Bbox::new(0.0, 0.0, 1.5, 1.5, 0.9, vec![]),
        ];

        let largest = get_largest_bbox(bboxes);
        assert_eq!(largest.x2, 2.0);
        assert_eq!(largest.y2, 2.0);
    }

    #[test]
    fn test_non_maximum_suppression() {
        let bboxes = vec![
            Bbox::new(0.0, 0.0, 2.0, 2.0, 0.9, vec![]),
            Bbox::new(0.1, 0.1, 2.1, 2.1, 0.8, vec![]),
            Bbox::new(3.0, 3.0, 4.0, 4.0, 0.7, vec![]),
        ];

        let nms_result = non_maximum_suppression(bboxes, 0.5);
        assert_eq!(nms_result.len(), 2);
        assert_eq!(nms_result[0].confidence, 0.9);
        assert_eq!(nms_result[1].confidence, 0.7);
    }

    #[test]
    fn test_calculate_iou() {
        let box1 = Bbox::new(0.0, 0.0, 2.0, 2.0, 1.0, vec![]);
        let box2 = Bbox::new(1.0, 1.0, 3.0, 3.0, 1.0, vec![]);

        let iou = calculate_iou(&box1, &box2);
        assert_relative_eq!(iou, 1.0 / 7.0, epsilon = 1e-6);
    }
}
