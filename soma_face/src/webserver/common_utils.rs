use crate::{
    core::{
        common_utils::{sort_conf_bbox, Bbox},
        onnx_backend::{Inference, InferenceResult},
    },
    get_face::yolo::GetFaceYolo,
};
use actix_multipart::form::tempfile::TempFile;
use actix_web::web::Bytes;
use anyhow::{Error, Result};
use base64::{engine::general_purpose, Engine as _};
use image::{DynamicImage, ImageBuffer, ImageFormat, ImageReader, Rgb};

#[cfg(feature = "readvid")]
use opencv::{
    core::{Mat, Vector},
    imgcodecs::imencode,
    imgproc::{self, cvt_color},
    prelude::*,
    videoio::{self, VideoCapture, CAP_ANY},
};
use std::fmt::Write;
use std::io::{Cursor, Read};

/// cropped DynamicImage and its' coords,
///
/// we are not using GetLargestFaceResponse because we dont want to store big ass strings aight
#[derive(Clone, Debug)]
pub struct CropImagePair {
    pub cropped: DynamicImage,
    pub bbox: Bbox,
}

pub struct MultipartImageResponse {
    boundary: String,
    fields: Vec<(String, Bytes, Option<String>, Option<String>)>,
}

impl MultipartImageResponse {
    pub fn new() -> Self {
        Self {
            boundary: "----WebKitFormBoundary7MA4YWxkTrZu0gW".to_string(),
            fields: Vec::new(),
        }
    }

    pub fn add_field(
        &mut self,
        name: &str,
        data: Vec<u8>,
        filename: Option<&str>,
        content_type: Option<&str>,
    ) {
        self.fields.push((
            name.to_string(),
            Bytes::from(data),
            filename.map(|s| s.to_string()),
            content_type.map(|s| s.to_string()),
        ));
    }

    pub fn to_string(&self) -> String {
        let mut result = String::new();
        for (name, data, filename, content_type) in &self.fields {
            writeln!(&mut result, "--{}", self.boundary).unwrap();
            write!(
                &mut result,
                "Content-Disposition: form-data; name=\"{}\"",
                name
            )
            .unwrap();
            if let Some(filename) = filename {
                write!(&mut result, "; filename=\"{}\"", filename).unwrap();
            }
            writeln!(&mut result).unwrap();
            if let Some(content_type) = content_type {
                writeln!(&mut result, "Content-Type: {}", content_type).unwrap();
            }
            writeln!(&mut result).unwrap();
            result.push_str(&String::from_utf8_lossy(data));
            writeln!(&mut result).unwrap();
        }
        writeln!(&mut result, "--{}--", self.boundary).unwrap();
        result
    }
}

/// turns a [TempFile] from [actix_multipart::Multipart] into a [DynamicImage]
pub fn tempfile_to_dynimg(input_tempfile: TempFile) -> actix_web::Result<DynamicImage> {
    let mut file = input_tempfile.file;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let img = ImageReader::new(Cursor::new(buffer))
        .with_guessed_format()?
        .decode()
        .unwrap();
    Ok(img)
}

pub fn dynimg_to_bytes(input_img: &DynamicImage) -> Vec<u8> {
    let mut img_bytes: Vec<u8> = Vec::new();
    input_img
        .write_to(&mut Cursor::new(&mut img_bytes), image::ImageFormat::Png)
        .unwrap();
    img_bytes
}

pub fn image_to_base64(img: &DynamicImage) -> Result<String, Box<dyn std::error::Error>> {
    let mut buffer = Cursor::new(Vec::new());
    img.write_to(&mut buffer, ImageFormat::Png)?;
    let bytes = buffer.into_inner();
    let base64_string = general_purpose::STANDARD.encode(bytes);
    Ok(base64_string)
}
/// converts goofy ahh cv::Mat to
/// shitty image::DynamicImage
/// (for inference of course)
///
#[cfg(feature = "readvid")]
fn mat_to_dynamic_image(mat: &Mat) -> Result<DynamicImage, Error> {
    let rows = mat.rows();
    let cols = mat.cols();
    let channels = mat.channels();
    assert!(channels == 3, "Only 3-channel (RGB) images are supported");
    let mut img_buf: Vec<u8> = vec![0; (rows * cols * channels) as usize];
    let mat_slice = mat.data_bytes().unwrap();
    img_buf.copy_from_slice(mat_slice);
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(cols as u32, rows as u32, img_buf).unwrap();
    Ok(DynamicImage::ImageRgb8(img))
}

/// read from path and get best detection/image pairs for each frame (when there is detections)
/// "MEMORY EFFICIENT" AHH FUNCTION
/// TODO: fix mem leak fuck
///
#[cfg(feature = "readvid")]
pub fn read_video_from_path_and_get_detections(
    video_path: &str,
    loaded_model: actix_web::web::Data<GetFaceYolo>,
) -> Result<Vec<CropImagePair>, Error> {
    let mut cap = VideoCapture::from_file(video_path, opencv::videoio::CAP_ANY)?;
    let mut best_pairs: Vec<CropImagePair> = vec![];
    loop {
        let opened = videoio::VideoCapture::is_opened(&cap)?;
        if !opened {
            break;
        }
        let mut frame = Mat::default();
        let _a = cap.read(&mut frame)?;

        // frame vibe check smh
        // skips if no good
        if _a != false {
            let mut rgb_image = Mat::default();
            cvt_color(&frame, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0)?;
            if rgb_image.size()?.width > 0 && rgb_image.size()?.height > 0 {
                let mat2dyn = mat_to_dynamic_image(&rgb_image)?;
                let mut _inference_results = match loaded_model.forward(&mat2dyn, 0.7).unwrap() {
                    InferenceResult::FaceDetection(res) => res,
                    _ => panic!(),
                };

                // process detections if there is any
                if _inference_results.len() > 0 {
                    _inference_results = sort_conf_bbox(&mut _inference_results);
                    // select the one with the higest confidence from frame
                    let best = CropImagePair {
                        cropped: _inference_results[0]
                            .to_owned()
                            .crop_bbox(&mat2dyn)
                            .unwrap(),
                        bbox: _inference_results[0].to_owned(),
                    };
                    best_pairs.push(best);
                } else {
                    continue;
                }
            }
        } else {
            break;
        }
    }
    let _ = videoio::VideoCapture::release(&mut cap)?;
    Ok(best_pairs)
}

/// read from path and convert frames to DynamicImage,
///
#[cfg(feature = "readvid")]
pub fn read_video_from_path_and_convert_frames(
    video_path: &str,
) -> Result<Vec<DynamicImage>, Error> {
    let mut cap = VideoCapture::from_file(video_path, opencv::videoio::CAP_ANY)?;
    let mut frame_vec: Vec<DynamicImage> = vec![];
    loop {
        let opened = videoio::VideoCapture::is_opened(&cap)?;
        if !opened {
            break;
        }
        let mut frame = Mat::default();
        let _a = cap.read(&mut frame)?;
        if _a != false {
            let mut rgb_image = Mat::default();
            cvt_color(&frame, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0)?;
            if rgb_image.size()?.width > 0 && rgb_image.size()?.height > 0 {
                let mat2dyn = mat_to_dynamic_image(&rgb_image)?;
                frame_vec.push(mat2dyn);
            }
        } else {
            break;
        }
    }
    let _ = videoio::VideoCapture::release(&mut cap)?;
    Ok(frame_vec)
}

#[cfg(feature = "readvid")]
pub fn get_total_frame_from_video(video_path: &str) -> Result<i32, Error> {
    let mut cap = VideoCapture::from_file(video_path, CAP_ANY)?;
    let total_frames = cap.get(opencv::videoio::CAP_PROP_FRAME_COUNT)? as u32;
    videoio::VideoCapture::release(&mut cap)?;
    Ok(total_frames as i32)
}

#[cfg(feature = "readvid")]
pub fn capture_frames_at_intervals(
    video_path: &str,
    num_frames: u32,
) -> Result<Vec<DynamicImage>, Error> {
    let mut cap = VideoCapture::from_file(video_path, CAP_ANY)?;
    let total_frames = cap.get(opencv::videoio::CAP_PROP_FRAME_COUNT)? as u32;
    let interval = total_frames / (num_frames + 1);
    let mut frames = Vec::new();

    for i in 1..num_frames {
        let frame_pos = i * interval;
        cap.set(opencv::videoio::CAP_PROP_POS_FRAMES, frame_pos as f64)?;

        let mut frame = Mat::default();
        if cap.read(&mut frame)? {
            let mut rgb_image = Mat::default();
            cvt_color(&frame, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0)?;
            frames.push(mat_to_dynamic_image(&rgb_image)?);
        } else {
            break;
        }
    }
    let _ = videoio::VideoCapture::release(&mut cap)?;
    Ok(frames)
}
#[cfg(feature = "readvid")]
/// does it on a total frames count thing.
pub fn capture_frames_at_interval_and_detect(
    video_path: &str,
    loaded_model: actix_web::web::Data<GetFaceYolo>,
    num_frames: u32,
) -> Result<Vec<CropImagePair>, Error> {
    let mut cap = VideoCapture::from_file(video_path, CAP_ANY)?;
    let total_frames = cap.get(opencv::videoio::CAP_PROP_FRAME_COUNT)? as u32;
    let interval = total_frames / (num_frames + 1);
    let mut best_pairs = Vec::new();

    for i in 1..=num_frames {
        let frame_pos = i * interval;
        cap.set(opencv::videoio::CAP_PROP_POS_FRAMES, frame_pos as f64)?;

        let mut frame = Mat::default();
        if cap.read(&mut frame)? {
            let mut rgb_image = Mat::default();
            cvt_color(&frame, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0)?;
            if rgb_image.size()?.width > 0 && rgb_image.size()?.height > 0 {
                let mat2dyn = mat_to_dynamic_image(&rgb_image)?;
                let mut _inference_results = match loaded_model.forward(&mat2dyn, 0.7).unwrap() {
                    InferenceResult::FaceDetection(res) => res,
                    _ => panic!(),
                };

                // process detections if there is any
                if _inference_results.len() > 0 {
                    _inference_results = sort_conf_bbox(&mut _inference_results);
                    // select the one with the higest confidence from frame
                    let best = CropImagePair {
                        cropped: _inference_results[0]
                            .to_owned()
                            .crop_bbox(&mat2dyn)
                            .unwrap(),
                        bbox: _inference_results[0].to_owned(),
                    };
                    best_pairs.push(best);
                } else {
                    continue;
                }
            }
        } else {
            break;
        }
    }
    Ok(best_pairs)
}
