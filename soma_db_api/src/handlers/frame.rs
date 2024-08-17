use serde::{Deserialize, Serialize};
/// Inserts tagged image from `soma_desc`
///
/// # Arguments
/// * `filename` - filename of the saved image in the image server
///
/// * `original_filename` - original file of the image , in case the image is a frame from a video or some shit
///
/// * `original_type` - original format (jpeg? png? mp4?)
///
/// * `face_count` - the faces in the image, if there is any
///
/// * `frame_tags` - tags from `soma_desc`
#[derive(Debug, Serialize, Deserialize)]
pub struct InsertTaggedImageRequest {
    filename: String,
    original_filename: Option<String>,
    original_type: i32,
    face_count: i32,
    frame_tags: Vec<String>,
}
