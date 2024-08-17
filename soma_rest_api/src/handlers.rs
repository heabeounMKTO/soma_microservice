use actix_multipart::form::tempfile::TempFile;
use actix_multipart::form::text::Text;
use actix_multipart::form::MultipartForm;
use serde::{Deserialize, Serialize};

#[derive(Debug, MultipartForm)]
pub struct AddFaceRequest {
    pub input: TempFile,
    pub aligned: Text<bool>,
}

#[derive(Debug, Serialize)]
pub struct AddFaceResponse {
    pub id: String,
    pub message: String,
}



