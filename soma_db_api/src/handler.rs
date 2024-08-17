use serde::{Deserialize, Serialize};
use actix_multipart::form::MultipartForm;
use actix_multipart::Multipart;
use anyhow::{Result, Error};
use actix_multipart::form::tempfile::TempFile;
use actix_multipart::form::text::Text;


/// all them INSERTS are the same
#[derive(Debug, Serialize, Deserialize)] 
pub struct GenericResponse {
    status: i32,
    message: String
}

impl GenericResponse {
    pub fn ok() -> GenericResponse {
        GenericResponse {
            status: 200,
            message: String::from("success")
        }
    }
}

