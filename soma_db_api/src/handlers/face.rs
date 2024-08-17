use actix_multipart::form::tempfile::TempFile;
use actix_multipart::form::text::Text;
use actix_multipart::form::MultipartForm;
use actix_multipart::Multipart;
use anyhow::{Error, Result};
use pgvector::Vector;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct GetFaceByUuidRequest {
    pub face_uuid: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetFaceDetailResponse {
    pub id: i64,
    pub name: Option<String>,
    pub face_uuid: String,
    pub gender: Option<i32>,
    pub embedding: Vec<f32>,
}

/// embedding: face vector [f32; 512]
///
/// name: name of the face , optional
///
/// gender: 0 = male , 1 = female. etc.  
#[derive(Debug, Serialize, Deserialize)]
pub struct InsertFaceRequest {
    pub embedding: Vec<f32>,
    pub name: Option<String>,
    pub gender: Option<i64>,
    pub face_uuid: String,
}

impl InsertFaceRequest {
    pub fn new(
        embedding: Vec<f32>,
        name: Option<String>,
        gender: Option<i64>,
        face_uuid: String,
    ) -> InsertFaceRequest {
        InsertFaceRequest {
            embedding,
            name,
            gender,
            face_uuid,
        }
    }
}

#[derive(Debug, MultipartForm)]
pub struct GetSimilarFacesByImageRequest {
    face: TempFile,
    count: Text<i32>,
    aligned: Text<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetSimilarFacesByImageResponse {
    face_id: Vec<i32>,
}

#[derive(Debug, MultipartForm)]
pub struct GetSimilarFaceByImageRequest {
    pub input: TempFile,
    pub aligned: Text<bool>,
    pub count: Text<i32>
}


#[derive(Debug, Serialize, Deserialize)]
pub struct GetSimilarFacesByUuidRequest {
    pub face_uuid: String,
    pub count: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetSimilarFacesByUuidResponse {
    pub face: GetFaceDetailResponse,
    pub cosine_similarity: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetSimilarFacesByEmbeddingRequest {
    pub face_embedding: Vec<f32>,
    pub count: i64
}
