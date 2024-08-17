use crate::common_utils::print_splash;
use crate::common_utils::{base64_to_bytes, base64_to_tempfile, tempfile_to_dynimg};
use crate::handlers::{AddFaceRequest, AddFaceResponse};
use crate::{DB_API_POSTFACE_URL, DB_API_SIMILAR_FACE_UUID_URL, GET_FACE_URL, GET_FACE_VEC_URL, GET_LARGEST_FACE_URL, DB_API_SIMILAR_FACE_IMAGE_URL};
use actix_multipart::form::MultipartForm;
use actix_web;
use actix_web::http::header::ContentType;
use actix_web::rt::spawn;
use actix_web::{get, post, web, HttpRequest, HttpResponse};
use anyhow::{Error, Result};
use reqwest::blocking::multipart;
use reqwest::{Client, ClientBuilder, Response};
use serde::{Deserialize, Serialize};
use soma_db_api::handlers::GenericResponse;
use tempfile::NamedTempFile;
use tokio::task::spawn_blocking;
use uuid::Uuid;

// im not writing the structs once more =))
use soma_db_api::handlers::face::{ GetSimilarFacesByUuidRequest,  GetSimilarFacesByUuidResponse, InsertFaceRequest, GetSimilarFacesByEmbeddingRequest, GetSimilarFaceByImageRequest};
use soma_face::webserver::handler::{
    GetFaceResponse, GetFaceResponseNone, GetFaceVecResponse, GetLargestFaceResponse,
};

#[get("/")]
async fn index() -> HttpResponse {
    HttpResponse::Ok()
        .content_type(ContentType::plaintext())
        .insert_header(("X-Hdr", "sample"))
        .body("haro")
}

fn get_face_vec_from_named_temp_file(
    input_file: NamedTempFile,
    client: reqwest::blocking::Client,
    aligned: bool,
) -> Result<GetFaceVecResponse> {
    let text_str = match aligned {
        true => String::from("true"),
        _ => String::from("false"),
    };
    let form = multipart::Form::new()
        .text("aligned", text_str)
        .file("input", input_file)
        .unwrap();
    let response = client
        .post(GET_FACE_VEC_URL.clone())
        .multipart(form)
        .send()
        .unwrap();
    let lmao = match response.json() {
        Ok(resp) => resp,
        Err(e) => unreachable!(),
    };
    Ok(lmao)
}

fn get_face_vec_from_tempfile(
    input_tempfile: actix_multipart::form::tempfile::TempFile,
    client: reqwest::blocking::Client,
    aligned: bool,
) -> Result<GetFaceVecResponse> {
    let text_str = match aligned {
        true => String::from("true"),
        _ => String::from("false"),
    };
    let form = multipart::Form::new()
        .text("aligned", text_str)
        .file("input", input_tempfile.file)
        .unwrap();
    let response = client
        .post(GET_FACE_VEC_URL.clone())
        .multipart(form)
        .send()
        .unwrap();
    let lmao: GetFaceVecResponse = response.json().unwrap();
    Ok(lmao)
}

fn get_face_from_tempfile(
    input_tempfile: actix_multipart::form::tempfile::TempFile,
    client: reqwest::blocking::Client,
) -> Result<GetFaceResponse> {
    let form = multipart::Form::new()
        .file("input", input_tempfile.file)
        .unwrap();
    let response = client
        .post(GET_FACE_URL.clone())
        .multipart(form)
        .send()
        .unwrap();
    let lmao: GetFaceResponse = response.json().unwrap();
    Ok(lmao)
}

fn get_biggest_face_from_tempfile(
    input_tempfile: actix_multipart::form::tempfile::TempFile,
    client: reqwest::blocking::Client,
) -> Result<GetLargestFaceResponse> {
    let form = multipart::Form::new()
        .file("input", input_tempfile.file)
        .unwrap();
    let response = client
        .post(GET_LARGEST_FACE_URL.clone())
        .multipart(form)
        .send()
        .unwrap();
    let lmao: GetLargestFaceResponse = response.json().unwrap();
    Ok(lmao)
}

#[post("/get_similar_faces_uuid")]
pub async fn get_similar_faces_uuid(
    form: web::Json<GetSimilarFacesByUuidRequest>,
    req: HttpRequest
) -> actix_web::Result<HttpResponse> {
    let client = reqwest::Client::new();
    let similar_faces_req = client.post(DB_API_SIMILAR_FACE_UUID_URL.clone()).json(&form).send().await.unwrap(); 
    if similar_faces_req.status().is_success() {
        let res: Vec<GetSimilarFacesByUuidResponse> = similar_faces_req.json().await.unwrap();
        // let fucc: Vec<GetSimilarFacesByUuidResponse> = vec![];
        Ok(HttpResponse::Ok().json(res))
    } else {
        Ok(HttpResponse::InternalServerError().json(GenericResponse {
            status: 500,
            message: String::from("there is an error getting similar faces")
        }))
    }
}

#[post("/get_similar_faces_image")]
pub async fn get_similar_faces_image(
    form: MultipartForm<GetSimilarFaceByImageRequest>,
    _: HttpRequest
) -> actix_web::Result<HttpResponse> {
    let read_form = form.into_inner();
    let temp_file = read_form.input;
    let align: bool = read_form.aligned.into_inner();
    println!("IS ALIGNED {:?}", &align);
    let instance_uuid = String::from(Uuid::new_v4());
    let db_client = reqwest::Client::new();
    let _resp: Response = match align {
        false => {
            // creates a bloking client to get faces
            // idk how else we can do to send multipart
            let largest_face = spawn_blocking(|| {
                let client = reqwest::blocking::Client::new();
                let biggest_face = get_biggest_face_from_tempfile(temp_file, client).unwrap();
                biggest_face
            })
            .await
            .unwrap();

            let read_face = base64_to_tempfile(&largest_face.cropped_face).unwrap();

            let face_vec: GetFaceVecResponse = spawn_blocking(|| {
                let client = reqwest::blocking::Client::new();
                get_face_vec_from_named_temp_file(read_face, client, true).unwrap()
            })
            .await
            .unwrap();
            let get_similar_face_by_image_request = GetSimilarFacesByEmbeddingRequest {
                face_embedding: face_vec.data,
                count: read_form.count.into_inner() as i64
            }; 
            let _resp = db_client
            .post(DB_API_SIMILAR_FACE_IMAGE_URL.clone())
                .json(&get_similar_face_by_image_request)
                .send()
                .await
            .unwrap();
                _resp
        }
        true => {
            // just raw dogging the vec detections bro
            let face_vec = spawn_blocking(|| {
                let client = reqwest::blocking::Client::new();
                get_face_vec_from_tempfile(temp_file, client, true).unwrap()
            })
            .await
            .unwrap();
            let get_similar_face_by_image_request = GetSimilarFacesByEmbeddingRequest {
                face_embedding: face_vec.data,
                count: read_form.count.into_inner() as i64
            }; 
            let _resp = db_client
            .post(DB_API_SIMILAR_FACE_IMAGE_URL.clone())
                .json(&get_similar_face_by_image_request)
                .send()
                .await
            .unwrap();
                _resp
        }
    };
    // TODO: change to proper response , am too sleepy
    if _resp.status().is_success() {
        let res: Vec<GetSimilarFacesByUuidResponse> = _resp.json().await.unwrap();
        // let fucc: Vec<GetSimilarFacesByUuidResponse> = vec![];
        Ok(HttpResponse::Ok().json(res))
    } else {
        Ok(HttpResponse::InternalServerError().into())
    }
}


/// adds face vector to database, returns uuid
#[post("/add_face")]
pub async fn add_face_vec(
    form: MultipartForm<AddFaceRequest>,
    req: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    println!("REQ {:?}", &req);
    let read_form = form.into_inner();
    let temp_file = read_form.input;
    let align: bool = read_form.aligned.into_inner();
    println!("IS ALIGNED {:?}", &align);
    let instance_uuid = String::from(Uuid::new_v4());
    let db_client = reqwest::Client::new();
    let _resp: Response = match align {
        false => {
            // creates a bloking client to get faces
            // idk how else we can do to send multipart
            let largest_face = spawn_blocking(|| {
                let client = reqwest::blocking::Client::new();
                let biggest_face = get_biggest_face_from_tempfile(temp_file, client).unwrap();
                biggest_face
            })
            .await
            .unwrap();

            let read_face = base64_to_tempfile(&largest_face.cropped_face).unwrap();

            let face_vec: GetFaceVecResponse = spawn_blocking(|| {
                let client = reqwest::blocking::Client::new();
                get_face_vec_from_named_temp_file(read_face, client, true).unwrap()
            })
            .await
            .unwrap();
            // println!("face_vec {:?}", face_vec);
            let insert_face_request = InsertFaceRequest::new(
                face_vec.data,
                Some("placeholder".to_string()),
                Some(1),
                String::from(&instance_uuid),
            );
            let _resp = db_client
                .post(DB_API_POSTFACE_URL.clone())
                .json(&insert_face_request)
                .send()
                .await
                .unwrap();
            _resp
        }
        true => {
            // just raw dogging the vec detections bro
            let face_vec = spawn_blocking(|| {
                let client = reqwest::blocking::Client::new();
                get_face_vec_from_tempfile(temp_file, client, true).unwrap()
            })
            .await
            .unwrap();
            let insert_face_request = InsertFaceRequest::new(
                face_vec.data,
                Some("placeholder".to_string()),
                Some(1),
                String::from(&instance_uuid),
            );
            let _resp = db_client
                .post(DB_API_POSTFACE_URL.clone())
                .json(&insert_face_request)
                .send()
                .await
                .unwrap();
            _resp
        }
    };
    if _resp.status().is_success() {
        Ok(HttpResponse::Ok().json(AddFaceResponse {
            id: instance_uuid,
            message: String::from("success"),
        }))
    } else {
        Ok(HttpResponse::InternalServerError().into())
    }
}