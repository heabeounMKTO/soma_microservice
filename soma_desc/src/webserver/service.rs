use actix_web::{get, post, web, HttpRequest, HttpResponse};
use actix_multipart::form::MultipartForm;


use actix_web::http::header::ContentType;
use anyhow::Error;
use crate::core::blip_model::BlipModel;
use candle_transformers::models::stable_diffusion::embeddings;
use dotenvy::dotenv;
use image::{image_dimensions, GenericImageView};
use crate::utils::image_utils::decode_base64;
use postgres;
use serde::{Deserialize, Serialize};
use std::env;
use crate::core::splash::print_splash;
use std::sync::Mutex;
use super::handler::{ImageDescRequest, ImageDescResponse};

#[get("/")]
pub async fn index(req: HttpRequest) -> HttpResponse {
    HttpResponse::Ok()
        .content_type(ContentType::plaintext())
        .insert_header(("X-Hdr", "sample"))
        .body("server is up :)")
}

#[post("/image_desc")]
pub async fn get_image_description(
    blip_model: web::Data<BlipModel>,
    request: web::Json<ImageDescRequest>,
    req: HttpRequest,
) -> HttpResponse {
    let image = decode_base64(&request.data).expect("cannot decode image");
    let emebeddings = blip_model
        .run(&image)
        .expect("cannot decode stream");
    let f = emebeddings.description;
    let _a = ImageDescResponse {
        data: String::from("image_dims"),
        message: f,
    };
    HttpResponse::Ok().json(_a)
}

