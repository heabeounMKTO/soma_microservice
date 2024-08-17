pub mod insertion;
pub mod queries;

use actix_web::{http::header::ContentType, HttpResponse};

pub async fn index() -> HttpResponse {
    HttpResponse::Ok()
        .content_type(ContentType::plaintext())
        .insert_header(("X-Hdr", "sample"))
        .body("hehe")
}
