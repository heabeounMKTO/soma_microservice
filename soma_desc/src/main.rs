mod core;
mod utils;
mod webserver;

use webserver::service::{get_image_description, index};
use actix_web::http::header::ContentType;
use actix_web::{web, App, HttpMessage, HttpRequest, HttpResponse, HttpServer};
use anyhow::Error;
use core::blip_model::BlipModel;
use candle_transformers::models::stable_diffusion::embeddings;
use dotenvy::dotenv;
use image::{image_dimensions, GenericImageView};
use utils::image_utils::decode_base64;
use postgres;
use serde::{Deserialize, Serialize};
use std::env;
use core::splash::print_splash;
use std::sync::Mutex;
#[derive(Serialize, Deserialize)]
pub struct ImageDescRequest {
    data: String,
}

#[derive(Serialize, Deserialize)]
pub struct ImageDescResponse {
    data: String,
    message: String,
}


async fn model_info(blip_model: web::Data<BlipModel>, req: HttpRequest) -> HttpResponse {
    HttpResponse::Ok()
        .content_type(ContentType::plaintext())
        .body("placeholder")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "actix_web=debug");
    env_logger::init();
    dotenv().ok();
    print_splash();
    let DB_HOST = env::var("DB_HOST").expect("cannot read DB_HOST");
    let DB_PORT = env::var("DB_PORT").expect("cannot read DB_PORT");
    let DB_USER = env::var("DB_USER").expect("cannot read DB_USER");
    let DB_PASSWORD = env::var("DB_PASSWORD").expect("cannot read DB_PASSWORD");
    let DB_DATABASE = env::var("DB_DATABASE").expect("cannot read DB_DATABASE");

    let SERVER_ADDRESS = env::var("SERVER_ADDRESS").expect("cannot read server addr");
    let SERVER_PORT = env::var("SERVER_PORT").expect("cannot read server port");
    let bind_addr = format!("{}:{}", SERVER_ADDRESS, SERVER_PORT);

    let postgres_addr: String = format!(
        "host={} user={} port={} dbname={} password={}",
        &DB_HOST, &DB_USER, &DB_PORT, &DB_DATABASE, &DB_PASSWORD
    );

    println!("starting server on address: {:?}", &bind_addr);
    HttpServer::new(move || {
        let blip_model = web::Data::new(
            BlipModel::init(candle_core::Device::new_cuda(0).unwrap()).unwrap(),
        );

        App::new()
            .service(index)
            .service(get_image_description)
            .app_data(blip_model)
    })
    .client_request_timeout(std::time::Duration::from_secs(0))
    .bind(&bind_addr)?
    .workers(1)
    .run()
    .await?;
    Ok(())
}
