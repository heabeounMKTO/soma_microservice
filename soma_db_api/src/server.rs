mod handlers;
mod operators;
mod utils;

use actix_web::http::header::ContentType;
use actix_web::{middleware::Logger, web, App, HttpRequest, HttpResponse, HttpServer};
use dotenvy::dotenv;
use operators::index;
use operators::insertion::insert_face_vector;
use operators::queries::{get_face_from_uuid, get_similar_faces_by_embedding, get_similar_faces_by_uuid};
use serde::{Deserialize, Serialize};
use std::env;
use utils::db_utils::init_pool;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "actix_web=debug");
    env_logger::init();
    dotenv().ok();
    let SERVER_ADDRESS = env::var("SERVER_ADDRESS").expect("cannot read server addr");
    let SERVER_PORT = env::var("SERVER_PORT").expect("cannot read server port");
    let bind_addr = format!("{}:{}", SERVER_ADDRESS, SERVER_PORT);
    let pool = web::Data::new(init_pool().await?);
    utils::print_splash();
    println!("starting server on {:?}", &bind_addr);
    HttpServer::new(move || {
        App::new()
            .app_data(pool.clone())
            .service(web::scope("/info").route("", web::get().to(index)))
            .service(insert_face_vector)
            .service(get_face_from_uuid)
            .service(get_similar_faces_by_uuid)
            .service(get_similar_faces_by_embedding)
            .wrap(Logger::default())
    })
    .bind(&bind_addr)?
    .workers(4)
    .run()
    .await
}