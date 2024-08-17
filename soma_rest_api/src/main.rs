mod common_utils;
mod handlers;
mod service;

use crate::service::index;
use actix_web::http::header::ContentType;
use actix_web::middleware::Logger;
use actix_web::{web, App, HttpMessage, HttpRequest, HttpResponse, HttpServer};
use dotenvy::dotenv;
use env_logger;
use lazy_static::lazy_static;
use service::{add_face_vec, get_similar_faces_uuid, get_similar_faces_image};
use std::env;

lazy_static! {
    /// WE BEING EXTRA LAZY WITH THIS ONE
    ///
    static ref GET_FACE_VEC_URL: String = {
        dotenv().ok();
        let addr = env::var("FACE_API_ADDRESS").expect("api addr not found!");
        format!("{}/get_vec", addr)
    };

    static ref GET_LARGEST_FACE_URL: String = {
        dotenv().ok();
        let addr = env::var("FACE_API_ADDRESS").expect("api addr not found!");
        format!("{}/get_largest_face", addr)
    };
    static ref GET_FACE_URL: String = {
        dotenv().ok();
        let addr = env::var("FACE_API_ADDRESS").expect("api addr not found!");
        format!("{}/get_face", addr)
    };

    static ref DB_API_POSTFACE_URL: String = {
        dotenv().ok();
        let addr = env::var("DB_API_ADDRESS").expect("db api addr not found");
        format!("{}/post_face_vec", addr)
    };
    
    static ref DB_API_SIMILAR_FACE_UUID_URL: String = {
        dotenv().ok();
        let addr = env::var("DB_API_ADDRESS").expect("db api addr not found");
        format!("{}/get_similar_faces_by_uuid", addr)
    };

    static ref DB_API_SIMILAR_FACE_IMAGE_URL: String = {
        dotenv().ok();
        let addr = env::var("DB_API_ADDRESS").expect("db api addr not found");
        format!("{}/get_similar_faces_by_embedding", addr)
    };
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();
    std::env::set_var("RUST_LOG", "actix_web=debug");
    env_logger::init();
    print_splash();
    HttpServer::new(move || 
        App::new()
        .service(index)
        .service(add_face_vec)
        .service(get_similar_faces_uuid) 
        .service(get_similar_faces_image)
        )
        .keep_alive(None)
        .bind("0.0.0.0:9995")?
        .run()
        .await?;
    Ok(())
}

fn print_splash() {
    const splash: &str = r#"
            .▄▄ ·       • ▌ ▄ ·.  ▄▄▄·     
            ▐█ ▀. ▪     ·██ ▐███▪▐█ ▀█     
            ▄▀▀▀█▄ ▄█▀▄ ▐█ ▌▐▌▐█·▄█▀▀█     
            ▐█▄▪▐█▐█▌.▐▌██ ██▌▐█▌▐█ ▪▐▌    
             ▀▀▀▀  ▀█▄▀▪▀▀  █▪▀▀▀ ▀  ▀     
            ▄▄▄  ▄▄▄ ..▄▄ · ▄▄▄▄▄          
            ▀▄ █·▀▄.▀·▐█ ▀. •██            
            ▐▀▀▄ ▐▀▀▪▄▄▀▀▀█▄ ▐█.▪          
            ▐█•█▌▐█▄▄▌▐█▄▪▐█ ▐█▌·          
            .▀  ▀ ▀▀▀  ▀▀▀▀  ▀▀▀           
    "#;

    println!("soma_service's");
    println!("{}", splash);
}