mod get_face;
mod get_face_vec;
mod webserver;

use actix_web::middleware::Logger;
use actix_web::{web, App, HttpServer};
use clap::Parser;
use dotenvy::dotenv;
use env_logger;
use get_face::yolo::GetFaceYolo;
use get_face::FaceExtractor;
use get_face_vec::arcface::GetFaceVecArcFace;
use std::env;
use utoipa::OpenApi;
use utoipa_swagger_ui::{SwaggerUi, Url};
use webserver::documentation::{
    GetFaceDocs, GetFaceDocsRetina, GetFaceDocsYolo, GetFaceVecDocsArcFace,
};
use webserver::service::{
    extract_face, 
    get_face_bbox_retinaface, 
    get_face_bbox_yolo,
    get_face_vectors,
    get_largest_face, index
};

#[derive(clap::Parser)]
struct CliArgs {
    /// number of workers for server
    /// defaults to 4 if none is specified
    #[arg(long)]
    workers: Option<i32>,

    /// if enabled , loads only YOLO model for face detection,
    /// defaults to true if not supplied
    #[arg(long)]
    force_yolo: Option<bool>,
    
    /// if enabled only loads face detection route and model
    /// defaults to false
    #[arg(long)]
    only_detect: Option<bool>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "actix_web=debug");

    dotenv().ok();
    let args = CliArgs::parse();

    let workers = match &args.workers {
        Some(ref _i32) => args.workers,
        None => Some(4),
    };
    let force_yolo = match &args.force_yolo {
        Some(ref _bool) => match _bool.to_owned() {
            true => {
                println!("face detector configured to use yolo only");
                args.force_yolo
            }
            false => {
                println!("face detector configured to use yolo and retinaface");
                args.force_yolo
            }
        },
        None => {
            println!("face detector configured to use yolo only");
            Some(true)
        }
    }
    .unwrap();
    


    let only_detect = match &args.only_detect {
        Some(ref _bool) => match _bool.to_owned() {
            true => {
                println!("loading face detector route");
                args.only_detect
            }
            false => {
                println!("loading face detector route");
                args.only_detect
            }
        },
        None => {
            println!("loading face detector and face vector routes");
            Some(false)
        }
    }
    .unwrap();


    let server_address = env::var("SERVER_ADDRESS").expect("cannot read server address");
    let server_port = env::var("SERVER_PORT").expect("cannot read server port");
    let bind_addr = format!("{}:{}", server_address, server_port);

    env_logger::init();
    let arcface_model_path = "./models/arcfaceresnet100-8.onnx";
    let retina_model_path = "./models/det_10g.onnx";
    let yolo_model_path = "./models/yoloface_8n.onnx";
    print_splash();
    println!(
        "starting server with {:?} workers",
        &workers.unwrap().to_owned()
    );
    HttpServer::new(move || {
        if force_yolo == false {
            match only_detect {
                true => {
                    let face_extractor = web::Data::new(
                        FaceExtractor::new(retina_model_path, yolo_model_path, 640, 640).unwrap(),
                    );
                    App::new()
                        .service(index)
                        .service(extract_face)
                        .service(get_largest_face)
                        .app_data(face_extractor)

                        // the docs section
                        .service(SwaggerUi::new("/docs/{_:.*}").urls(vec![
                            (Url::new("get_face", "/get_face"), GetFaceDocs::openapi()),
                        ]))
                        .wrap(Logger::default())
                }, 
                false => {
                    let face_arc = web::Data::new(GetFaceVecArcFace::new(arcface_model_path).unwrap());
                    let face_extractor = web::Data::new(
                        FaceExtractor::new(retina_model_path, yolo_model_path, 640, 640).unwrap(),
                    );
                    App::new()
                        .service(index)
                        .service(extract_face)
                        .service(get_largest_face)
                        .app_data(face_extractor)
                        .service(get_face_vectors)
                        .app_data(face_arc)
                        // the docs section
                        .service(SwaggerUi::new("/docs/{_:.*}").urls(vec![
                            (Url::new("get_face", "/get_face"), GetFaceDocs::openapi()),
                            (
                                Url::new("get_face_vec", "/get_vec"),
                                GetFaceVecDocsArcFace::openapi(),
                            ),
                        ]))
                        .wrap(Logger::default())
                }
            }
        } else {
            match only_detect { 
                true => { 
                    let face_extractor = web::Data::new(GetFaceYolo::new(yolo_model_path, 640, 640, false).unwrap());
                    App::new()
                        .service(index)
                        .service(get_face_bbox_yolo)
                        .service(get_largest_face)
                        .app_data(face_extractor)
                        // the docs section
                        .service(SwaggerUi::new("/docs/{_:.*}").urls(vec![
                            (
                                Url::new("get_face_yolo", "/get_face"),
                                GetFaceDocsYolo::openapi(),
                            ),
                        ]))
                        .wrap(Logger::default())
                },
                false => {
                    let face_arc = web::Data::new(GetFaceVecArcFace::new(arcface_model_path).unwrap());
                    let face_extractor =
                        web::Data::new(GetFaceYolo::new(yolo_model_path, 640, 640, false).unwrap());
                    App::new()
                        .service(index)
                        .service(get_face_bbox_yolo)
                        .service(get_largest_face)
                        .app_data(face_extractor)
                        .service(get_face_vectors)
                        .app_data(face_arc)
                        // the docs section
                        .service(SwaggerUi::new("/docs/{_:.*}").urls(vec![
                            (
                                Url::new("get_face_yolo", "/get_face"),
                                GetFaceDocsYolo::openapi(),
                            ),
                            (
                                Url::new("get_face_vec", "/get_vec"),
                                GetFaceVecDocsArcFace::openapi(),
                            ),
                        ]))
                        .wrap(Logger::default())
                }
            }
        }
    })
    .client_request_timeout(std::time::Duration::from_secs(0))
    .keep_alive(None)
    .bind(&bind_addr)?
    .workers(workers.unwrap() as usize)
    .run()
    .await
}



fn print_splash() {
let splash: &str = r#"
            .▄▄ ·       • ▌ ▄ ·.  ▄▄▄· 
            ▐█ ▀. ▪     ·██ ▐███▪▐█ ▀█ 
            ▄▀▀▀█▄ ▄█▀▄ ▐█ ▌▐▌▐█·▄█▀▀█ 
            ▐█▄▪▐█▐█▌.▐▌██ ██▌▐█▌▐█ ▪▐▌
             ▀▀▀▀  ▀█▄▀▪▀▀  █▪▀▀▀ ▀  ▀ 
            ·▄▄▄ ▄▄▄·  ▄▄· ▄▄▄ .       
            ▐▄▄·▐█ ▀█ ▐█ ▌▪▀▄.▀·       
            ██▪ ▄█▀▀█ ██ ▄▄▐▀▀▪▄       
            ██▌.▐█ ▪▐▌▐███▌▐█▄▄▌       
            ▀▀▀  ▀  ▀ ·▀▀▀  ▀▀▀        
"#;
    println!("soma_service's");
    println!("{}", splash);
}



