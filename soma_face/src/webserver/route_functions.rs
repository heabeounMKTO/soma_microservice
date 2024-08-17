use soma_core::common_utils::sort_conf_bbox;
use soma_core::common_utils::Bbox;
use soma_core::onnx_backend::{Inference, InferenceResult};
use crate::get_face::retinaface::GetFaceRetinaface;
use crate::get_face::yolo::GetFaceYolo;
use crate::get_face_vec::arcface::GetFaceVecArcFace;
use crate::webserver::common_utils::tempfile_to_dynimg;
use crate::webserver::handler::FaceResponse;
use crate::webserver::handler::{
    GetFaceRequest, GetFaceResponse, GetFaceVecRequest, GetFaceVecResponse,
};
use actix_multipart::form::MultipartForm;
use actix_web::http::header::ContentType;
use actix_web::{web, HttpRequest, HttpResponse};
use utoipa::OpenApi;
use utoipa_swagger_ui::{SwaggerUi, Url};

pub async fn index() -> HttpResponse {
    HttpResponse::Ok()
        .content_type(ContentType::plaintext())
        .insert_header(("X-Hdr", "sample"))
        .body("hehe")
}

pub async fn get_face_vectors(
    loaded_model: web::Data<GetFaceVecArcFace>,
    form: MultipartForm<GetFaceVecRequest>,
    req: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    println!("Request : {:?}", req);
    let get_face_vec_req = form.into_inner();
    let temp_file = get_face_vec_req.input;
    // let mut file = temp_file.file;
    let img = tempfile_to_dynimg(temp_file)?;
    let face_vec = loaded_model.forward(&img, 0.0).unwrap();
    let results = match face_vec {
        InferenceResult::FaceEmbedding(real) => real,
        _ => unreachable!(),
    };
    Ok(HttpResponse::Ok().json(GetFaceVecResponse { data: results }))
}

pub async fn get_face_bbox_yolo(
    loaded_model: web::Data<GetFaceYolo>,
    form: MultipartForm<GetFaceRequest>,
    req: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    println!("Request : {:?}", req);
    let get_face_req = form.into_inner();
    let temp_file = get_face_req.input;
    let img = tempfile_to_dynimg(temp_file)?;
    let bboxes = loaded_model.forward(&img, 0.1).unwrap();
    let mut _res = match bboxes {
        InferenceResult::FaceDetection(ayylmao) => ayylmao,
        _ => unreachable!(),
    };
    if _res.len() > 0 {
        let _a = sort_conf_bbox(&mut _res);
        let _highest = vec![_a[0].to_owned()];
        Ok(HttpResponse::Ok().json(GetFaceResponse {
            data: vec![FaceResponse::from_bbox_vec(&_highest)],
            message: String::from("success"),
        }))
    } else {
        let _highest: Vec<Bbox> = vec![];
        Ok(HttpResponse::Ok().json(GetFaceResponse {
            data: vec![FaceResponse::from_bbox_vec(&_highest)],
            message: String::from("no detections were found, please try with a better image"),
        }))
    }
}

pub async fn get_face_bbox_retinaface(
    loaded_model: web::Data<GetFaceRetinaface>,
    form: MultipartForm<GetFaceRequest>,
    req: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    println!("Request : {:?}", req);
    let get_face_req = form.into_inner();
    let temp_file = get_face_req.input;
    let img = tempfile_to_dynimg(temp_file)?;
    let bboxes = loaded_model.forward(&img, 0.1).unwrap();
    let mut _res = match bboxes {
        InferenceResult::FaceDetection(ayylmao) => ayylmao,
        _ => unreachable!(),
    };
    if _res.len() > 0 {
        let _a = sort_conf_bbox(&mut _res);
        let _highest = vec![_a[0].to_owned()];
        Ok(HttpResponse::Ok().json(GetFaceResponse {
            data: vec![FaceResponse::from_bbox_vec(&_highest)],
            message: String::from("success"),
        }))
    } else {
        let _highest: Vec<Bbox> = vec![];
        Ok(HttpResponse::Ok().json(GetFaceResponse {
            data: vec![FaceResponse {
                confidence: 0.0,
                height: 0,
                width: 0,
            }],
            message: String::from("no detections were found, please try with a better image"),
        }))
    }
}
