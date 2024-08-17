//! face services compilation

use soma_core::common_utils::{get_largest_bbox,sort_conf_bbox};
use soma_core::common_utils::Bbox;
use soma_core::onnx_backend::{Inference, InferenceResult};
use crate::get_face::retinaface::GetFaceRetinaface;
use crate::get_face::yolo::GetFaceYolo;
use crate::get_face::FaceExtractor;
use crate::get_face_vec::arcface::GetFaceVecArcFace;
use crate::webserver::common_utils::{tempfile_to_dynimg, dynimg_to_bytes, image_to_base64};
use crate::webserver::handler::FaceResponse;
use crate::webserver::handler::{
    GetFaceRequest, GetFaceResponse, GetFaceResponseNone, GetLargestFaceResponse, GetLargestFaceRequest, GetFaceVecRequest,GetFaceVecResponse,
};
use image::{DynamicImage};
use actix_multipart::form::MultipartForm;
use actix_web::http::header::ContentType;
use actix_web::{get, post, web, HttpRequest, HttpResponse};
use utoipa::OpenApi;
use utoipa_swagger_ui::{SwaggerUi, Url};

#[get("/")]
pub async fn index(req: HttpRequest) -> HttpResponse {
    HttpResponse::Ok()
        .content_type(ContentType::plaintext())
        .insert_header(("X-Hdr", "sample"))
        .body("server is up :)")
}

/// get face vector with `ArcFace`
/// this one always returns something
/// regardless of accuracy lmao.
/// please "align" (use cropped faces) for the best results
#[utoipa::path(
    context_path="",
    request_body(content = GetFaceVecRequest, content_type="multipart/form-data"),
    responses(
        (status=200, description="returns face vectors (latent) using `arcface` model, returns results regardless of if face is aligned or not so please use a aligned (cropped) face", body=GetFaceVecResponse),
    )
)]
#[post("/get_vec")]
pub async fn get_face_vectors(
    loaded_model: web::Data<GetFaceVecArcFace>,
    form: MultipartForm<GetFaceVecRequest>,
    req: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    let get_face_vec_req = form.into_inner();
    let temp_file = get_face_vec_req.input;
    // let mut file = temp_file.file;
    let img = tempfile_to_dynimg(temp_file)?;
    let t1 = std::time::Instant::now();
    let face_vec = loaded_model.forward(&img, 0.0).unwrap();
    println!("inference time {:?}", t1.elapsed());
    // confidence field is literally not used  ^
    let results = match face_vec {
        InferenceResult::FaceEmbedding(real) => real,
        _ => unreachable!(),
    };
    Ok(HttpResponse::Ok().json(GetFaceVecResponse { data: results }))
}

#[utoipa::path(
    context_path="",
    request_body(content = GetFaceRequest, content_type="multipart/form-data"),
    responses(
        (status=200, description="returns face detections using `yolov8` model", body=GetFaceResponse),
    )
)]
#[post("/get_face")]
pub async fn get_face_bbox_yolo(
    loaded_model: web::Data<GetFaceYolo>,
    form: MultipartForm<GetFaceRequest>,
    req: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    let get_face_req = form.into_inner();
    let temp_file = get_face_req.input;
    let img = tempfile_to_dynimg(temp_file)?;
    let t1 = std::time::Instant::now();
    let bboxes = loaded_model.forward(&img, 0.1).unwrap();
    println!("inference time {:?}", t1.elapsed());
    let mut _res = match bboxes {
        InferenceResult::FaceDetection(ayylmao) => ayylmao,
        _ => unreachable!(),
    };
    if _res.len() > 0 {
        let _a = sort_conf_bbox(&mut _res);
        Ok(HttpResponse::Ok().json(GetFaceResponse {
            data: FaceResponse::from_bbox_vec(&_a),
            message: String::from("success"),
        }))
    } else {
        Ok(HttpResponse::Ok().json(GetFaceResponseNone {
            message: String::from("no detections were found, please try with a better image!"),
        }))
    }
}

#[utoipa::path(
    context_path="",
    request_body(content = GetFaceRequest, content_type="multipart/form-data"),
    responses(
        (status=200, description="returns face detections using `retinaface_10g` model", body=GetFaceResponse),
    )
)]
#[post("/get_face/retina")]
pub async fn get_face_bbox_retinaface(
    loaded_model: web::Data<GetFaceRetinaface>,
    form: MultipartForm<GetFaceRequest>,
    req: HttpRequest,
) -> actix_web::Result<HttpResponse> {
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
        Ok(HttpResponse::Ok().json(GetFaceResponse {
            data: FaceResponse::from_bbox_vec(&_a),
            message: String::from("success"),
        }))
    } else {
        Ok(HttpResponse::Ok().json(GetFaceResponseNone {
            message: String::from("no detections were found, please try with a better image!"),
        }))
    }
}

#[utoipa::path(
    context_path="",
    request_body(content = GetFaceRequest, content_type="multipart/form-data"),
    responses(
        (status=200, description="returns face detections with highest confidence using `retinaface_10g` and `yolov8` model", body=GetFaceResponse),
    )
)]
#[post("/get_face")]
pub async fn extract_face(
    loaded_model: web::Data<FaceExtractor>,
    form: MultipartForm<GetFaceRequest>,
    req: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    let get_face_req = form.into_inner();
    let temp_file = get_face_req.input;
    let img = tempfile_to_dynimg(temp_file)?;
    let t1 = std::time::Instant::now();
    let bboxes = loaded_model.extract_face_from_image(&img, 0.1);
    println!("inference time {:?}", t1.elapsed());
    let _res = match bboxes {
        Ok(results) => results,
        Err(_) => {
           vec![]
        }
    };
    if _res.len() > 0 {
        Ok(HttpResponse::Ok().json(GetFaceResponse {
            data: FaceResponse::from_bbox_vec(&_res),
            message: String::from("sucess"),
        }))
    } else {
        Ok(HttpResponse::Ok().json(GetFaceResponseNone {
            message: String::from("no detections were found, please try with a better image!"),
        }))
    }
}

#[post("/get_largest_face")]
pub async fn get_largest_face(
    loaded_model: web::Data<GetFaceYolo>,
    form: MultipartForm<GetLargestFaceRequest>,
    _: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    let get_face_req = form.into_inner();
    let temp_file = get_face_req.input;
    let img = tempfile_to_dynimg(temp_file)?;
    let t1 = std::time::Instant::now();
    let bboxes = loaded_model.forward(&img, 0.5).unwrap();
    println!("inference time {:?}", t1.elapsed());

    let mut _res = match bboxes {
        InferenceResult::FaceDetection(ayylmao) => ayylmao,
        _ => unreachable!(),
    };

    if !_res.is_empty() {
        let mut _a = sort_conf_bbox(&mut _res);
        let largest_face: DynamicImage;

        // if more than 1 bbox select the biggest
        // println!("_res {:?}", &_res);
        if _a.len() > 1 {
            let biggest_bbox = get_largest_bbox(_a);
            largest_face = biggest_bbox.to_owned().crop_bbox(&img).unwrap();
        } else {
            largest_face = _a[0].to_owned().crop_bbox(&img).unwrap();
        }
        let base64_image = image_to_base64(&largest_face)?;
        Ok(HttpResponse::Ok()
            .content_type("multipart/mixed; boundary=boundary")
            .json(GetLargestFaceResponse::new(
                FaceResponse::from_bbox_vec(&_res)[0], // just copy the damn thing
                base64_image,
            )))
    } else {
        Ok(HttpResponse::Ok().json(GetFaceResponseNone {
            message: String::from("no detections were found, please try with a better image!"),
        }))
    }
}
