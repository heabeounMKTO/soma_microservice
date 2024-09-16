//! face services compilation
use crate::core::common_utils::get_largest_bbox;
use crate::core::common_utils::rotate_image_with_index;
use crate::core::common_utils::sort_conf_bbox;
use crate::core::common_utils::Bbox;
use crate::core::onnx_backend::{Inference, InferenceResult};
use crate::get_face::retinaface::GetFaceRetinaface;
use crate::get_face::yolo::GetFaceYolo;
use crate::get_face::FaceExtractor;
use crate::get_face_vec::arcface::GetFaceVecArcFace;
use crate::webserver::common_utils::image_to_base64;
#[cfg(feature = "readvid")]
use crate::webserver::common_utils::{
    capture_frames_at_interval_and_detect, capture_frames_at_intervals, get_total_frame_from_video,
    read_video_from_path_and_convert_frames, read_video_from_path_and_get_detections,
};

use crate::webserver::common_utils::{tempfile_to_dynimg, CropImagePair};
use crate::webserver::handler::FaceResponse;
use crate::webserver::handler::{
    GetFaceRequest, GetFaceResponse, GetFaceResponseNone, GetFaceVecRequest, GetFaceVecResponse,
    GetLargestFaceRequest, GetLargestFaceResponse,
};
use actix_multipart::form::MultipartForm;
use actix_web::http::header::ContentType;
use actix_web::{get, post, web, HttpRequest, HttpResponse};
use image::DynamicImage;

use crate::webserver::handler::GetFaceFromVideoRequest;

/// infos to show in the index function
struct IndexInfo {
    model_name: String,
    workers: i32,
}

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
    let mut _res: Vec<Bbox> = vec![];
    let mut mod_img: DynamicImage = DynamicImage::default();
    // rotate image and run forward pass until we find ,
    // if we dont find anything despite the lengths we go through,
    // just give up and return
    for idx in 0..4 {
        println!("idx {:?}", idx);
        let _rot = rotate_image_with_index(&img, idx).unwrap();
        let fwd = loaded_model.forward(&_rot, 0.5).unwrap();
        let _pre_res = match fwd {
            InferenceResult::FaceDetection(res) => res,
            _ => unreachable!(),
        };
        if _pre_res.is_empty() {
            continue;
        } else {
            _res = _pre_res;
            mod_img = _rot;
            break;
        }
    }
    // println!("res {:?}", _res);
    if !_res.is_empty() {
        // let mut _a = (&mut _res);
        let largest_face: DynamicImage;
        
        // if more than 1 bbox select the biggest
        // println!("_res {:?}", &_res);
        if _res.len() > 1 {
            let biggest_bbox = get_largest_bbox(_res.to_owned());
            largest_face = biggest_bbox.to_owned().crop_and_align_face(&mod_img, (224,224)).unwrap();
        } else {
            largest_face = _res[0].to_owned().crop_bbox(&mod_img).unwrap();
        }
        let base64_image = image_to_base64(&largest_face)?;
        // TODO: BASE 64 ALL THAT SHIT SMH FUCK BYTES, FUCK MULTIPART
        Ok(HttpResponse::Ok()
            .content_type("multipart/mixed; boundary=boundary")
            .json(GetLargestFaceResponse::new(
                FaceResponse::from_bbox_vec(&_res)[0],
                base64_image,
            )))
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
    let mut _res: Vec<Bbox> = vec![];
    // let mut mod_img: DynamicImage = DynamicImage::default();
    // rotate image and run forward pass until we find ,
    // if we dont find anything despite the lengths we go through,
    // just give up and return
    for idx in 0..4 {
        println!("idx {:?}", idx);
        let _rot = rotate_image_with_index(&img, idx).unwrap();
        let fwd = loaded_model.forward(&_rot, 0.5).unwrap();
        let _pre_res = match fwd {
            InferenceResult::FaceDetection(res) => res,
            _ => unreachable!(),
        };
        if _pre_res.is_empty() {
            continue;
        } else {
            _res = _pre_res;
            break;
        }
    }
    if !_res.is_empty() {
        let mut _a = sort_conf_bbox(&mut _res);
        // _a = _a.into_iter().map(|x| x.to_xywh()).collect();
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
    if !_res.is_empty() {
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
    if !_res.is_empty() {
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

/// loads a video from path given by client , finds all
/// faces and then sorts based on confidence to get the
/// best one from all frames
///
#[cfg(feature = "readvid")]
#[post("/get_face_from_video")]
pub async fn extract_face_video(
    request_data: web::Json<GetFaceFromVideoRequest>,
    loaded_model: web::Data<GetFaceYolo>,
    req: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    println!("DEBUG: FORM {:?}", &request_data);
    println!("DEBUG: REQ {:?}", &req);
    let total_frames = get_total_frame_from_video(&request_data.video_path).unwrap();
    println!("total fraems {:?}", &total_frames);
    let mut load_video: Vec<DynamicImage> = vec![];
    // dont chunk video if it's under two seconds
    if &total_frames < &50 {
        load_video = read_video_from_path_and_convert_frames(&request_data.video_path).unwrap();
    } else {
        load_video = capture_frames_at_intervals(
            &request_data.video_path,
            (total_frames as f32 * 0.15) as u32,
        )
        .unwrap();
    }
    let mut pairs: Vec<CropImagePair> = vec![];
    for image in load_video.iter() {
        let mut _run_inference = match loaded_model.forward(image, 0.5).unwrap() {
            InferenceResult::FaceDetection(res) => res,
            // something is seriously wrong if somehow
            // the results comes up as FaceEmbedding,
            // therefore lose your shit
            _ => panic!(),
        };
        if _run_inference.len() > 0 {
            let _sort = sort_conf_bbox(&mut _run_inference);

            // get detection with higest conf
            let best = CropImagePair {
                cropped: _sort[0].to_owned().crop_bbox(image).unwrap(),
                bbox: _sort[0].to_owned(),
            };
            pairs.push(best);
        } else {
            continue;
        }
    }
    println!("founded {:?}", pairs.len());
    if pairs.len() > 0 {
        let best_pair = {
            if pairs.len() > 2 {
                // if bigger than 2 seelct best one
                pairs.sort_by(|a, b| b.bbox.confidence.partial_cmp(&a.bbox.confidence).unwrap());
                pairs[0].clone()
            } else {
                // if not empty select the first one
                pairs[0].clone()
            }
        };
        let b64_image = image_to_base64(&best_pair.cropped)?;
        Ok(HttpResponse::Ok()
            .content_type("multipart/mixed; boundary=boundary")
            .json(GetLargestFaceResponse::new(
                //TODO: refactor this , dont
                //have to alloc array twice :\
                FaceResponse::from_bbox_vec(&vec![best_pair.bbox])[0],
                b64_image,
            )))
    } else {
        Ok(HttpResponse::Ok().json(GetFaceResponseNone {
            message: String::from("no detections were found, please try with a better image!"),
        }))
    }
}
