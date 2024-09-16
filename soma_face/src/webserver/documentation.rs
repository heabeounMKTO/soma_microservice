use crate::webserver;
use utoipa::OpenApi;
use webserver::handler::{
    GetFaceRequest, GetFaceResponse, GetFaceResponseNone, GetFaceVecRequest, GetFaceVecResponse,
};

#[derive(OpenApi)]
#[openapi(
    paths(crate::webserver::service::get_face_bbox_yolo),
    components(schemas(GetFaceRequest, GetFaceResponse))
)]
pub struct GetFaceDocsYolo;

#[derive(OpenApi)]
#[openapi(
    paths(crate::webserver::service::get_face_bbox_retinaface),
    components(schemas(GetFaceRequest, GetFaceResponse, GetFaceResponseNone))
)]
pub struct GetFaceDocsRetina;

#[derive(OpenApi)]
#[openapi(
    paths(crate::webserver::service::get_face_vectors),
    components(schemas(GetFaceVecRequest, GetFaceVecResponse))
)]
pub struct GetFaceVecDocsArcFace;

#[derive(OpenApi)]
#[openapi(
    paths(crate::webserver::service::extract_face),
    components(schemas(GetFaceRequest, GetFaceResponse, GetFaceResponseNone))
)]
pub struct GetFaceDocs;
