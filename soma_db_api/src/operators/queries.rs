use actix_multipart::form::MultipartForm;
use actix_web::{get, post, web};
use actix_web::{HttpRequest, HttpResponse};
use anyhow::{Error, Result};
use deadpool_postgres::{GenericClient, Pool};
use pgvector::Vector;
use soma_db_api::handlers::face::{
    GetFaceByUuidRequest, GetFaceDetailResponse, 
    GetSimilarFacesByUuidRequest, GetSimilarFacesByUuidResponse,
};
use soma_db_api::handlers::GenericResponse;

use crate::handlers::face::GetSimilarFacesByEmbeddingRequest;



#[post("/get_face_by_uuid")]
pub async fn get_face_from_uuid(
    pool: web::Data<Pool>,
    form: web::Json<GetFaceByUuidRequest>,
    _: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    let client = pool.get().await.unwrap();

    // TODO: REFACTOR GET FACE 
    let rows = client.query("SELECT id, name, gender,embedding , face_uuid FROM face_embeddings WHERE face_uuid = $1", &[&form.face_uuid]).await.unwrap();
    if rows.len() > 0 {
        let results: Vec<GetFaceDetailResponse> = rows
            .into_iter()
            .map(|row| {
                let embedding: pgvector::Vector = row.get("embedding");
                GetFaceDetailResponse {
                    id: row.get("id"),
                    name: row.get("name"),
                    face_uuid: row.get("face_uuid"),
                    gender: row.get("gender"),
                    embedding: embedding.to_vec(),
                }
            })
            .collect();
        Ok(HttpResponse::Ok().json(results))
    } else {
        Ok(HttpResponse::Ok().json(GenericResponse {
            status: 200,
            message: String::from("no results were found for the given face_uuid"),
        }))
    }
}

#[post("/get_similar_faces_by_embedding")]
pub async fn get_similar_faces_by_embedding(
    pool: web::Data<Pool>,
    form: web::Json<GetSimilarFacesByEmbeddingRequest>
) -> actix_web::Result<HttpResponse> {
    let client = pool.get().await.unwrap();
    let face_embedding = pgvector::Vector::from(form.face_embedding.to_owned());
    let rows = client.query("SELECT id,name,gender,embedding,face_uuid, 1 - (embedding <=> $1) AS cosine_similarity FROM face_embeddings ORDER BY cosine_similarity DESC LIMIT $2", &[&face_embedding, &form.count]).await.unwrap();
    let similar_faces_results = {  
        if rows.len() > 0 {

            let results: Vec<GetSimilarFacesByUuidResponse> = rows
                    .into_iter()
                    .map(|row| {
                        let embedding: pgvector::Vector = row.get("embedding");
                        let face = GetFaceDetailResponse {
                            id: row.get("id"),
                            name: row.get("name"),
                            face_uuid: row.get("face_uuid"),
                            gender: row.get("gender"),
                            embedding: embedding.to_vec(),
                        };
                        let cosine_sim = row.get("cosine_similarity");
                        GetSimilarFacesByUuidResponse {
                            face: face,
                            cosine_similarity: cosine_sim,
                        }
                    })
                    .collect();

                results
        } else {
            let _res: Vec<GetSimilarFacesByUuidResponse> = vec![];
            _res
        }
    };
    if similar_faces_results.len() > 0 {
        Ok(HttpResponse::Ok().json(similar_faces_results))
    } else {
        let results: Vec<GetSimilarFacesByUuidResponse> = vec![];
        // just send a empty vec for now 
        Ok(HttpResponse::Ok().json(results))
    }
}

#[post("/get_similar_faces_by_uuid")]
pub async fn get_similar_faces_by_uuid(
    pool: web::Data<Pool>,
    form: web::Json<GetSimilarFacesByUuidRequest>,
    _: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    let client = pool.get().await.unwrap();
    // get face embedding first
    let rows = client.query("SELECT id, name, gender,embedding , face_uuid FROM face_embeddings WHERE face_uuid = $1", &[&form.face_uuid]).await.unwrap();
    let face_query_result: Vec<GetFaceDetailResponse> = {
        if rows.len() > 0 {
            let results: Vec<GetFaceDetailResponse> = rows
                .into_iter()
                .map(|row| {
                    let embedding: pgvector::Vector = row.get("embedding");
                    GetFaceDetailResponse {
                        id: row.get("id"),
                        name: row.get("name"),
                        face_uuid: row.get("face_uuid"),
                        gender: row.get("gender"),
                        embedding: embedding.to_vec(),
                    }
                })
                .collect();
            results
        } else {
            let results: Vec<GetFaceDetailResponse> = vec![];
            results
        }
    };
    let similar_faces_results = {
        if face_query_result.len() > 0 {
            let face_embedding = pgvector::Vector::from(face_query_result[0].embedding.to_owned());
            let rows = client.query("SELECT id,name,gender,embedding,face_uuid, 1 - (embedding <=> $1) AS cosine_similarity FROM face_embeddings ORDER BY cosine_similarity DESC LIMIT $2", &[&face_embedding, &form.count]).await.unwrap();
            let results: Vec<GetSimilarFacesByUuidResponse> = rows
                .into_iter()
                .map(|row| {
                    let embedding: pgvector::Vector = row.get("embedding");
                    let face = GetFaceDetailResponse {
                        id: row.get("id"),
                        name: row.get("name"),
                        face_uuid: row.get("face_uuid"),
                        gender: row.get("gender"),
                        embedding: embedding.to_vec(),
                    };
                    let cosine_sim = row.get("cosine_similarity");
                    GetSimilarFacesByUuidResponse {
                        face: face,
                        cosine_similarity: cosine_sim,
                    }
                })
                .collect();
            results
        } else {
            let _res: Vec<GetSimilarFacesByUuidResponse> = vec![];
            _res
        }
    };
    if similar_faces_results.len() > 0 {
        Ok(HttpResponse::Ok().json(similar_faces_results))
    } else {
        let results: Vec<GetSimilarFacesByUuidResponse> = vec![];
        // just send a empty vec for now 
        Ok(HttpResponse::Ok().json(results))
    }
}

