use crate::handlers::face::InsertFaceRequest;
use crate::handlers::GenericResponse;
use actix_web::{post, web};
use actix_web::{HttpRequest, HttpResponse};
use deadpool_postgres::Pool;
use pgvector::Vector;
use serde_json;

#[post("/post_face_vec")]
pub async fn insert_face_vector(
    pool: web::Data<Pool>,
    form: web::Json<InsertFaceRequest>,
    req: HttpRequest,
) -> actix_web::Result<HttpResponse> {
    // perform vector length check
    if form.embedding.len() != 512 {
        Ok(HttpResponse::BadRequest().json(GenericResponse {
            status: 400,
            message: String::from(
                "invalid vector dimension , input vector must be exactly 512 long!",
            ),
        }))
    } else {
        let client = pool.get().await.unwrap();
        // let gender_string: Option<String> = match form.gender {
        //     Some(gender) => Some(gender.to_string()),
        //     None => None
        // };
        let pgvec_vector = Vector::from(form.embedding.to_owned());
        let statement = client.prepare("INSERT INTO face_embeddings (name, embedding, gender, face_uuid) VALUES ($1, $2, $3, $4)").await.unwrap();
        client
            .execute(
                &statement,
                &[&form.name, &pgvec_vector, &form.gender, &form.face_uuid],
            )
            .await
            .unwrap();
        Ok(HttpResponse::Created().json(GenericResponse::ok()))
    }
}
