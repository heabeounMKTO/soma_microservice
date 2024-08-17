pub mod face;
pub mod frame;

use serde::{Deserialize, Serialize};

/// all them INSERTS are the same
#[derive(Debug, Serialize, Deserialize)]
pub struct GenericResponse {
    pub status: i32,
    pub message: String,
}

impl GenericResponse {
    pub fn ok() -> GenericResponse {
        GenericResponse {
            status: 200,
            message: String::from("success"),
        }
    }
}
