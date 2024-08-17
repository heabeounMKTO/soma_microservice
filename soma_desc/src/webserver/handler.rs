use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize)]
pub struct ImageDescRequest {
    pub data: String,
}

#[derive(Serialize, Deserialize)]
pub struct ImageDescResponse {
    pub data: String,
    pub message: String,
}

