//! enabling the `smol`
//! feature , runs all onnx models on 
//! `tract` instead of `ort` for onnx inference



pub mod onnx_backend;
pub mod common_utils;

#[cfg(feature="candle_models")]
pub mod blip_model;
#[cfg(feature="candle_models")]
pub mod blip_image_utils;



