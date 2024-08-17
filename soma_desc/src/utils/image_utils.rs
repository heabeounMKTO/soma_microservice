use anyhow::{Error, Result};
use base64::{
    engine::{self, general_purpose},
    Engine as _,
};
use image::{self, DynamicImage};

///
/// shinji decode the fucking b64 STRING
pub fn decode_base64(b64_image: &str) -> Result<image::DynamicImage> {
    let base64_decode: Vec<u8> = general_purpose::STANDARD.decode(b64_image)?;
    let _img = image::load_from_memory(&base64_decode.as_slice())?;
    Ok(_img)
}

pub fn tensor_from_image(image: &DynamicImage) -> Result<candle_core::Tensor, candle_core::Error> {
    let img = image.resize_to_fill(384, 384, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data =
        candle_core::Tensor::from_vec(data, (384, 384, 3), &candle_core::Device::new_cuda(0)?)?
            .permute((2, 0, 1))?;
    let mean = candle_core::Tensor::new(
        &[0.48145466f32, 0.4578275, 0.40821073],
        &candle_core::Device::new_cuda(0)?,
    )?
    .reshape((3, 1, 1))?;
    let std = candle_core::Tensor::new(
        &[0.26862954f32, 0.261_302_6, 0.275_777_1],
        &candle_core::Device::new_cuda(0)?,
    )?
    .reshape((3, 1, 1))?;
    (data.to_dtype(candle_core::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

pub fn load_image<P: AsRef<std::path::Path>>(
    p: P,
) -> Result<candle_core::Tensor, candle_core::Error> {
    let img = image::io::Reader::open(p)?
        .decode()
        .map_err(candle_core::Error::wrap)?
        .resize_to_fill(384, 384, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data =
        candle_core::Tensor::from_vec(data, (384, 384, 3), &candle_core::Device::new_cuda(0)?)?
            .permute((2, 0, 1))?;
    let mean = candle_core::Tensor::new(
        &[0.48145466f32, 0.4578275, 0.40821073],
        &candle_core::Device::new_cuda(0)?,
    )?
    .reshape((3, 1, 1))?;
    let std = candle_core::Tensor::new(
        &[0.26862954f32, 0.261_302_6, 0.275_777_1],
        &candle_core::Device::new_cuda(0)?,
    )?
    .reshape((3, 1, 1))?;
    (data.to_dtype(candle_core::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

