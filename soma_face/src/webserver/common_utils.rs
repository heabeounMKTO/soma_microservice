use actix_multipart::form::tempfile::TempFile;
use base64::{engine::general_purpose, Engine as _};
use image::{DynamicImage,ImageFormat, ImageReader};
use std::io::{Cursor, Read};

/// turns a [TempFile] from [actix_multipart::Multipart] into a [DynamicImage]
pub fn tempfile_to_dynimg(input_tempfile: TempFile) -> actix_web::Result<DynamicImage> {
    let mut file = input_tempfile.file;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let img = ImageReader::new(Cursor::new(buffer))
        .with_guessed_format()?
        .decode()
        .unwrap();
    Ok(img)
}


pub fn dynimg_to_bytes(input_img: &DynamicImage) -> Vec<u8> {
    let mut img_bytes: Vec<u8> = Vec::new();
    input_img.write_to(&mut Cursor::new(&mut img_bytes), image::ImageFormat::Png).unwrap(); 
    img_bytes
}

pub fn image_to_base64(img: &DynamicImage) -> Result<String, Box<dyn std::error::Error>> {
    let mut buffer = Cursor::new(Vec::new());
    img.write_to(&mut buffer, ImageFormat::Png)?;
    let bytes = buffer.into_inner();
    let base64_string = general_purpose::STANDARD.encode(bytes);
    Ok(base64_string)
}
