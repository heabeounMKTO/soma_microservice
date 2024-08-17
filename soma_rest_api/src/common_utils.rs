use ::tempfile::NamedTempFile;
use actix_multipart::form::tempfile::TempFile;
use anyhow::{Error, Result};
use base64::{
    engine::{self, general_purpose},
    Engine as _,
};
use image::{DynamicImage, ImageReader};
use serde::{Deserialize, Serialize};
use std::io::Write;
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

pub fn base64_to_tempfile(base64_string: &str) -> Result<NamedTempFile, Error> {
    let mut temp_file = NamedTempFile::new()?;
    let image_bytes = base64_to_bytes(base64_string)?;
    temp_file.write_all(&image_bytes)?;
    temp_file.flush()?;
    Ok(temp_file)
}

pub fn base64_to_bytes(base64_string: &str) -> Result<Vec<u8>> {
    let decode = general_purpose::STANDARD.decode(base64_string)?;
    Ok(decode)
}

pub fn dynimg_to_bytes(input_img: &DynamicImage) -> Vec<u8> {
    let mut img_bytes: Vec<u8> = Vec::new();
    input_img
        .write_to(&mut Cursor::new(&mut img_bytes), image::ImageFormat::Png)
        .unwrap();
    img_bytes
}

const splash: &str = r#"
            .▄▄ ·       • ▌ ▄ ·.  ▄▄▄·     
            ▐█ ▀. ▪     ·██ ▐███▪▐█ ▀█     
            ▄▀▀▀█▄ ▄█▀▄ ▐█ ▌▐▌▐█·▄█▀▀█     
            ▐█▄▪▐█▐█▌.▐▌██ ██▌▐█▌▐█ ▪▐▌    
             ▀▀▀▀  ▀█▄▀▪▀▀  █▪▀▀▀ ▀  ▀     
            ▄▄▄  ▄▄▄ ..▄▄ · ▄▄▄▄▄          
            ▀▄ █·▀▄.▀·▐█ ▀. •██            
            ▐▀▀▄ ▐▀▀▪▄▄▀▀▀█▄ ▐█.▪          
            ▐█•█▌▐█▄▄▌▐█▄▪▐█ ▐█▌·          
            .▀  ▀ ▀▀▀  ▀▀▀▀  ▀▀▀           
    "#;

pub fn print_splash() {
    println!("soma_service's");
    println!("{}", splash);
}
