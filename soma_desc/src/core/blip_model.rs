use crate::utils::image_utils::tensor_from_image;

use anyhow::{Error, Result};
use candle_core;
use candle_core::DType;
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::blip::{self, BlipForConditionalGeneration};
use dotenvy::dotenv;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::env;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::{tokenizer, Tokenizer};

const SEP_TOKEN_ID: u32 = 102;

pub struct BlipModel {
    // blip_tokenizer: tokenizers::Tokenizer,
    blip_device: candle_core::Device,
    blip_model: BlipForConditionalGeneration,
}

pub struct BlipResult {
    pub description: String,
}

impl BlipModel {
    pub fn init(device: candle_core::Device) -> Result<BlipModel> {
        dotenv().ok();
        // let blip_tokenizer = env::var("BLIP_TOKENIZER").expect("cannot find BLIP_TOKENIZER");
        let blip_model = env::var("BLIP_MODEL").expect("cannot find BLIP_MODEL");

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[blip_model], candle_core::DType::F32, &device)?
        };
        let config = blip::Config::image_captioning_large();
        let model = blip::BlipForConditionalGeneration::new(&config, vb)?;
        Ok(BlipModel {
            blip_device: device,
            blip_model: model,
        })
    }

    pub fn run(&mut self, image: &DynamicImage) -> Result<BlipResult> {
        let _image = tensor_from_image(image)?.to_device(&self.blip_device)?;
        let image_embeddings = _image.unsqueeze(0)?.apply(self.blip_model.vision_model())?;
        let mut token_ids = vec![30522u32];
        let mut logits_processor =
            candle_transformers::generation::LogitsProcessor::new(1337, None, None);
        let blip_tokenizer = env::var("BLIP_TOKENIZER").expect("cannot find BLIP_TOKENIZER");

        let tokenizer = tokenizers::Tokenizer::from_file(blip_tokenizer).map_err(Error::msg)?;
        let mut _tokenizer = TokenOutputStream::new(tokenizer);
        let mut words: Vec<String> = vec![];
        for index in 0..100 {
            let context_size = if index > 0 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);
            let input_ids = candle_core::Tensor::new(&token_ids[start_pos..], &self.blip_device)?
                .unsqueeze(0)?;
            let logits = self
                .blip_model
                .text_decoder()
                .forward(&input_ids, &image_embeddings)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;
            let token = logits_processor.sample(&logits)?;
            if token == SEP_TOKEN_ID {
                break;
            }
            // println!("DEBUG: TOKEN: {:?}", &token);
            token_ids.push(token);
            if let Some(t) = _tokenizer.next_token(token)? {
                use std::io::Write;
                words.push(t);
                std::io::stdout().flush()?;
            }
        }

        /*this is a hack , just "resets the model" , DO NOT USE IN PROD (YET)  */

        let blip_model = env::var("BLIP_MODEL").expect("cannot find BLIP_MODEL");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[blip_model],
                candle_core::DType::F32,
                &self.blip_device,
            )?
        };
        let config = blip::Config::image_captioning_large();
        let model = blip::BlipForConditionalGeneration::new(&config, vb)?;
        self.blip_model = model;
        //-----------------------//
        let _words: String = words.to_owned().join(""); 
        let _vec = _words.to_owned();
        println!("string vec {:?}", _vec);


        Ok(BlipResult {
            description: words.join(""),
        })
    }
}

