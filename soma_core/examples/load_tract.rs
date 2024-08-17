use soma_core::onnx_backend::{OnnxModel};
use tract_onnx::prelude::*;
use tract_core::model::typed::RunnableModel;
use clap::Parser;
#[derive(Parser)]
struct CliArgs {
    #[arg(long)]
    input_image: String,
    
    #[arg(long)]
    weights: String,
}

pub fn main() {
    let args = CliArgs::parse();

