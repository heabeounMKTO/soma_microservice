use crate::core::common_utils::{image_to_ndarray, Bbox};
use crate::core::onnx_backend::{Inference, InferenceResult, OnnxModel};
use anyhow::{Error, Result};
use image::imageops;
use image::DynamicImage;
use ndarray::{
    array, s, stack, Array, Array2, Array3, ArrayBase, ArrayView2, ArrayViewD, Axis, Dim, Ix3,
    IxDynImpl, OwnedRepr,
};
use ort::{inputs, CPUExecutionProvider, GraphOptimizationLevel, Session, SessionOutputs};

pub struct GetFaceRetinaface {
    pub onnx_model: OnnxModel,
    pub width: i32,
    pub height: i32,
}

impl GetFaceRetinaface {
    pub fn new(
        model_path: &str,
        width: i32,
        height: i32,
        use_fp16: bool,
    ) -> Result<GetFaceRetinaface, Error> {
        Ok(GetFaceRetinaface {
            onnx_model: GetFaceRetinaface::load(model_path, use_fp16),
            width,
            height,
        })
    }
}

impl Inference for GetFaceRetinaface {
    fn forward(
        &self,
        input_image: &DynamicImage,
        confidence_threshold: f32,
    ) -> Result<InferenceResult, Error> {
        let model_config = ScrfdFaceConfig::from_loaded_session(&self.onnx_model);
        let preproces_image = preprocess_face_f32_retina(input_image)?;
        let inference = self
            .onnx_model
            .model
            .run(inputs!["input.1" => preproces_image.view()]?)?;
        let mut process_infernece = process_detections(
            inference,
            model_config.stride_fpn,
            model_config.fmc,
            model_config.anchors,
            confidence_threshold,
            true,
        )?;
        Ok(InferenceResult::FaceDetection(
            process_infernece
                .iter_mut()
                .map(|x| x.apply_image_scale(input_image, self.width as f32, self.height as f32))
                .collect(),
        ))
    }
    fn load(model_path: &str, fp16: bool) -> OnnxModel {
        let model: ort::Session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();
        OnnxModel {
            model,
            is_fp16: fp16,
        }
    }
}

struct ScrfdFaceConfig {
    fmc: usize,
    stride_fpn: Vec<i32>,
    anchors: usize,
}

/// configs for 10g / 2.5g kpss , other models are not suppourted yet
impl ScrfdFaceConfig {
    pub fn from_loaded_session(model: &OnnxModel) -> ScrfdFaceConfig {
        match model.model.outputs.len() {
            6 => ScrfdFaceConfig {
                fmc: 3,
                stride_fpn: vec![8, 16, 32],
                anchors: 2,
            },
            9 => ScrfdFaceConfig {
                fmc: 3,
                stride_fpn: vec![8, 16, 32],
                anchors: 2,
            },
            _ => {
                panic!("SCRFD model not suppourted (yet)")
            }
        }
    }
}

pub fn preprocess_face_f32_retina(
    image_source: &DynamicImage,
) -> Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>, Error> {
    let img = image_source.resize_exact(640, 640, imageops::FilterType::Triangle);

    let mut ndarray_image = image_to_ndarray(&img);
    ndarray_image -= 127.5;
    ndarray_image *= 1.0 / 128.0;
    let _final = ndarray_image.permuted_axes((2, 0, 1));
    Ok(_final.insert_axis(ndarray::Axis(0)))
}

fn stack_anchor_center(anchor_centers: &Array2<f32>, num_anchors: usize) -> Array2<f32> {
    let (rows, cols) = anchor_centers.dim();
    let mut stacked = Array3::<f32>::zeros((rows, num_anchors, cols));
    for i in 0..num_anchors {
        stacked.slice_mut(s![.., i, ..]).assign(anchor_centers);
    }

    stacked.into_shape((rows * num_anchors, cols)).unwrap()
}

fn anchor_centers(height: usize, width: usize) -> Array<f32, Ix3> {
    let mut y_coords = Array::zeros((height, width));
    let mut x_coords = Array::zeros((height, width));
    for i in 0..height {
        for j in 0..width {
            y_coords[[i, j]] = i as f32;
            x_coords[[i, j]] = j as f32;
        }
    }
    let mut anchor_centers = Array::zeros((height, width, 2));
    anchor_centers.slice_mut(s![.., .., 0]).assign(&x_coords);
    anchor_centers.slice_mut(s![.., .., 1]).assign(&y_coords);
    anchor_centers
}

fn distance2kps(
    points: &ArrayView2<f32>,
    distance: &ArrayViewD<f32>,
    max_shape: Option<(usize, usize)>,
) -> Array2<f32> {
    let mut preds = Vec::new();

    for i in (0..distance.shape()[1]).step_by(2) {
        let px = &points.slice(s![.., i % 2]) + &distance.slice(s![.., i]);
        let py = &points.slice(s![.., (i % 2) + 1]) + &distance.slice(s![.., i + 1]);

        let px = if let Some((_, max_x)) = max_shape {
            px.mapv(|x| x.max(0.0).min(max_x as f32))
        } else {
            px.to_owned()
        };

        let py = if let Some((max_y, _)) = max_shape {
            py.mapv(|y| y.max(0.0).min(max_y as f32))
        } else {
            py.to_owned()
        };

        preds.push(px);
        preds.push(py);
    }

    let preds: Vec<_> = preds.iter().map(|a| a.view()).collect();
    stack(Axis(1), &preds).unwrap()
}

fn distance2bbox(
    points: &ArrayView2<f32>,
    distance: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
) -> Array2<f32> {
    let x1 = &points.slice(s![.., 0]) - &distance.slice(s![.., 0]);
    let y1 = &points.slice(s![.., 1]) - &distance.slice(s![.., 1]);
    let x2 = &points.slice(s![.., 0]) + &distance.slice(s![.., 2]);
    let y2 = &points.slice(s![.., 1]) + &distance.slice(s![.., 3]);
    stack(Axis(1), &[x1.view(), y1.view(), x2.view(), y2.view()]).unwrap()
}

fn reshape_kpss(kpss: Array2<f32>) -> Array3<f32> {
    let shape = kpss.shape();
    let new_shape = (shape[0], shape[1] / 2, 2);
    kpss.into_shape(new_shape).expect("Failed to reshape array")
}

/// get bboxes from onnx outputs
pub fn process_detections(
    inference: SessionOutputs,
    stride_fpn: Vec<i32>,
    fmc: usize,
    num_anchors: usize,
    confidence_thresh: f32,
    include_kps: bool, // returns face keypoints
) -> Result<Vec<Bbox>, Error> {
    let mut raw_bboxes: Vec<Bbox> = vec![];
    // process
    for (idx, stride) in stride_fpn.iter().enumerate() {
        let scores = &inference[idx].try_extract_tensor::<f32>()?.into_owned();
        let _bbox_preds = &inference[idx + fmc]
            .try_extract_tensor::<f32>()?
            .into_owned();

        let bbox_preds = _bbox_preds.to_owned() * (stride.to_owned() as f32);
        let height = (640 / stride) as usize; // 640 = input size.
        let width = (640 / stride) as usize;
        let ac = anchor_centers(height, width);

        let mut _anchor_centers = (ac.to_owned() * (stride.to_owned() as f32))
            .into_shape((height * width, 2))
            .unwrap();
        let _stack = stack_anchor_center(&_anchor_centers, num_anchors);
        let scaled_bbox = distance2bbox(&_stack.view(), &bbox_preds);
        let mut _reshape_kpss: ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>> = array![[[]]];
        if include_kps {
            let mut _kps = &inference[idx + fmc * 2]
                .try_extract_tensor::<f32>()?
                .into_owned();
            let kps = _kps * stride.to_owned() as f32;
            let decoded_kps = distance2kps(&_stack.view(), &kps.view(), None);
            _reshape_kpss = reshape_kpss(decoded_kps);
        }

        let _fucc: Vec<Bbox> = scores
            .indexed_iter()
            .enumerate()
            .filter_map(|(idx, (val, conf))| {
                if conf >= &confidence_thresh {
                    Some(Bbox {
                        x1: scaled_bbox.slice(s![val[0], ..]).to_owned()[0],
                        y1: scaled_bbox.slice(s![val[0], ..]).to_owned()[1],
                        x2: scaled_bbox.slice(s![val[0], ..]).to_owned()[2],
                        y2: scaled_bbox.slice(s![val[0], ..]).to_owned()[3],
                        confidence: conf.to_owned(),
                        kpss: if include_kps {
                            _reshape_kpss
                                .slice(s![val[0], .., ..])
                                .outer_iter()
                                .map(|x| x.to_vec())
                                .collect()
                        } else {
                            vec![vec![]]
                        },
                    })
                } else {
                    None
                }
            })
            .collect();
        raw_bboxes = _fucc;
    }
    Ok(raw_bboxes)
}
