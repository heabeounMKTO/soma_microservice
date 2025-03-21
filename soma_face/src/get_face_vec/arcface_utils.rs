use ndarray::{Array, Array1, Array2, Axis, s};
use ndarray_linalg::Solve;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use imageproc::geometric_transformations::{warp_into, ProjectiveTransform};

// Equivalent to arcface_dst in Python
const ARCFACE_DST: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

// Represents a transformation matrix
#[derive(Debug, Clone)]
pub struct TransformationMatrix {
    pub matrix: Array2<f32>,
}

impl TransformationMatrix {
    // Create a new transformation matrix
    pub fn new(matrix: Array2<f32>) -> Self {
        Self { matrix }
    }

    // Get the raw matrix as a 2x3 array
    pub fn as_2x3(&self) -> [[f32; 3]; 2] {
        [
            [self.matrix[[0, 0]], self.matrix[[0, 1]], self.matrix[[0, 2]]],
            [self.matrix[[1, 0]], self.matrix[[1, 1]], self.matrix[[1, 2]]],
        ]
    }

    // Calculate scale of the transformation
    pub fn get_scale(&self) -> f32 {
        (self.matrix[[0, 0]].powi(2) + self.matrix[[0, 1]].powi(2)).sqrt()
    }
}

// Equivalent to SimilarityTransform in Python
#[derive(Debug, Clone)]
pub struct SimilarityTransform {
    pub matrix: Array2<f32>,
}

impl SimilarityTransform {
    // Create a new identity transformation
    pub fn new() -> Self {
        Self {
            matrix: Array2::eye(3),
        }
    }

    // Create a new transformation with scaling
    pub fn with_scale(scale: f32) -> Self {
        let mut matrix = Array2::eye(3);
        matrix[[0, 0]] = scale;
        matrix[[1, 1]] = scale;
        Self { matrix }
    }

    // Create a new transformation with translation
    pub fn with_translation(tx: f32, ty: f32) -> Self {
        let mut matrix = Array2::eye(3);
        matrix[[0, 2]] = tx;
        matrix[[1, 2]] = ty;
        Self { matrix }
    }

    // Create a new transformation with rotation
    pub fn with_rotation(angle: f32) -> Self {
        let mut matrix = Array2::eye(3);
        matrix[[0, 0]] = angle.cos();
        matrix[[0, 1]] = -angle.sin();
        matrix[[1, 0]] = angle.sin();
        matrix[[1, 1]] = angle.cos();
        Self { matrix }
    }

    // Combine two transformations
    pub fn add(&self, other: &SimilarityTransform) -> Self {
        let combined = self.matrix.dot(&other.matrix);
        Self { matrix: combined }
    }

    // Estimate transformation from source points to destination points
    pub fn estimate(&mut self, src: &Array2<f32>, dst: &Array2<f32>) {
        assert_eq!(src.shape(), dst.shape());
        assert_eq!(src.shape()[1], 2);

        let n = src.shape()[0];
        
        // Compute centroids
        let src_centroid = src.mean_axis(Axis(0)).unwrap();
        let dst_centroid = dst.mean_axis(Axis(0)).unwrap();
        
        // Shift points to centroids
        let src_centered = src.mapv(|x| x) - &src_centroid;
        let dst_centered = dst.mapv(|x| x) - &dst_centroid;
        
        // Calculate covariance matrix
        let mut cov = Array2::<f32>::zeros((2, 2));
        for i in 0..n {
            for j in 0..2 {
                for k in 0..2 {
                    cov[[j, k]] += src_centered[[i, j]] * dst_centered[[i, k]];
                }
            }
        }
        
        // Compute SVD (simplified here, in practice use a library function)
        // This is a simplified version - use proper SVD algorithms for production
        let s_xx = cov[[0, 0]];
        let s_xy = cov[[0, 1]];
        let s_yx = cov[[1, 0]];
        let s_yy = cov[[1, 1]];
        
        let mu = ((s_xy - s_yx).powi(2) + (s_xx + s_yy).powi(2)).sqrt();
        
        // Calculate rotation
        let sin_theta = (s_xy - s_yx) / mu;
        let cos_theta = (s_xx + s_yy) / mu;
        
        // Calculate scale
        let src_var = (src_centered.clone() * src_centered).sum() / (n as f32);
        let scale = mu / (2.0 * src_var);
        
        // Populate transformation matrix
        self.matrix = Array2::eye(3);
        self.matrix[[0, 0]] = scale * cos_theta;
        self.matrix[[0, 1]] = -scale * sin_theta;
        self.matrix[[1, 0]] = scale * sin_theta;
        self.matrix[[1, 1]] = scale * cos_theta;
        
        // Calculate translation
        self.matrix[[0, 2]] = dst_centroid[0] - (self.matrix[[0, 0]] * src_centroid[0] + self.matrix[[0, 1]] * src_centroid[1]);
        self.matrix[[1, 2]] = dst_centroid[1] - (self.matrix[[1, 0]] * src_centroid[0] + self.matrix[[1, 1]] * src_centroid[1]);
    }
    
    // Get the parameters as a 2x3 transformation matrix
    pub fn get_params(&self) -> TransformationMatrix {
        TransformationMatrix::new(self.matrix.slice(s![0..2, ..]).to_owned())
    }
}

// Estimate normalization matrix (equivalent to estimate_norm in Python)
pub fn estimate_norm(lmk: &Array2<f32>, image_size: u32, mode: &str) -> TransformationMatrix {
    assert_eq!(lmk.shape(), &[5, 2]);
    assert!(image_size % 112 == 0 || image_size % 128 == 0);
    
    let (ratio, diff_x) = if image_size % 112 == 0 {
        (image_size as f32 / 112.0, 0.0)
    } else {
        (image_size as f32 / 128.0, 8.0 * (image_size as f32 / 128.0))
    };
    
    // Create destination points
    let mut dst = Array2::<f32>::zeros((5, 2));
    for i in 0..5 {
        dst[[i, 0]] = ARCFACE_DST[i][0] * ratio + diff_x;
        dst[[i, 1]] = ARCFACE_DST[i][1] * ratio;
    }
    
    // Estimate transformation
    let mut tform = SimilarityTransform::new();
    tform.estimate(lmk, &dst);
    
    // Return transformation matrix
    tform.get_params()
}

// Crop and normalize face (equivalent to norm_crop in Python)
pub fn norm_crop(
    img: &DynamicImage, 
    landmark: &Array2<f32>, 
    image_size: u32,
    mode: &str
) -> DynamicImage {
    let m = estimate_norm(landmark, image_size, mode);
    
    // Create a new empty image
    let mut output = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(image_size, image_size);
    
    // Create transformation
    let transform = ProjectiveTransform::from_matrix([
        [m.matrix[[0, 0]], m.matrix[[0, 1]], m.matrix[[0, 2]]],
        [m.matrix[[1, 0]], m.matrix[[1, 1]], m.matrix[[1, 2]]],
        [0.0, 0.0, 1.0],
    ]).unwrap();
    
    // Apply transformation
    warp_into(img, &transform, imageproc::geometric_transformations::Interpolation::Bilinear, Rgb([0, 0, 0]), &mut output);
    
    DynamicImage::ImageRgb8(output)
}

// Same as norm_crop but also returns the transformation matrix
pub fn norm_crop2(
    img: &DynamicImage, 
    landmark: &Array2<f32>, 
    image_size: u32, 
    mode: &str
) -> (DynamicImage, TransformationMatrix) {
    let m = estimate_norm(landmark, image_size, mode);
    
    // Create a new empty image
    let mut output = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(image_size, image_size);
    
    // Create transformation
    let transform = ProjectiveTransform::from_matrix([
        [m.matrix[[0, 0]], m.matrix[[0, 1]], m.matrix[[0, 2]]],
        [m.matrix[[1, 0]], m.matrix[[1, 1]], m.matrix[[1, 2]]],
        [0.0, 0.0, 1.0],
    ]).unwrap();
    
    // Apply transformation
    warp_into(img, &transform, imageproc::geometric_transformations::Interpolation::Bilinear, Rgb([0, 0, 0]), &mut output);
    
    (DynamicImage::ImageRgb8(output), m)
}

// Equivalent to square_crop in Python
pub fn square_crop(
    img: &DynamicImage,
    target_size: u32
) -> (DynamicImage, f32) {
    let (width, height) = img.dimensions();
    let (new_width, new_height, scale) = if height > width {
        let new_height = target_size;
        let new_width = (width as f32 * target_size as f32 / height as f32) as u32;
        (new_width, new_height, target_size as f32 / height as f32)
    } else {
        let new_width = target_size;
        let new_height = (height as f32 * target_size as f32 / width as f32) as u32;
        (new_width, new_height, target_size as f32 / width as f32)
    };
    
    // Resize image
    let resized = img.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);
    
    // Create a new square image filled with zeros
    let mut output = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(target_size, target_size);
    
    // Copy resized image to output
    for y in 0..resized.height() {
        for x in 0..resized.width() {
            let pixel = resized.get_pixel(x, y);
            output.put_pixel(x, y, pixel);
        }
    }
    
    (DynamicImage::ImageRgb8(output), scale)
}

// Transform image (equivalent to transform in Python)
pub fn transform_image(
    img: &DynamicImage,
    center: (f32, f32),
    output_size: u32,
    scale: f32,
    rotation: f32
) -> (DynamicImage, TransformationMatrix) {
    let scale_ratio = scale;
    let rot = rotation * std::f32::consts::PI / 180.0;
    
    // Create transformations
    let t1 = SimilarityTransform::with_scale(scale_ratio);
    
    let cx = center.0 * scale_ratio;
    let cy = center.1 * scale_ratio;
    let t2 = SimilarityTransform::with_translation(-cx, -cy);
    
    let t3 = SimilarityTransform::with_rotation(rot);
    
    let t4 = SimilarityTransform::with_translation(output_size as f32 / 2.0, output_size as f32 / 2.0);
    
    // Combine transformations
    let t = t1.add(&t2).add(&t3).add(&t4);
    let m = t.get_params();
    
    // Create a new empty image
    let mut output = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(output_size, output_size);
    
    // Create transformation
    let transform = ProjectiveTransform::from_matrix([
        [m.matrix[[0, 0]], m.matrix[[0, 1]], m.matrix[[0, 2]]],
        [m.matrix[[1, 0]], m.matrix[[1, 1]], m.matrix[[1, 2]]],
        [0.0, 0.0, 1.0],
    ]).unwrap();
    
    // Apply transformation
    warp_into(img, &transform, imageproc::geometric_transformations::Interpolation::Bilinear, Rgb([0, 0, 0]), &mut output);
    
    (DynamicImage::ImageRgb8(output), m)
}

// Transform 2D points (equivalent to trans_points2d in Python)
pub fn trans_points2d(pts: &Array2<f32>, m: &TransformationMatrix) -> Array2<f32> {
    let n = pts.shape()[0];
    let mut new_pts = Array2::<f32>::zeros((n, 2));
    
    for i in 0..n {
        let pt = [pts[[i, 0]], pts[[i, 1]], 1.0];
        let x = m.matrix[[0, 0]] * pt[0] + m.matrix[[0, 1]] * pt[1] + m.matrix[[0, 2]];
        let y = m.matrix[[1, 0]] * pt[0] + m.matrix[[1, 1]] * pt[1] + m.matrix[[1, 2]];
        new_pts[[i, 0]] = x;
        new_pts[[i, 1]] = y;
    }
    
    new_pts
}

// Transform 3D points (equivalent to trans_points3d in Python)
pub fn trans_points3d(pts: &Array2<f32>, m: &TransformationMatrix) -> Array2<f32> {
    let n = pts.shape()[0];
    let scale = m.get_scale();
    let mut new_pts = Array2::<f32>::zeros((n, 3));
    
    for i in 0..n {
        let pt = [pts[[i, 0]], pts[[i, 1]], 1.0];
        let x = m.matrix[[0, 0]] * pt[0] + m.matrix[[0, 1]] * pt[1] + m.matrix[[0, 2]];
        let y = m.matrix[[1, 0]] * pt[0] + m.matrix[[1, 1]] * pt[1] + m.matrix[[1, 2]];
        new_pts[[i, 0]] = x;
        new_pts[[i, 1]] = y;
        new_pts[[i, 2]] = pts[[i, 2]] * scale;
    }
    
    new_pts
}

// Transform points (2D or 3D) (equivalent to trans_points in Python)
pub fn trans_points(pts: &Array2<f32>, m: &TransformationMatrix) -> Array2<f32> {
    if pts.shape()[1] == 2 {
        trans_points2d(pts, m)
    } else {
        trans_points3d(pts, m)
    }
}
