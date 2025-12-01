use opencv::{
    core,
    imgproc,
    prelude::*,
};

pub fn resize(img: &Mat, width: i32, height: i32) -> opencv::Result<Mat> {
    let mut resized = Mat::default();
    let size = core::Size::new(width, height);
    imgproc::resize(img, &mut resized, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;
    Ok(resized)
}

pub fn rotate(img: &Mat, angle: f64, scale: f64) -> opencv::Result<Mat> {
    let mut rotated = Mat::default();
    let center = core::Point2f::new((img.cols() / 2) as f32, (img.rows() / 2) as f32);
    let rot_mat = imgproc::get_rotation_matrix_2d(center, angle, scale)?;
    imgproc::warp_affine(img, &mut rotated, &rot_mat, img.size()?, imgproc::INTER_LINEAR, core::BORDER_CONSTANT, core::Scalar::default())?;
    Ok(rotated)
}
