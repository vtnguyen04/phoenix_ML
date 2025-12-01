use opencv::{
    core,
    imgproc,
    prelude::*,
};

pub fn to_grayscale(img: &Mat) -> opencv::Result<Mat> {
    let mut gray = Mat::default();
    imgproc::cvt_color(img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    Ok(gray)
}

pub fn blur(img: &Mat, ksize: i32) -> opencv::Result<Mat> {
    let mut blurred = Mat::default();
    let size = core::Size::new(ksize, ksize);
    imgproc::gaussian_blur(img, &mut blurred, size, 0.0, 0.0, core::BORDER_DEFAULT)?;
    Ok(blurred)
}

pub fn canny(img: &Mat, threshold1: f64, threshold2: f64) -> opencv::Result<Mat> {
    let mut edges = Mat::default();
    imgproc::canny(img, &mut edges, threshold1, threshold2, 3, false)?;
    Ok(edges)
}
