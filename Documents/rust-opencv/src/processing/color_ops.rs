use opencv::{
    imgproc,
    prelude::*,
};

pub fn equalize_histogram(img: &Mat) -> opencv::Result<Mat> {
    let mut equalized = Mat::default();
    imgproc::equalize_hist(img, &mut equalized)?;
    Ok(equalized)
}
