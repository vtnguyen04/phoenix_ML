use anyhow::Result;
use opencv::{prelude::*, imgcodecs, core::Vector};
use base64::{engine::general_purpose, Engine as _};

mod network;
mod processing;

use network::udp_sender::{send_data, Payload};
use processing::{basic_ops, color_ops, geometric_ops};

fn main() -> Result<()> {
    // --- Configuration ---
    let image_path = "sample.jpg";
    let udp_addr = "127.0.0.1:8080";

    // --- Load Image ---
    let img = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        anyhow::bail!("Failed to load image: {}", image_path);
    }

    // --- Image Processing Pipeline ---
    let gray = basic_ops::to_grayscale(&img)?;
    let equalized = color_ops::equalize_histogram(&gray)?;
    let blurred = basic_ops::blur(&equalized, 5)?;
    let edges = basic_ops::canny(&blurred, 50.0, 150.0)?;
    let resized = geometric_ops::resize(&edges, 224, 224)?;

    // --- Prepare Payload ---
    let mut buf = Vector::<u8>::new();
    imgcodecs::imencode(".jpg", &resized, &mut buf, &Vector::<i32>::new())?;
    let encoded_img = general_purpose::STANDARD.encode(buf.as_slice());
    let payload = Payload {
        image: encoded_img,
    };

    // --- Send Data over UDP ---
    send_data(&payload, udp_addr)?;
    println!("Data sent to {}", udp_addr);

    Ok(())
}
