use serde::{Deserialize, Serialize};
use std::net::UdpSocket;

#[derive(Serialize, Deserialize, Debug)]
pub struct Payload {
    pub image: String,
}

pub fn send_data(payload: &Payload, addr: &str) -> anyhow::Result<()> {
    let socket = UdpSocket::bind("0.0.0.0:0")?;
    let encoded = serde_json::to_string(payload)?;
    socket.send_to(encoded.as_bytes(), addr)?;
    Ok(())
}
