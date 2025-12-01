# Rust Installation

This command downloads and executes the `rustup` installer, which installs the Rust toolchain (including `rustc`, `cargo`, and `rustup`).

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
```

- `curl`: A command-line tool for transferring data with URLs.
- `--proto '=https'` and `--tlsv1.2`: These flags ensure a secure connection using HTTPS and TLS version 1.2.
- `-sSf`: These flags make `curl` silent (`-s`), show errors (`-S`), and fail on server errors (`-f`).
- `|`: This is a pipe, which sends the output of the `curl` command to the `sh` command.
- `sh -s -- -y`: This executes the downloaded script. The `-s` flag tells `sh` to read from standard input, and `-y` automatically accepts the default installation options.

# Project Initialization

This command initializes a new Rust project in the current directory.

```bash
cargo init --vcs none
```

- `cargo init`: This command creates a new `Cargo.toml` file and a `src` directory with a `main.rs` file, turning the current directory into a Rust package.
- `--vcs none`: This flag tells `cargo` not to initialize a Git repository.

# Adding Dependencies

The following dependencies were added to the `Cargo.toml` file:

```toml
[dependencies]
opencv = "0.97.2"
ndarray = "0.15.6"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
base64 = "0.22.1"
```

- `opencv`: Provides bindings to the OpenCV library for image processing.
- `ndarray`: A library for N-dimensional arrays, useful for numerical computing.
- `serde`: A framework for serializing and deserializing Rust data structures.
- `serde_json`: A crate for serializing and deserializing data in JSON format.
- `anyhow`: A library for flexible error handling.
- `base64`: A crate for encoding and decoding data in base64.

# Sample Image

A sample image `sample.jpg` is included in the project. You can replace it with your own image.

# Running the Pipeline

To run the pipeline, you need to have both the Rust and Python environments set up.

## 1. Run the Python Receiver

First, install the Python dependencies:

```bash
pip install -r requirements.txt
```

Then, run the Python receiver script:

```bash
python udp_receiver.py
```

The script will wait for data from the Rust application.

## 2. Run the Rust Pipeline

In another terminal, run the Rust application:

```bash
source "$HOME/.cargo/env" && cargo run
```

The Rust application will process the image and send the data to the Python receiver. The Python script will then display the received image.
