# ADR 002: Sử dụng ONNX Runtime làm Inference Engine

## Trạng thái
Đã chấp thuận (Accepted)

## Bối cảnh
Việc sử dụng trực tiếp các framework như PyTorch hay TensorFlow trong production gây ra image size cực lớn (>2GB) và hiệu năng inference không được tối ưu cho CPU/GPU.

## Quyết định
Sử dụng định dạng **ONNX (Open Neural Network Exchange)** và thư viện **ONNX Runtime** để thực thi model.

## Hệ quả
- **Ưu điểm**:
    - Hiệu năng inference cao hơn 2-5 lần so với PyTorch thô.
    - Giảm thiểu image size Docker (loại bỏ dependencies nặng của torch).
    - Hỗ trợ tốt cho cả CPU và GPU mà không cần thay đổi code ứng dụng.
- **Nhược điểm**:
    - Yêu cầu thêm một bước export model từ training pipeline sang định dạng `.onnx`.
