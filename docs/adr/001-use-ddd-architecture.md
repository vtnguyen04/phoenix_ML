# ADR 001: Sử dụng kiến trúc Domain-Driven Design (DDD)

## Trạng thái
Đã chấp thuận (Accepted)

## Bối cảnh
Hệ thống ML Inference thường bị sa lầy vào việc trộn lẫn logic toán học (ML) với logic hạ tầng (API, Database). Điều này dẫn đến code khó bảo trì, khó unit test và khó thay đổi engine dự đoán.

## Quyết định
Chúng tôi áp dụng kiến trúc DDD với 3 lớp tách biệt:
1. **Domain Layer**: Chứa "nghiệp vụ" ML (Entities, Value Objects, Drift Logic). Không phụ thuộc vào framework.
2. **Application Layer**: Điều phối luồng (Orchestration). Ví dụ: Lấy features -> Download model -> Predict.
3. **Infrastructure Layer**: Các adapter thực thi (FastAPI, Redis, ONNX Runtime, Postgres).

## Hệ quả
- **Ưu điểm**: 
    - Có thể unit test logic dự đoán mà không cần Database hay API.
    - Dễ dàng thay đổi từ ONNX sang TensorRT chỉ bằng cách thêm một Adapter mới.
    - Code sạch, tuân thủ SOLID.
- **Nhược điểm**: 
    - Cấu trúc thư mục phức tạp hơn cho người mới bắt đầu.
