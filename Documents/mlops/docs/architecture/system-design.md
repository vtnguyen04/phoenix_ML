# Kiến trúc chi tiết Phoenix ML Platform

## 1. Luồng dữ liệu (Data Flow)

### Quy trình Inference (Dự đoán)
1. **Request**: Client gửi `entity_id` tới endpoint `/predict`.
2. **Enrichment**: `PredictHandler` gọi `FeatureStore` (Redis) để lấy vector đặc trưng của người dùng dựa trên `entity_id`.
3. **Artifact Retrieval**: Hệ thống kiểm tra xem model file đã có trong `model_cache` chưa. Nếu chưa, gọi `ArtifactStorage` để tải model về.
4. **Execution**: `ONNXInferenceEngine` nạp model và thực thi tính toán trên luồng riêng (`asyncio.to_thread`) để không chặn API.
5. **Validation**: Kết quả được đóng gói vào `Prediction` Entity để kiểm tra ràng buộc (ví dụ: confidence score phải từ 0-1).
6. **Async Logging**: Trước khi trả response, một Background Task được kích hoạt để lưu log vào `PredictionLogRepository`.

## 2. Hệ thống Monitoring (Drift Detection)
- **Thuật toán**: Kolmogorov-Smirnov Test.
- **Reference Data**: Được lưu trữ cùng với metadata của model.
- **Production Data**: Lấy từ cửa sổ thời gian gần nhất (Window-based) trong log database.

## 3. Quản lý mô hình (Model Registry)
Hệ thống sử dụng bảng `models` trong Postgres để quản lý metadata:
- `id`, `version`: Định danh duy nhất.
- `uri`: Đường dẫn tới storage (S3/Local).
- `stage`: Trạng thái vòng đời (Development, Staging, Production).
