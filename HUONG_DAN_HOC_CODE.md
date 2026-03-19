# 🗺️ LỘ TRÌNH TỰ HỌC SOURCE CODE PHOENIX ML PLATFORM (A-Z)

Chào mừng bạn đến với Phoenix ML Platform - một hệ thống MLOps tự chữa lành (Self-healing) cực kỳ chuyên nghiệp. Đừng lo lắng nếu bạn cảm thấy choáng ngợp, tài liệu này sẽ dẫn dắt bạn đi từ những khái niệm cơ bản nhất.

---

## 🌟 CẤP ĐỘ 1: BỨC TRANH TỔNG QUAN (CONCEPT)
**Mục tiêu:** Hiểu hệ thống này làm cái gì?

1.  **Dự án này là gì?** Nó là một "nhà máy" vận hành các mô hình trí tuệ nhân tạo (AI). Nó không chỉ chạy AI mà còn canh gác cho AI (nếu AI dự đoán sai quá nhiều, nó sẽ tự động cảnh báo và chạy lại quy trình huấn luyện).
2.  **Từ khóa quan trọng:**
    *   **Inference:** Chạy mô hình AI để đưa ra dự đoán (ví dụ: đưa vào ảnh -> AI trả về "Đây là con mèo").
    *   **Drift Detection:** "Bệnh của AI". Khi dữ liệu thực tế khác quá xa dữ liệu AI đã học, AI sẽ bị "trôi" (drift) và dự đoán sai. Hệ thống này có "cảm biến" để phát hiện điều đó.

---

## 🏗️ CẤP ĐỘ 2: KIẾN TRÚC 3 LỚP (ARCHITECTURAL LAYERS)
**Mục tiêu:** Hiểu tại sao code lại chia ra thành `domain`, `application`, `infrastructure`.

Dự án sử dụng kiến trúc **DDD (Domain-Driven Design)**. Hãy tưởng tượng nó như một nhà hàng:

1.  **Lớp Domain (`src/domain/`):** **Công thức nấu ăn.** 
    *   Chứa những logic cốt lõi nhất. Ví dụ: Một món ăn gồm những nguyên liệu gì? (Tương ứng: Một Model AI gồm ID, Version, Framework nào?).
    *   *File tiêu biểu:* `src/domain/inference/entities/model.py`.
2.  **Lớp Application (`src/application/`):** **Bếp trưởng.** 
    *   Điều phối mọi thứ. Nhận đơn hàng, gọi nguyên liệu từ kho, bảo lớp Domain nấu, rồi gửi đi.
    *   *File tiêu biểu:* `src/application/handlers/predict_handler.py`.
3.  **Lớp Infrastructure (`src/infrastructure/`):** **Các dụng cụ nhà bếp.**
    *   Code để kết nối Database (Tủ lạnh), code chạy GPU (Bếp điện), code gửi thông báo (Nhân viên giao hàng).
    *   *File tiêu biểu:* `src/infrastructure/ml_engines/onnx_executor.py`.

---

## 🔄 CẤP ĐỘ 3: LUỒNG DỮ LIỆU (DATA FLOW)
**Mục tiêu:** Hiểu một yêu cầu chạy như thế nào trong code.

Hãy đọc code theo trình tự một "Request" từ khách hàng:
1.  **Cửa ngõ:** `src/infrastructure/http/fastapi_server.py` tiếp nhận yêu cầu.
2.  **Giao việc:** Nó tạo ra một `PredictCommand` và gửi cho `predict_handler.py`.
3.  **Lấy dữ liệu:** Handler này gọi `infrastructure` để lấy dữ liệu từ "Feature Store" (Redis).
4.  **Chạy AI:** Handler gọi `infrastructure` để chạy mô hình qua `ONNXEngine` hoặc `TensorRTEngine`.
5.  **Trả kết quả:** Kết quả được đóng gói vào `PredictionResponse` và trả lại cho khách hàng.

---

## 🛡️ CẤP ĐỘ 4: TÍNH NĂNG TỰ CHỮA LÀNH (SELF-HEALING)
**Mục tiêu:** Hiểu các "cảm biến" thông minh của hệ thống.

Đây là phần "xịn" nhất của dự án:
*   **Drift Detection:** Xem `src/domain/monitoring/services/drift_detector.py`. Code này dùng toán học (thống kê) để so sánh dữ liệu hiện tại với dữ liệu cũ.
*   **Auto-Retrain:** Nếu phát hiện lỗi, hệ thống sẽ gửi một tin nhắn qua **Kafka** (một hàng đợi tin nhắn) để kích hoạt script `scripts/train_model.py` chạy lại.

---

## 🚀 CẤP ĐỘ 5: THỰC HÀNH & CHẠY THỬ (HANDS-ON)
**Mục tiêu:** Thấy tận mắt hệ thống hoạt động.

Bạn có thể chạy các script giả lập sau để hiểu code:
1.  **Giả lập khách hàng:** `python scripts/simulate_traffic.py` (tạo ra các yêu cầu dự đoán liên tục).
2.  **Giả lập lỗi dữ liệu:** `python scripts/simulate_drift.py` (làm cho dữ liệu bị sai lệch để xem hệ thống cảnh báo).
3.  **Xem Dashboard:** Vào thư mục `frontend/`, chạy `npm run dev` để xem biểu đồ theo dõi AI.

---

### 💡 LỜI KHUYÊN CHO BẠN:
*   **Đừng đọc hết 100% file.** Hãy tập trung vào các file trong `src/application/handlers/` trước, vì đó là nơi "điều binh khiển tướng".
*   **Sử dụng Debugger:** Đặt điểm dừng (breakpoint) tại `predict_handler.py` và bấm chạy để xem dữ liệu nhảy qua từng lớp như thế nào.
*   **Hỏi tôi:** Nếu gặp một dòng code khó hiểu (ví dụ: `@abstractmethod` hay `asyncio.gather`), hãy hỏi tôi ngay, tôi sẽ giải thích bằng ví dụ đời thường!

Chúc bạn chinh phục được hệ thống Phoenix ML này! 🚀
