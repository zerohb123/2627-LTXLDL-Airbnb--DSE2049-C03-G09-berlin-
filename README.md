# Phân tích Thị trường Airbnb Berlin (2025–2026) — Nhóm DSE2049-C03-G09

Quy trình tự động hóa thu thập, kiểm định chất lượng (QA), tính toán KPI không gian – thời gian, phân tích cảm xúc & khía cạnh đánh giá bằng LLM (`gemini-3.5-flash-lite`) và trực quan hóa dữ liệu thị trường Airbnb tại Berlin qua 4 mốc snapshot (`2025-09-23`, `2025-12-27`, `2026-03-28`, `2026-06-26`).

## 1. Yêu cầu môi trường & Cài đặt
* **Phiên bản Python kiểm thử:** xem tệp `.python-version` ở thư mục gốc.
* **Cài đặt môi trường ảo và thư viện (Windows PowerShell):**

    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt

* **Trên Linux / macOS:**

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## 2. Biến môi trường
* **Tên biến môi trường cho khóa Gemini API:** `GEMINI_API_KEY` (xem mẫu trong `.env.example`).
* **Lưu ý khi chấm bài:** Toàn bộ kết quả gọi LLM cho 1.000 mẫu đã được lưu sẵn tại `data/processed/llm_cache.jsonl`. Quy trình chạy đầy đủ **không cần** thiết lập `GEMINI_API_KEY`.

## 3. Lệnh chạy toàn bộ quy trình (1 lệnh)
Chạy từ thư mục gốc của repository:

    python run_pipeline.py --config configs/berlin.yaml

*(Nếu đã có sẵn dữ liệu thô trong `data/raw/`, có thể thêm cờ `--skip-download` để tiết kiệm thời gian tải).*

## 4. Tệp đầu ra để đối chiếu
* **Báo cáo kiểm định QA:** `reports/qa_report.csv`
* **Bảng KPI không gian – thời gian – LLM:** `data/processed/kpi_area_snapshot.csv` (và `reports/kpi.csv`)
* **Kết quả phân tích LLM & Cache:** `data/processed/llm_results.csv`, `data/processed/llm_cache.jsonl`
* **Bảng chấm điểm đối chứng LLM vs. Baseline (100 nhãn tay):** `reports/model_comparison.csv`, `reports/llm_eval.csv`
* **Biểu đồ trực quan hóa (6 hình):** thư mục `figures/` (`1_*.png` đến `6_*.png`)
* **Báo cáo kỹ thuật:** `reports/report.pdf`