# Khai báo Minh bạch Sử dụng AI (AI_USAGE.md)

## 1. Công cụ AI đã dùng (vào việc gì)
* **Gemini (giao diện chat):**
  * **Thiết kế & tối ưu trực quan hóa (`src/viz.py`):** Gợi ý lựa chọn loại biểu đồ phù hợp cho từng góc độ phân tích (không gian, thời gian, kết quả LLM), viết mã ánh xạ dữ liệu khu vực sang bản đồ ranh giới `neighbourhoods.geojson` bằng `geopandas`, xử lý chuẩn hóa kích thước bong bóng (`scatter`) và tinh chỉnh bố cục `matplotlib`/`seaborn` (bảng màu, chú giải ngoài khung, định dạng tiền tệ EUR).
  * **Xây dựng & kiểm thử quy trình dữ liệu (`src/download.py`, `src/qa.py`, `src/kpi.py`, `src/baseline.py`, `src/evaluate.py`, `run_pipeline.py`):** Hỗ trợ rà soát cú pháp `pandas`, tư vấn biểu thức chính quy (Regex) làm sạch tiền tệ/số phòng tắm, thiết kế cơ chế điền khuyết liên mốc (`fallback_medians`), xây dựng mô hình đối chứng từ điển Anh – Đức và đóng gói pipeline chạy một lệnh qua `subprocess`.
* **Gemini API (`gemini-3.5-flash-lite` qua thư viện `google-genai`):**
  * Dùng làm mô hình lõi trong `src/llm.py` để trích xuất có cấu trúc (Structured Outputs với `Pydantic`) cảm xúc, khía cạnh khen/chê và ngôn ngữ cho 1.000 đánh giá Airbnb Berlin.

## 2. Các Prompt then chốt đã sử dụng
1. **Prompt gợi ý và thiết kế hệ thống biểu đồ (`src/viz.py`):** Yêu cầu đề xuất các dạng biểu đồ thể hiện đồng thời 3 chiều không gian, thời gian và phân tích cảm xúc LLM từ bảng KPI; hướng dẫn ánh xạ mã vùng `A01–A08` sang tên quận Berlin trong tệp `neighbourhoods.geojson` và định dạng trục Y theo đơn vị nghìn Euro (`K`).
2. **Prompt tinh chỉnh thẩm mỹ & khắc phục lỗi chồng chéo nhãn biểu đồ (`src/viz.py`):** Yêu cầu xử lý lỗi tràn chú giải (legend) ở biểu đồ chuỗi thời gian và biểu đồ bong bóng (`bbox_to_anchor`), thêm nhãn giá trị trực tiếp trên đỉnh cột biểu đồ giá trung bình và vẽ biểu đồ thanh ngang chồng 100% (stacked bar chart) phân bố cảm xúc theo từng khía cạnh LLM.
3. **Prompt trích xuất cảm xúc & khía cạnh (`src/llm.py`):** Yêu cầu phân loại từng review đa ngôn ngữ (Anh/Đức) vào 3 nhãn `positive / neutral / negative`, ánh xạ đa nhãn vào đúng danh sách 7 khía cạnh (`vị trí`, `sạch sẽ`, `chủ nhà`, `giá trị`, `tiện nghi`, `độ ồn`, `an toàn`), trả về JSON tuân thủ schema và xử lý theo lô 30 reviews/batch.
4. **Prompt xử lý khuyết dữ liệu giá liên mốc (`src/qa.py`):** Thiết kế cơ chế lưu giá trung vị dự phòng (`fallback_medians`) theo cặp `(room_type, analysis_area)` từ mốc `2025-09-23` để điền khuyết cho hai mốc `2025-12-27` và `2026-03-28` bị thiếu 100% giá gốc.
5. **Prompt xây dựng phương pháp đối chứng (`src/baseline.py`):** Xây dựng bộ từ điển từ khóa song ngữ Anh – Đức (Lexicon & Rule-based) làm mốc so sánh không dùng LLM.
6. **Prompt đánh giá chéo đa nhãn (`src/evaluate.py`):** Sử dụng `MultiLabelBinarizer` và `classification_report` của `scikit-learn` để tính `Accuracy` và `Macro-F1` đồng thời cho cả LLM và Baseline.
7. **Prompt đảm bảo tính tái lập không cần API Key (`src/llm.py`, `run_pipeline.py`):** Thiết kế cơ chế lưu đệm `data/processed/llm_cache.jsonl` và cờ `--skip-llm` để pipeline chạy thông suốt khi chấm tự động mà không cần `GEMINI_API_KEY`.

## 3. AI sai ở đâu (Ví dụ thực tế từ quá trình làm bài)
* **Sai sót trong mã trực quan hóa (`src/viz.py`):**
  * Ở phiên bản đầu của biểu đồ bong bóng (`3_scatter_occupancy_revenue.png`), AI dùng tỷ lệ kích thước tuyến tính mặc định khiến các bong bóng lớn đè kín nhau và phần chú giải (`legend`) liệt kê quá nhiều mốc số lượng phòng gây tràn khung hình. Nhóm đã sửa sang hàm chuẩn hóa bậc hai (`norm_val ** 2`) và lọc chú giải chỉ giữ 3 mốc đại diện (`min`, `median`, `max`).
  * Ở biểu đồ bản đồ (`4_geojson_map.png`), AI ban đầu ghép trực tiếp mã `A01–A08` với cột `neighbourhood_group` trong `neighbourhoods.geojson` khiến toàn bộ bản đồ bị trống dữ liệu; nhóm đã phát hiện và bổ sung bảng ánh xạ `area_mapping` sang tên quận hành chính của Berlin (`Mitte`, `Pankow`,...).
* **Sai sót trong mã xử lý dữ liệu (`src/download.py`, `src/llm.py`):**
  * Ban đầu `src/download.py` tự động ghi đè tệp `data/samples/labeled_100.csv` mỗi khi chạy lại pipeline, làm mất dữ liệu nhóm đã gán nhãn tay; đồng thời `src/llm.py` báo lỗi `EnvironmentError` khi chạy trên máy không đặt `GEMINI_API_KEY`. Nhóm đã bổ sung điều kiện kiểm tra tồn tại tệp và chế độ đọc từ cache `.jsonl`.
* **Sai sót trong phân loại của mô hình LLM:**
  * Một số đánh giá ngắn mang tính mỉa mai hoặc khen chê đan xen (ví dụ: phòng sạch sẽ nhưng cầu thang bộ quá cao) đôi khi bị LLM gán nhãn `positive` thay vì `neutral`, hoặc bỏ sót khía cạnh `giá trị` khi khách chỉ nhắc gián tiếp bằng từ *"worth every cent"*.

## 4. Nhóm kiểm chứng như thế nào
* **Kiểm chứng trực quan hóa (`figures/`):** Đối chiếu từng con số hiển thị trên đỉnh cột và trục tọa độ của cả 6 biểu đồ (`1_time_series_revenue.png` đến `6_llm_aspect_sentiment_breakdown.png`) với bảng số liệu gốc trong `data/processed/kpi_area_snapshot.csv` và `reports/llm_aspect_summary.csv`, đảm bảo không sai lệch số liệu và không lỗi hiển thị tiếng Việt/ký hiệu tiền tệ `€`.
* **Kiểm chứng dữ liệu & QA:** Kiểm tra bảng `reports/qa_report.csv` xác nhận 0 dòng trùng lặp, 0 giá trị `NaN` còn sót ở cột `price_clean` trên cả 4.000 bản ghi thuộc 4 mốc thời gian.
* **Kiểm chứng định lượng LLM trên 100 mẫu gán nhãn tay (`data/samples/labeled_100.csv`):** Chạy đối chứng độc lập qua `src/evaluate.py` (`reports/model_comparison.csv`):
  * **LLM (`gemini-3.5-flash-lite`):** Sentiment Accuracy = **0.9700**, Sentiment Macro-F1 = **0.8255**, Aspects Macro-F1 = **0.8471**.
  * **Baseline (Lexicon Anh – Đức):** Sentiment Accuracy = **0.9300**, Sentiment Macro-F1 = **0.4045**, Aspects Macro-F1 = **0.7221**.
* **Kiểm chứng tái lập toàn trình:** Xóa biến môi trường `GEMINI_API_KEY` và chạy lại một lệnh `python run_pipeline.py --config configs/berlin.yaml --skip-download` để xác nhận toàn bộ bảng biểu và 6 hình trong `figures/` được tái tạo chính xác.