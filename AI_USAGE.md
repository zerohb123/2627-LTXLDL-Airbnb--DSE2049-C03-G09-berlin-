## Phân rã Mức độ Sử dụng AI theo Từng Mô-đun Mã nguồn (`src/`)

Trong toàn bộ dự án, công cụ AI hội thoại (Gemini) được sử dụng với vai trò **trợ lý kỹ thuật (coding assistant)** để tối ưu cú pháp và xử lý ngoại lệ, trong khi **toàn bộ logic nghiệp vụ, tham số cấu hình và kiểm định số liệu** đều do thành viên nhóm tự thiết kế và xác nhận.

### 1. Thu thập dữ liệu (`src/download.py`)
* **Hỗ trợ của AI:** Gợi ý cấu trúc tải tệp tự động qua thư viện `requests`/`urllib`, bổ sung thanh tiến trình (`tqdm`) và cơ chế kiểm tra tệp tồn tại để tránh tải lặp lại nhiều lần.
* **Nhóm tự thực hiện:** Xác định nguồn dữ liệu Inside Airbnb Berlin theo đúng 4 mốc thời gian (`snapshots` trong `configs/berlin.yaml`) và kiểm tra tính toàn vẹn của các tệp `.csv.gz` sau khi giải nén.

### 2. Làm sạch & Kiểm định chất lượng dữ liệu (`src/check_data.py`, `src/qa.py`)
* **Hỗ trợ của AI:** Tư vấn biểu thức chính quy (Regex) để chuẩn hóa cột giá tiền (`price` từ chuỗi ký tự `$` sang số thực) và tối ưu hóa các phép kiểm tra giá trị khuyết (`NaN`), trùng lặp khóa chính bằng `pandas`.
* **Nhóm tự thực hiện:** Đặt ra các quy tắc nghiệp vụ kiểm thử QA (ví dụ: loại bỏ giá phòng $\le 0$, kiểm tra miền giá trị hợp lệ của tọa độ địa lý Berlin, tỷ lệ đánh giá trống) và đối chiếu log cảnh báo thực tế.

### 3. Tính toán chỉ số kinh doanh (`src/kpi.py`)
* **Hỗ trợ của AI:** Gợi ý cách viết các hàm tổng hợp nhóm (`groupby`, `agg`) nhanh gọn và xử lý trường hợp chia cho 0 khi tính tỷ lệ lấp đầy hoặc biến động giá theo mùa.
* **Nhóm tự thực hiện:** Định nghĩa công thức các chỉ số KPI cốt lõi (giá trung bình/trung vị theo quận và loại phòng, tỷ trọng Superhost, mức độ tập trung nguồn cung) và đối chiếu chéo số liệu đầu ra.

### 4. Trực quan hóa dữ liệu khám phá (`src/viz.py`)
* **Hỗ trợ của AI:** Hỗ trợ tinh chỉnh thẩm mỹ biểu đồ `matplotlib`/`seaborn` (căn chỉnh nhãn trục tránh đè chữ, bảng màu, định dạng đơn vị tiền tệ EUR và xuất ảnh độ phân giải cao).
* **Nhóm tự thực hiện:** Lựa chọn loại biểu đồ phù hợp với từng câu hỏi phân tích và tự rút trích các nhận định về xu hướng mùa vụ (mùa hè cao điểm, mùa đông thấp điểm) của thị trường Berlin.

### 5. Phân tích Cảm xúc & Khía cạnh bằng LLM (`src/llm.py`)
* **Ứng dụng AI làm mô hình lõi:** Tích hợp API `gemini-3.5-flash-lite` (`google-genai`) để phân loại đa ngôn ngữ cho 1.000 đánh giá theo 3 nhãn cảm xúc và 7 khía cạnh.
* **Hỗ trợ của AI về kỹ thuật:** Tư vấn thiết kế `Pydantic Schema` (`response_schema`) để ép kiểu JSON đầu ra, xây dựng cơ chế gom nhóm **30 reviews/batch** kèm lưu đệm `llm_cache.json` nhằm tuân thủ giới hạn 15 RPM của Free Tier.
* **Nhóm tự thực hiện:** Tự gán nhãn thủ công 100 mẫu chuẩn (`data/samples/labeled_100.csv`) để làm tập đối chứng độc lập, chạy đánh giá đạt `Sentiment Accuracy = 0.9800`, `Sentiment Macro-F1 = 0.8698` và `Aspects Macro-F1 = 0.8555`.

### 6. Tổng hợp kết quả LLM & Điều phối Pipeline (`src/analyze_llm.py`, `run_pipeline.py`)
* **Hỗ trợ của AI:** Gợi ý khung mã nguồn tách nhãn đa trị (`aspects_pred`) để tính tần suất từng khía cạnh, vẽ biểu đồ thanh ngang chồng (stacked bar chart) và cấu trúc `subprocess` điều phối tuần tự 7 bước trong `run_pipeline.py`.
* **Nhóm tự thực hiện:** Kiểm thử chạy toàn trình từ đầu đến cuối trên môi trường thực tế, xử lý lỗi biến môi trường và tổng hợp insight phục vụ báo cáo cuối kỳ.

