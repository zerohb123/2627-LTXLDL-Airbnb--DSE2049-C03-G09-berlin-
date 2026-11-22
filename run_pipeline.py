import os
import sys
import time
import argparse
import subprocess

PIPELINE_STEPS = [
    ("1. Tải & lọc dữ liệu thô", "src/download.py"),
    ("2. Kiểm tra & làm sạch dữ liệu (QA)", "src/qa.py"),
    ("3. Trích xuất cảm xúc & khía cạnh (LLM)", "src/llm.py"),
    ("4. Chạy phương pháp đối chứng (Baseline)", "src/baseline.py"),
    ("5. Chấm điểm đối chứng trên nhãn tay (Evaluate)", "src/evaluate.py"),
    ("6. Tính toán chỉ số KPI không gian - thời gian - LLM", "src/kpi.py"),
    ("7. Trực quan hóa toàn bộ biểu đồ (Viz)", "src/viz.py"),
]

def run_step(step_name, script_path, config_path):
    if not os.path.exists(script_path):
        print(f"[BỎ QUA] Không tìm thấy tệp {script_path}")
        return

    print(f"\n{'=' * 65}")
    print(f">>> BƯỚC {step_name} ({script_path})")
    print(f"{'=' * 65}")

    start_time = time.time()
    cmd = [sys.executable, script_path, "--config", config_path]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        sys.exit(f"\n[LỖI] Pipeline dừng lại tại bước: {step_name}")

    elapsed = time.time() - start_time
    print(f">>> Hoàn thành '{step_name}' trong {elapsed:.2f} giây.")

def main():
    parser = argparse.ArgumentParser(description="Chạy toàn bộ pipeline phân tích Airbnb Berlin")
    parser.add_argument("--config", default="configs/berlin.yaml", help="Đường dẫn tới tệp cấu hình YAML")
    parser.add_argument("--skip-download", action="store_true", help="Bỏ qua bước tải dữ liệu nếu đã có sẵn")
    parser.add_argument("--skip-llm", action="store_true", help="Bỏ qua bước gọi Gemini API")
    args = parser.parse_args()

    total_start = time.time()
    print(f"BẮT ĐẦU CHẠY PIPELINE VỚI CẤU HÌNH: {args.config}")

    for step_name, script_path in PIPELINE_STEPS:
        if args.skip_download and script_path == "src/download.py":
            print(f"[BỎ QUA] {step_name} (do bật cờ --skip-download)")
            continue
        if args.skip_llm and script_path == "src/llm.py":
            print(f"[BỎ QUA] {step_name} (do bật cờ --skip-llm)")
            continue

        run_step(step_name, script_path, args.config)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 65}")
    print(f">>> TOÀN BỘ PIPELINE ĐÃ HOÀN TẤT THÀNH CÔNG ({total_elapsed:.2f} giây)!")
    print(f"{'=' * 65}")

if __name__ == "__main__":
    main()