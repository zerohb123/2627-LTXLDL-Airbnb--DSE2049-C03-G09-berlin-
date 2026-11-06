import os
import re
import yaml
import numpy as np
import pandas as pd

def clean_currency(val):
    """Chuyển chuỗi tiền tệ ($120.00, €120, ...) thành float."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if not val_str or val_str.lower() == 'nan':
        return np.nan
    clean_val = re.sub(r'[^\d.]', '', val_str.replace(',', ''))
    try:
        return float(clean_val)
    except ValueError:
        return np.nan

def extract_bathrooms(val):
    """Trích xuất số phòng tắm từ chuỗi văn bản bathrooms_text."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    if 'half-bath' in val_str or 'half bath' in val_str:
        return 0.5
    match = re.search(r'(\d+(\.\d+)?)', val_str)
    return float(match.group(1)) if match else np.nan

def run_qa(config_path="configs/berlin.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    snapshots = config["snapshots"]
    nights_cap = config.get("qa", {}).get("nights_cap", 365)
    dedupe_keys = config.get("qa", {}).get("dedupe_keys", ["id", "last_scraped"])

    os.makedirs("data/clean", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    qa_summary = []
    print(">>> BẮT ĐẦU CHẠY PIPELINE QA...")

    # Bộ nhớ lưu lại giá trung vị chuẩn theo từng loại phòng và vùng
    fallback_medians = {}

    for snap in snapshots:
        print(f"\n--- Đang xử lý QA cho mốc: {snap} ---")
        l_path = f"data/processed/{snap}_listings.csv"
        c_path = f"data/processed/{snap}_calendar.csv"

        df_l = pd.read_csv(l_path, low_memory=False, dtype=str)
        df_l['id'] = df_l['id'].astype(str).str.strip()
        initial_rows = len(df_l)

        # 1. Khử trùng lặp
        existing_keys = [k for k in dedupe_keys if k in df_l.columns]
        if existing_keys:
            df_l = df_l.drop_duplicates(subset=existing_keys)
        dedup_dropped = initial_rows - len(df_l)

        # 2. Xử lý giá tiền (Price)
        df_l['price_clean'] = np.nan
        if 'price' in df_l.columns:
            df_l['price_clean'] = df_l['price'].apply(clean_currency)

        # Kiểm tra xem listings có giá hợp lệ hay không
        valid_price_count = df_l['price_clean'].notna().sum()
        
        # Nếu listings bị rỗng giá, tìm trong calendar
        if valid_price_count <= 50 and os.path.exists(c_path):
            print(f"  [Lưu ý] Đang tìm kiếm giá từ bảng calendar cho mốc {snap}...")
            df_c = pd.read_csv(c_path, low_memory=False, dtype=str)
            df_c['listing_id'] = df_c['listing_id'].astype(str).str.strip()
            
            # Tìm xem có cột giá nào tồn tại không
            price_col = None
            for candidate in ['price', 'adjusted_price', 'daily_price', 'rate']:
                if candidate in df_c.columns:
                    price_col = candidate
                    break

            if price_col:
                df_c['price_num'] = df_c[price_col].apply(clean_currency)
                avg_price = df_c.groupby('listing_id')['price_num'].median().dropna().to_dict()
                df_l['price_clean'] = df_l['id'].map(avg_price)
            else:
                print(f"  [Lưu ý] Không có cột giá trong calendar của mốc {snap}.")

        nan_price_before = int(df_l['price_clean'].isna().sum())

        # Điền khuyết nội bộ mốc nếu có ít nhất một phần dữ liệu giá
        if df_l['price_clean'].notna().sum() > 0:
            median_group = df_l.groupby(['room_type', 'analysis_area'])['price_clean'].transform('median')
            overall_median = df_l['price_clean'].median()
            df_l['price_clean'] = df_l['price_clean'].fillna(median_group).fillna(overall_median)

            # Cập nhật vào kho median dự phòng
            for (rt, area), sub in df_l.groupby(['room_type', 'analysis_area']):
                med = sub['price_clean'].median()
                if pd.notna(med):
                    fallback_medians[(rt, area)] = med

        # Điền khuyết liên mốc (mượn median từ mốc có dữ liệu cho mốc rỗng hoàn toàn)
        if df_l['price_clean'].isna().any() and fallback_medians:
            print(f"  [Điền khuyết liên mốc] Áp dụng giá trung vị chuẩn cho mốc {snap}...")
            for (rt, area), med in fallback_medians.items():
                mask = (df_l['room_type'] == rt) & (df_l['analysis_area'] == area) & (df_l['price_clean'].isna())
                df_l.loc[mask, 'price_clean'] = med
            # Điền giá trị trung vị toàn cục nếu còn sót
            global_fallback = np.median(list(fallback_medians.values())) if fallback_medians else 100.0
            df_l['price_clean'] = df_l['price_clean'].fillna(global_fallback)

        nan_price_after = int(df_l['price_clean'].isna().sum())

        # 3. Chuẩn hóa phòng tắm (bathrooms)
        if 'bathrooms_text' in df_l.columns:
            df_l['bathrooms_clean'] = df_l['bathrooms_text'].apply(extract_bathrooms)
            median_bath = df_l.groupby('room_type')['bathrooms_clean'].transform('median').fillna(1.0)
            df_l['bathrooms_clean'] = df_l['bathrooms_clean'].fillna(median_bath)
        else:
            df_l['bathrooms_clean'] = 1.0

        # 4. Giới hạn số đêm thuê (nights_cap)
        for col in ['minimum_nights', 'maximum_nights']:
            if col in df_l.columns:
                df_l[col] = pd.to_numeric(df_l[col], errors='coerce').fillna(1)
                df_l[col] = df_l[col].clip(upper=nights_cap)

        # Lưu file dữ liệu sạch
        clean_l_path = f"data/clean/{snap}_listings.csv"
        df_l.to_csv(clean_l_path, index=False)
        print(f"  -> Đã lưu dữ liệu sạch: {clean_l_path}")

        qa_summary.append({
            "snapshot": snap,
            "initial_rows": initial_rows,
            "dedup_dropped": dedup_dropped,
            "nan_price_before": nan_price_before,
            "nan_price_imputed": nan_price_before - nan_price_after,
            "nan_price_remaining": nan_price_after,
            "nights_capped_max": nights_cap
        })

    # Xuất báo cáo tổng hợp QA
    df_report = pd.DataFrame(qa_summary)
    report_path = "reports/qa_report.csv"
    df_report.to_csv(report_path, index=False)
    print(f"\n>>> ĐÃ TẠO BÁO CÁO QA THÀNH CÔNG: {report_path}")
    print(df_report.to_string())

if __name__ == "__main__":
    run_qa()