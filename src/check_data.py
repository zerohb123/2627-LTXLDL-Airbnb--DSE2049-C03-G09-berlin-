import os
import pandas as pd
import yaml

def generate_check_report():
    with open("configs/berlin.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Mã dữ liệu của nhóm theo sheet Phân công
    ma_du_lieu = "DSE2049-C02-G09-berlin"
    snapshots = config["snapshots"]
    
    rows = []
    nan_90_cols_per_snap = {}
    
    print(">>> Đang kiểm tra và tổng hợp số liệu 4 mốc...")
    
    for moc_idx, snap in enumerate(snapshots, start=1):
        raw_l = f"data/raw/{snap}_listings.csv.gz"
        raw_c = f"data/raw/{snap}_calendar.csv.gz"
        raw_r = f"data/raw/{snap}_reviews.csv.gz"
        
        proc_l = f"data/processed/{snap}_listings.csv"
        proc_c = f"data/processed/{snap}_calendar.csv"
        proc_r = f"data/processed/{snap}_reviews.csv"
        
        # 1. Kích thước file nén (byte)
        l_bytes = os.path.getsize(raw_l) if os.path.exists(raw_l) else ""
        c_bytes = os.path.getsize(raw_c) if os.path.exists(raw_c) else ""
        r_bytes = os.path.getsize(raw_r) if os.path.exists(raw_r) else ""
        
        # 2. Tải về ok?
        tai_ve_ok = "co" if (os.path.exists(raw_l) and os.path.exists(raw_c) and os.path.exists(raw_r)) else "khong"
        
        # 3. Đọc dữ liệu đã lọc
        df_l = pd.read_csv(proc_l, low_memory=False)
        df_c = pd.read_csv(proc_c, low_memory=False)
        df_r = pd.read_csv(proc_r, low_memory=False)
        
        listings_dong = len(df_l)
        calendar_dong = len(df_c)
        reviews_dong = len(df_r)
        
        # Đếm loại phòng
        entire_home = int((df_l['room_type'] == 'Entire home/apt').sum()) if 'room_type' in df_l.columns else 0
        private_room = int((df_l['room_type'] == 'Private room').sum()) if 'room_type' in df_l.columns else 0
        
        # Đếm 0 review và thiếu giá
        khong_co_review = int((pd.to_numeric(df_l['number_of_reviews'], errors='coerce') == 0).sum()) if 'number_of_reviews' in df_l.columns else 0
        thieu_price = int(df_l['price'].isna().sum()) if 'price' in df_l.columns else 0
        
        so_cot_listings = len(df_l.columns)
        
        # Tìm cột NaN >= 90%
        nan_ratio = df_l.isna().mean()
        nan_90_cols = nan_ratio[nan_ratio >= 0.9].index.tolist()
        so_cot_nan_90 = len(nan_90_cols)
        nan_90_cols_per_snap[snap] = nan_90_cols
        
        rows.append({
            "ma_du_lieu": ma_du_lieu,
            "moc": moc_idx,
            "ngay": snap,
            "tai_ve_ok": tai_ve_ok,
            "listings_bytes": l_bytes,
            "calendar_bytes": c_bytes,
            "reviews_bytes": r_bytes,
            "listings_dong": listings_dong,
            "calendar_dong": calendar_dong,
            "reviews_dong": reviews_dong,
            "entire_home": entire_home,
            "private_room": private_room,
            "khong_co_review": khong_co_review,
            "thieu_price": thieu_price,
            "so_cot_listings": so_cot_listings,
            "so_cot_nan_90": so_cot_nan_90
        })
        
    df_check = pd.DataFrame(rows)
    out_file = f"check_{ma_du_lieu}.csv"
    df_check.to_csv(out_file, index=False, encoding="utf-8")
    
    print(f"\n>>> ĐÃ TẠO THÀNH CÔNG: {out_file}\n")
    print(df_check.to_string())
    
    print("\n" + "="*50)
    print("THÔNG TIN TRẢ LỜI CÂU HỎI 3 (CÁC CỘT >= 90% NaN):")
    print("="*50)
    for s, cols in nan_90_cols_per_snap.items():
        print(f"- Mốc {s} ({len(cols)} cột): {', '.join(cols)}")

if __name__ == "__main__":
    generate_check_report()