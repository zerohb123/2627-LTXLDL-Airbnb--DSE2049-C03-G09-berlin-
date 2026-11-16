import os
import argparse
import yaml
import numpy as np
import pandas as pd

def compute_kpis(config_path="configs/berlin.yaml"):
    """Đọc dữ liệu sạch và tính toán các KPI kinh doanh then chốt."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    snapshots = config["snapshots"]
    area_col = config.get("qa", {}).get("area_col", "analysis_area")

    os.makedirs("reports", exist_ok=True)
    all_kpi_records = []

    print(">>> ĐANG XỬ LÝ: Tính toán các chỉ số KPI...")

    for snap in snapshots:
        clean_l_path = f"data/clean/{snap}_listings.csv"

        if not os.path.exists(clean_l_path):
            print(f"  [Cảnh báo] Bỏ qua mốc {snap} do không tìm thấy tệp dữ liệu.")
            continue

        df_l = pd.read_csv(clean_l_path, low_memory=False)

        if 'price_clean' not in df_l.columns:
            df_l['price_clean'] = np.nan

        if 'availability_30' in df_l.columns:
            avail_30 = pd.to_numeric(df_l['availability_30'], errors='coerce').clip(0, 30)
            df_l['booked_nights_30'] = 30.0 - avail_30
            df_l['listing_rev_30'] = df_l['price_clean'] * df_l['booked_nights_30']
        else:
            df_l['booked_nights_30'] = np.nan
            df_l['listing_rev_30'] = np.nan

        for area, group in df_l.groupby(area_col):
            total_listings = len(group)

            if 'room_type' in group.columns:
                entire_homes = int((group['room_type'] == 'Entire home/apt').sum())
                private_rooms = int((group['room_type'] == 'Private room').sum())
                entire_home_ratio = entire_homes / total_listings if total_listings > 0 else 0.0
            else:
                entire_homes, private_rooms, entire_home_ratio = 0, 0, np.nan

            mean_price = group['price_clean'].mean()
            median_price = group['price_clean'].median()

            total_reviews = pd.to_numeric(group.get('number_of_reviews', 0), errors='coerce').sum()
            avg_reviews_per_month = pd.to_numeric(group.get('reviews_per_month', 0), errors='coerce').mean()

            if 'booked_nights_30' in group.columns and group['booked_nights_30'].notna().any():
                est_occ_rate_30 = (group['booked_nights_30'] / 30.0).mean()
                est_revenue_30 = group['listing_rev_30'].sum()
            else:
                est_occ_rate_30 = np.nan
                est_revenue_30 = np.nan

            all_kpi_records.append({
                "snapshot": snap,
                "analysis_area": area,
                "total_listings": total_listings,
                "entire_homes": entire_homes,
                "private_rooms": private_rooms,
                "entire_home_ratio": round(entire_home_ratio, 4) if pd.notna(entire_home_ratio) else np.nan,
                "mean_price": round(mean_price, 2) if pd.notna(mean_price) else np.nan,
                "median_price": round(median_price, 2) if pd.notna(median_price) else np.nan,
                "est_occupancy_rate_30": round(est_occ_rate_30, 4) if pd.notna(est_occ_rate_30) else np.nan,
                "est_revenue_30": round(est_revenue_30, 2) if pd.notna(est_revenue_30) else np.nan,
                "total_reviews": int(total_reviews),
                "avg_reviews_per_month": round(avg_reviews_per_month, 2) if pd.notna(avg_reviews_per_month) else 0.0
            })

    df_kpi = pd.DataFrame(all_kpi_records)
    out_path = "reports/kpi.csv"
    df_kpi.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n>>> HOÀN TẤT: Dữ liệu KPI đã được lưu tại {out_path} ({len(df_kpi)} bản ghi).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tính toán KPI không gian & thời gian Airbnb")
    parser.add_argument("--config", default="configs/berlin.yaml", help="Đường dẫn tới file cấu hình YAML")
    args = parser.parse_args()
    compute_kpis(args.config)