import os
import argparse
import yaml
import numpy as np
import pandas as pd

def compute_kpis(config_path="configs/berlin.yaml"):
    """Đọc dữ liệu sạch + kết quả LLM và tính toán các KPI không gian - thời gian."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    snapshots = config["snapshots"]
    area_col = config.get("qa", {}).get("area_col", "analysis_area")

    # Đọc kết quả phân tích LLM để tích hợp KPI theo khu vực
    llm_res_path = "data/processed/llm_results.csv"
    df_llm = None
    if os.path.exists(llm_res_path):
        df_llm = pd.read_csv(llm_res_path)
        df_llm["listing_id"] = df_llm["listing_id"].astype(str).str.strip()
        df_llm["is_positive"] = (df_llm["sentiment_pred"].astype(str).str.lower() == "positive").astype(float)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    all_kpi_records = []

    print(">>> ĐANG XỬ LÝ: Tính toán các chỉ số KPI...")

    for snap in snapshots:
        clean_l_path = f"data/clean/{snap}_listings.csv"
        if not os.path.exists(clean_l_path):
            print(f"  [Cảnh báo] Bỏ qua mốc {snap} do không tìm thấy tệp dữ liệu.")
            continue

        df_l = pd.read_csv(clean_l_path, low_memory=False)
        df_l["id"] = df_l["id"].astype(str).str.strip()

        # Ghép điểm cảm xúc tích cực trung bình của từng listing từ LLM (nếu có)
        if df_llm is not None:
            llm_by_listing = df_llm.groupby("listing_id").agg(
                llm_sample_reviews=("id", "count"),
                llm_pos_share=("is_positive", "mean")
            ).reset_index()
            df_l = df_l.merge(llm_by_listing, left_on="id", right_on="listing_id", how="left")
        else:
            df_l["llm_sample_reviews"] = 0
            df_l["llm_pos_share"] = np.nan

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

            llm_rev_cnt = int(group["llm_sample_reviews"].fillna(0).sum())
            llm_pos_rate = group["llm_pos_share"].mean()

            all_kpi_records.append({
                "snapshot": snap,
                "analysis_area": area,
                "total_listings": total_listings,
                "entire_homes": entire_homes,
                "private_rooms": private_rooms,
                "entire_home_ratio": round(entire_home_ratio, 4) if pd.notna(entire_home_ratio) else np.nan,
                "mean_price_eur": round(mean_price, 2) if pd.notna(mean_price) else np.nan,
                "median_price_eur": round(median_price, 2) if pd.notna(median_price) else np.nan,
                "est_occupancy_rate_30": round(est_occ_rate_30, 4) if pd.notna(est_occ_rate_30) else np.nan,
                "est_revenue_30_eur": round(est_revenue_30, 2) if pd.notna(est_revenue_30) else np.nan,
                "total_reviews": int(total_reviews),
                "avg_reviews_per_month": round(avg_reviews_per_month, 2) if pd.notna(avg_reviews_per_month) else 0.0,
                "llm_analyzed_reviews": llm_rev_cnt,
                "llm_positive_rate": round(llm_pos_rate, 4) if pd.notna(llm_pos_rate) else np.nan
            })

    df_kpi = pd.DataFrame(all_kpi_records)
    df_kpi.to_csv("data/processed/kpi_area_snapshot.csv", index=False, encoding="utf-8")
    df_kpi.to_csv("reports/kpi.csv", index=False, encoding="utf-8")
    print(f"\n>>> HOÀN TẤT: Đã lưu KPI tại data/processed/kpi_area_snapshot.csv ({len(df_kpi)} bản ghi).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tính toán KPI không gian, thời gian & tích hợp LLM")
    parser.add_argument("--config", default="configs/berlin.yaml", help="Đường dẫn tới file cấu hình YAML")
    args = parser.parse_args()
    compute_kpis(args.config)