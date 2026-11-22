import os
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

def format_k(x, pos):
    """Định dạng số ngàn (K) cho trục Y"""
    return f'{int(x/1000)}K' if x >= 1000 else f'{int(x)}'

def create_visualizations(config_path="configs/berlin.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kpi_path = "data/processed/kpi_area_snapshot.csv" if os.path.exists("data/processed/kpi_area_snapshot.csv") else "reports/kpi.csv"
    out_dir = "figures"
    geojson_path = "data/raw/neighbourhoods.geojson"

    if not os.path.exists(kpi_path):
        print(f"Lỗi: Không tìm thấy {kpi_path}.")
        return

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    df = pd.read_csv(kpi_path)

    # Tương thích tên cột cũ và mới
    rev_col = "est_revenue_30_eur" if "est_revenue_30_eur" in df.columns else "est_revenue_30"
    price_col = "mean_price_eur" if "mean_price_eur" in df.columns else "mean_price"

    sorted_areas = sorted(df['analysis_area'].unique())
    sns.set_theme(style="whitegrid", context="talk")
    custom_palette = sns.color_palette("tab10", len(sorted_areas))

    print(">>> ĐANG XỬ LÝ: Kết xuất các biểu đồ không gian - thời gian...")

    # 1. BIỂU ĐỒ CHUỖI THỜI GIAN
    plt.figure(figsize=(12, 7))
    ax1 = sns.lineplot(
        data=df, x='snapshot', y=rev_col, hue='analysis_area',
        hue_order=sorted_areas, marker='o', markersize=9, linewidth=3.5, palette=custom_palette
    )
    plt.title('Chuỗi thời gian: Doanh thu ước tính theo Vùng', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Mốc thời gian (Snapshot)', fontsize=13, fontweight='bold')
    plt.ylabel('Doanh thu ước tính (€)', fontsize=13, fontweight='bold')
    ax1.yaxis.set_major_formatter(FuncFormatter(format_k))
    plt.legend(title='Vùng (Area)', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/1_time_series_revenue.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. BIỂU ĐỒ GIÁ TRUNG BÌNH THEO VÙNG Ở MỐC MỚI NHẤT
    latest_snap = df['snapshot'].max()
    df_latest = df[df['snapshot'] == latest_snap].copy()

    plt.figure(figsize=(12, 7))
    ax2 = sns.barplot(
        data=df_latest, x='analysis_area', y=price_col,
        hue='analysis_area', hue_order=sorted_areas,
        palette=custom_palette, legend=False
    )
    plt.title(f'Giá trung bình theo Vùng (Mốc {latest_snap})', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Vùng (Area)', fontsize=13, fontweight='bold')
    plt.ylabel('Giá trung bình (€)', fontsize=13, fontweight='bold')

    for p in ax2.patches:
        ax2.annotate(
            f"{p.get_height():.0f}€", (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', fontsize=12, color='black', xytext=(0, 8), textcoords='offset points', fontweight='bold'
        )
    plt.tight_layout()
    plt.savefig(f"{out_dir}/2_aspect_mean_price.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. BIỂU ĐỒ BONG BÓNG: QUAN HỆ TỈ LỆ LẤP ĐẦY VÀ DOANH THU
    plt.figure(figsize=(15, 8.5))
    df_sorted = df.sort_values(by='total_listings', ascending=False)

    unique_listings = sorted(df['total_listings'].unique())
    min_size, max_size = 150, 2500
    size_dict = {}

    if len(unique_listings) > 1:
        v_min, v_max = min(unique_listings), max(unique_listings)
        for val in unique_listings:
            norm_val = ((val - v_min) / (v_max - v_min)) ** 2
            size_dict[val] = min_size + norm_val * (max_size - min_size)
    else:
        size_dict = {unique_listings[0]: 1000}

    ax3 = sns.scatterplot(
        data=df_sorted, x='est_occupancy_rate_30', y=rev_col,
        hue='analysis_area', size='total_listings',
        sizes=size_dict, hue_order=sorted_areas,
        alpha=0.7, palette=custom_palette,
        edgecolor='black', linewidth=1.5
    )

    plt.title('Quan hệ Tỉ lệ lấp đầy & Doanh thu ước tính', fontsize=17, fontweight='bold', pad=20)
    plt.xlabel('Tỉ lệ lấp đầy 30 ngày', fontsize=13, fontweight='bold')
    plt.ylabel('Doanh thu ước tính (€)', fontsize=13, fontweight='bold')
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax3.yaxis.set_major_formatter(FuncFormatter(format_k))

    handles, labels = ax3.get_legend_handles_labels()
    new_labels, new_handles = [], []
    if len(unique_listings) >= 3:
        size_values_to_keep = [
            str(min(unique_listings)),
            str(unique_listings[len(unique_listings)//2]),
            str(max(unique_listings))
        ]
    else:
        size_values_to_keep = [str(x) for x in unique_listings]

    for h, label in zip(handles, labels):
        if label == 'analysis_area':
            new_labels.append('Vùng (Area)')
            new_handles.append(h)
        elif label == 'total_listings':
            new_labels.append('Số lượng Listing')
            new_handles.append(h)
        elif label.isdigit():
            if label in size_values_to_keep:
                new_labels.append(label)
                new_handles.append(h)
        else:
            new_labels.append(label)
            new_handles.append(h)

    plt.legend(
        handles=new_handles, labels=new_labels, bbox_to_anchor=(1.02, 1), loc='upper left',
        fontsize=11, labelspacing=1.8, handletextpad=1.2, borderpad=1.2
    )
    plt.tight_layout()
    plt.savefig(f"{out_dir}/3_scatter_occupancy_revenue.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. BẢN ĐỒ GEOJSON
    if os.path.exists(geojson_path):
        try:
            import geopandas as gpd
            area_mapping = {
                'A01': 'Mitte', 'A02': 'Pankow', 'A03': 'Friedrichshain-Kreuzberg',
                'A04': 'Neukölln', 'A05': 'Charlottenburg-Wilm.', 'A06': 'Tempelhof - Schöneberg',
                'A07': 'Lichtenberg', 'A08': 'Treptow - Köpenick'
            }
            df_latest['district_name'] = df_latest['analysis_area'].map(area_mapping)

            gdf = gpd.read_file(geojson_path)
            merge_col = 'neighbourhood_group' if 'neighbourhood_group' in gdf.columns else 'neighbourhood'
            gdf_merged = gdf.merge(df_latest, left_on=merge_col, right_on='district_name', how='left')

            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            gdf.plot(ax=ax, color='#f0f0f0', edgecolor='white', linewidth=0.8)

            gdf_valid = gdf_merged.dropna(subset=[price_col])
            if not gdf_valid.empty:
                gdf_valid.plot(
                    column=price_col, cmap='YlOrRd', linewidth=1.5, ax=ax, edgecolor='#222222',
                    legend=True, legend_kwds={'label': "Giá trung bình (€)", 'shrink': 0.75}
                )

            plt.title(f'Bản đồ Không gian: Giá trung bình khu vực Berlin ({latest_snap})', fontsize=18, fontweight='bold', pad=20)
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(f"{out_dir}/4_geojson_map.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  [Lỗi] Không thể vẽ bản đồ GeoJSON: {e}")

    # 5 & 6. BIỂU ĐỒ KẾT QUẢ PHÂN TÍCH LLM (GỘP TỪ analyze_llm.py)
    llm_res_path = "data/processed/llm_results.csv" if os.path.exists("data/processed/llm_results.csv") else "reports/llm_results.csv"
    if os.path.exists(llm_res_path):
        print(">>> ĐANG XỬ LÝ: Kết xuất biểu đồ phân tích LLM...")
        sns.reset_defaults()
        allowed_aspects = config.get("llm", {}).get(
            "aspects", ["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn"]
        )
        df_llm = pd.read_csv(llm_res_path)

        sent_counts = df_llm["sentiment_pred"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
        sent_pct = (sent_counts / len(df_llm) * 100).round(2)
        pd.DataFrame({"sentiment": sent_counts.index, "count": sent_counts.values, "percentage": sent_pct.values}).to_csv(
            "reports/llm_sentiment_summary.csv", index=False, encoding="utf-8"
        )

        aspect_rows = []
        for asp in allowed_aspects:
            mask = df_llm["aspects_pred"].fillna("").apply(lambda s: asp in [x.strip() for x in str(s).split(",")])
            sub = df_llm[mask]
            total = len(sub)
            pos = (sub["sentiment_pred"] == "positive").sum()
            neu = (sub["sentiment_pred"] == "neutral").sum()
            neg = (sub["sentiment_pred"] == "negative").sum()
            aspect_rows.append({
                "aspect": asp,
                "total_mentions": total,
                "mention_rate_pct": round(total / len(df_llm) * 100, 2),
                "positive": pos,
                "neutral": neu,
                "negative": neg,
                "positive_rate_pct": round(pos / total * 100, 2) if total > 0 else 0.0,
                "negative_rate_pct": round(neg / total * 100, 2) if total > 0 else 0.0,
            })

        df_aspect_summary = pd.DataFrame(aspect_rows).sort_values("total_mentions", ascending=False)
        df_aspect_summary.to_csv("reports/llm_aspect_summary.csv", index=False, encoding="utf-8")

        # Biểu đồ 5: Tần suất 7 khía cạnh
        plt.figure(figsize=(9, 5))
        df_plot = df_aspect_summary.sort_values("total_mentions", ascending=True)
        bars = plt.barh(df_plot["aspect"], df_plot["total_mentions"], color="#2b7bba")
        plt.xlabel("Số lượt review nhắc tới (trên 1.000 mẫu)")
        plt.title("Mức độ quan tâm của khách thuê Airbnb Berlin theo 7 khía cạnh")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        for bar, pct in zip(bars, df_plot["mention_rate_pct"]):
            w = bar.get_width()
            plt.text(w + 8, bar.get_y() + bar.get_height() / 2, f"{int(w)} ({pct}%)", va="center", fontsize=9)
        plt.xlim(0, df_plot["total_mentions"].max() * 1.18)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/5_llm_aspect_frequency.png", dpi=200)
        plt.close()

        # Biểu đồ 6: Phân bố cảm xúc theo từng khía cạnh
        plt.figure(figsize=(9, 5))
        df_stack = df_aspect_summary.sort_values("total_mentions", ascending=True).copy()
        p_pos = df_stack["positive"] / df_stack["total_mentions"].replace(0, 1) * 100
        p_neu = df_stack["neutral"] / df_stack["total_mentions"].replace(0, 1) * 100
        p_neg = df_stack["negative"] / df_stack["total_mentions"].replace(0, 1) * 100

        plt.barh(df_stack["aspect"], p_pos, color="#2ca02c", label="Positive")
        plt.barh(df_stack["aspect"], p_neu, left=p_pos, color="#ffbb78", label="Neutral")
        plt.barh(df_stack["aspect"], p_neg, left=p_pos + p_neu, color="#d62728", label="Negative")
        plt.xlabel("Tỷ lệ phần trăm (%)")
        plt.title("Phân bố cảm xúc theo từng khía cạnh (Berlin Airbnb)")
        plt.legend(loc="lower right")
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/6_llm_aspect_sentiment_breakdown.png", dpi=200)
        plt.close()

    print(f"\n>>> HOÀN TẤT: Toàn bộ biểu đồ đã được lưu tại {out_dir}/!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trực quan hóa toàn bộ dữ liệu không gian, thời gian & LLM")
    parser.add_argument("--config", default="configs/berlin.yaml")
    create_visualizations(parser.parse_args().config)