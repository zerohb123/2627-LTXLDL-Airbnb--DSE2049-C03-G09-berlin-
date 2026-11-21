import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

def format_k(x, pos):
    """Định dạng số ngàn (K) cho trục Y"""
    return f'{int(x/1000)}K' if x >= 1000 else f'{int(x)}'

def create_visualizations():
    kpi_path = "reports/kpi.csv"
    out_dir = "figures"
    geojson_path = "data/raw/neighbourhoods.geojson"

    if not os.path.exists(kpi_path):
        print(f"Lỗi: Không tìm thấy {kpi_path}.")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(kpi_path)
    
    # Danh sách mã vùng A01 - A08 được sắp xếp chuẩn
    sorted_areas = sorted(df['analysis_area'].unique())

    sns.set_theme(style="whitegrid", context="talk")
    custom_palette = sns.color_palette("tab10", len(sorted_areas))

    print(">>> ĐANG XỬ LÝ: Kết xuất các biểu đồ trực quan với mã vùng Axx...")

    # 1. BIỂU ĐỒ CHUỖI THỜI GIAN
    plt.figure(figsize=(12, 7))
    ax1 = sns.lineplot(
        data=df, x='snapshot', y='est_revenue_30', hue='analysis_area', 
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

    # 2. BIỂU ĐỒ ASPECT (GIÁ TRUNG BÌNH THEO VÙNG Ở MỐC MỚI NHẤT)
    latest_snap = df['snapshot'].max()
    df_latest = df[df['snapshot'] == latest_snap].copy()
    
    plt.figure(figsize=(12, 7))
    ax2 = sns.barplot(
        data=df_latest, x='analysis_area', y='mean_price', 
        hue='analysis_area', hue_order=sorted_areas, 
        palette=custom_palette, legend=False
    )
    plt.title(f'Biểu đồ Aspect: Giá trung bình theo Vùng (Mốc {latest_snap})', fontsize=16, fontweight='bold', pad=20)
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
        data=df_sorted, x='est_occupancy_rate_30', y='est_revenue_30', 
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
    
    # Xử lý chú giải cho mã vùng Axx và kích thước bong bóng
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

    # 4. BẢN ĐỒ GEOJSON (Chỉ ánh xạ tên quận tại bước này để khớp bản đồ địa lý)
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
            
            gdf_valid = gdf_merged.dropna(subset=['mean_price'])
            if not gdf_valid.empty:
                gdf_valid.plot(
                    column='mean_price', cmap='YlOrRd', linewidth=1.5, ax=ax, edgecolor='#222222', 
                    legend=True, legend_kwds={'label': "Giá trung bình (€)", 'shrink': 0.75}
                )
                
            plt.title(f'Bản đồ Không gian: Giá trung bình khu vực Berlin ({latest_snap})', fontsize=18, fontweight='bold', pad=20)
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(f"{out_dir}/4_geojson_map.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  -> Đã lưu bản đồ GeoJSON: {out_dir}/4_geojson_map.png")
                
        except Exception as e:
            print(f"  [Lỗi] Không thể vẽ bản đồ GeoJSON: {e}")
    else:
        print(f"  [Cảnh báo] Không tìm thấy tệp GeoJSON tại {geojson_path} (bỏ qua biểu đồ 4).")

    print(f"\n>>> HOÀN TẤT: Toàn bộ biểu đồ đã được lưu tại {out_dir}/!")

if __name__ == "__main__":
    create_visualizations()