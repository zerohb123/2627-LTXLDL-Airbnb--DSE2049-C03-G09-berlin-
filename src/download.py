import os            # Thao tác với hệ điều hành: tạo thư mục, kiểm tra file tồn tại
import gzip          # Hỗ trợ giải nén file định dạng .gz
import shutil        # Tiện ích sao chép / di chuyển file hệ thống
import requests      # Gửi yêu cầu HTTP để tải file từ Internet
import yaml          # Đọc dữ liệu từ file định dạng .yaml (PyYAML)
import pandas as pd  # Thư viện xử lý và phân tích dữ liệu bảng
from tqdm import tqdm# Hiển thị thanh tiến trình (progress bar) khi tải file

def download_file(url, dest_path):
    """Tải file từ URL về dest_path với thanh tiến trình."""
    if os.path.exists(dest_path):
        print(f"  [Đã có] {os.path.basename(dest_path)}")
        return
    print(f"  [Đang tải] {url}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Không thể tải từ {url} (Mã lỗi: {response.status_code})")
    
    total_size = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024*1024):
            size = file.write(data)
            bar.update(size)

def download_and_filter(config_path="configs/berlin.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_url = config["base_url"].rstrip("/")
    snapshots = config["snapshots"]
    
    # 1. Đọc dữ liệu lọc của nhóm
    print(">>> 1. Đang đọc danh sách ID cần lọc...")
    df_filter_listing = pd.read_csv(config["filter_files"]["listings"], sep=None, engine='python')
    # Chuẩn hóa tên cột
    listing_id_col = 'listing_id' if 'listing_id' in df_filter_listing.columns else 'id'
    target_listing_ids = set(df_filter_listing[listing_id_col].astype(int))
    
    df_filter_review = pd.read_csv(config["filter_files"]["reviews"], sep=None, engine='python')
    review_id_col = 'review_id' if 'review_id' in df_filter_review.columns else 'id'
    target_review_ids = set(df_filter_review[review_id_col].astype(int))
    
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)

    # 2. Xử lý từng mốc snapshot
    for snap in snapshots:
        print(f"\n==========================================")
        print(f"Xử lý mốc snapshot: {snap}")
        print(f"==========================================")
        
        # Đường dẫn download Inside Airbnb
        url_listings = f"{base_url}/{snap}/data/listings.csv.gz"
        url_calendar = f"{base_url}/{snap}/data/calendar.csv.gz"
        url_reviews = f"{base_url}/{snap}/data/reviews.csv.gz"

        raw_listings_path = f"data/raw/{snap}_listings.csv.gz"
        raw_calendar_path = f"data/raw/{snap}_calendar.csv.gz"
        raw_reviews_path = f"data/raw/{snap}_reviews.csv.gz"

        # Tải file thô về data/raw/ (được gitignore bảo vệ)
        download_file(url_listings, raw_listings_path)
        download_file(url_calendar, raw_calendar_path)
        download_file(url_reviews, raw_reviews_path)

        # Lọc Listings
        print(f"  -> Đang lọc listings ({snap})...")
        df_raw_l = pd.read_csv(raw_listings_path, compression='gzip', low_memory=False)
        df_filtered_l = df_raw_l[df_raw_l['id'].isin(target_listing_ids)].copy()
        
        # Gắn cột analysis_area từ file cấu hình của nhóm
        area_map = dict(zip(df_filter_listing[listing_id_col].astype(int), df_filter_listing['analysis_area']))
        df_filtered_l['analysis_area'] = df_filtered_l['id'].map(area_map)
        
        processed_l_path = f"data/processed/{snap}_listings.csv"
        df_filtered_l.to_csv(processed_l_path, index=False)
        print(f"     Lưu: {processed_l_path} ({len(df_filtered_l)} dòng)")

        # Lọc Calendar
        print(f"  -> Đang lọc calendar ({snap})...")
        chunks = []
        for chunk in pd.read_csv(raw_calendar_path, compression='gzip', chunksize=100000, low_memory=False):
            matched = chunk[chunk['listing_id'].isin(target_listing_ids)]
            if not matched.empty:
                chunks.append(matched)
        if chunks:
            df_filtered_c = pd.concat(chunks, ignore_index=True)
            processed_c_path = f"data/processed/{snap}_calendar.csv"
            df_filtered_c.to_csv(processed_c_path, index=False)
            print(f"     Lưu: {processed_c_path} ({len(df_filtered_c)} dòng)")

        # Lọc Reviews
        print(f"  -> Đang lọc reviews ({snap})...")
        df_raw_r = pd.read_csv(raw_reviews_path, compression='gzip', low_memory=False)
        df_filtered_r = df_raw_r[df_raw_r['id'].isin(target_review_ids)].copy()
        processed_r_path = f"data/processed/{snap}_reviews.csv"
        df_filtered_r.to_csv(processed_r_path, index=False)
        print(f"     Lưu: {processed_r_path} ({len(df_filtered_r)} dòng)")

    # 4. Tải tệp bản đồ GeoJSON (lấy theo mốc snapshot mới nhất)
    latest_snap = snapshots[-1]
    url_geojson = f"{base_url}/{latest_snap}/visualisations/neighbourhoods.geojson"
    raw_geojson_path = "data/raw/neighbourhoods.geojson"
    try:
        download_file(url_geojson, raw_geojson_path)
    except Exception as e:
        print(f"  [Cảnh báo] Không tải được GeoJSON từ {url_geojson}: {e}")

    # 5. Tách bộ dữ liệu mẫu LLM và nhãn tay (100 mẫu)
    print("\n>>> 3. Đang tạo bộ dữ liệu mẫu gán nhãn tay...")
    if 'Nhãn tay' in df_filter_review.columns:
        labeled_ids = set(df_filter_review[df_filter_review['Nhãn tay'] == 'x'][review_id_col].astype(int))
        # Lấy từ mốc snapshot đầu tiên
        first_rev_csv = f"data/processed/{snapshots[0]}_reviews.csv"
        df_first_rev = pd.read_csv(first_rev_csv)
        df_100 = df_first_rev[df_first_rev['id'].isin(labeled_ids)].copy()
        
        # Thêm các cột nhãn trống để nhóm tự gán nhãn
        df_100['sentiment_true'] = ""
        df_100['aspects_true'] = ""
        sample_path = "data/samples/labeled_100.csv"
        df_100.to_csv(sample_path, index=False)
        print(f"     Đã xuất file nhãn tay: {sample_path} ({len(df_100)} dòng)")

    print("\n>>> HOÀN THÀNH TẢI VÀ LỌC DỮ LIỆU THÀNH CÔNG!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/berlin.yaml")
    download_and_filter(parser.parse_args().config)