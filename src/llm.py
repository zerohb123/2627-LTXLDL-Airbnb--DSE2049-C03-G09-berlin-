import os
import json
import time
import argparse
import yaml
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel
from typing import Literal

class ReviewItemResult(BaseModel):
    id: str
    sentiment: Literal["positive", "neutral", "negative"]
    aspects: list[str]
    language: str

class BatchReviewAnalysis(BaseModel):
    results: list[ReviewItemResult]

def load_cache(jsonl_path, legacy_json_path="data/processed/llm_cache.json"):
    """Đọc cache từ định dạng JSONL chuẩn (tự động kế thừa nếu có file .json cũ)."""
    cache = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    rid = str(item.get("id"))
                    cache[rid] = {
                        "sentiment": item.get("sentiment", "neutral"),
                        "aspects": item.get("aspects", []),
                        "language": item.get("language", "unknown")
                    }
    elif os.path.exists(legacy_json_path):
        with open(legacy_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            for rid, val in raw.items():
                cache[str(rid)] = {
                    "sentiment": val.get("sentiment", "neutral"),
                    "aspects": val.get("aspects", []),
                    "language": val.get("language", "en")
                }
        save_cache(cache, jsonl_path)
    return cache

def save_cache(cache, jsonl_path):
    """Lưu toàn bộ cache ra định dạng JSONL (mỗi dòng 1 bản ghi)."""
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rid, val in cache.items():
            record = {
                "id": str(rid),
                "sentiment": val.get("sentiment", "neutral"),
                "aspects": val.get("aspects", []),
                "language": val.get("language", "unknown")
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def analyze_batch(client, types, model_name, batch_items, allowed_aspects, max_retries=5):
    """Phân tích đồng thời một nhóm (batch) review."""
    payload = [{"id": rid, "comment": text[:1200]} for rid, text in batch_items]
    prompt = f"""
    Phân tích danh sách các đánh giá khách thuê Airbnb sau đây.
    Với MỖI phần tử trong danh sách đầu vào, trả về đúng "id" tương ứng cùng:
    1. "sentiment":
       - "positive": hài lòng, khen ngợi (kể cả có góp ý nhỏ).
       - "neutral": trung tính, mô tả khách quan, khen chê ngang nhau, hoặc thông báo tự động.
       - "negative": phàn nàn, thất vọng, gặp sự cố nghiêm trọng.
    2. "aspects": danh sách các khía cạnh được nhắc tới, CHỈ chọn từ tập đúng các nhãn sau:
       {json.dumps(allowed_aspects, ensure_ascii=False)}
       Nếu không nhắc tới khía cạnh cụ thể nào, trả về [].
    3. "language": mã ngôn ngữ chính của bình luận (ví dụ: "en", "de", "fr", "es", "other").

    Dữ liệu đầu vào:
    {json.dumps(payload, ensure_ascii=False)}
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchReviewAnalysis,
                    temperature=0.0,
                ),
            )
            parsed = json.loads(response.text)
            out_map = {}
            for item in parsed.get("results", []):
                rid = str(item.get("id"))
                valid_aspects = [a for a in item.get("aspects", []) if a in allowed_aspects]
                out_map[rid] = {
                    "sentiment": str(item.get("sentiment", "neutral")).lower(),
                    "aspects": valid_aspects,
                    "language": str(item.get("language", "unknown")).lower()
                }
            return out_map
        except Exception as e:
            err_msg = str(e)
            if "API_KEY_INVALID" in err_msg or "400" in err_msg or "404" in err_msg:
                raise RuntimeError(f"Lỗi cấu hình API/Model, dừng chương trình: {e}")
            if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "503" in err_msg) and attempt < max_retries - 1:
                print("\n  [Rate Limit 429] Đang tạm nghỉ 60 giây để hồi hạn mức Free Tier...")
                time.sleep(60)
                continue
            raise e

def run_llm_pipeline(config_path="configs/berlin.yaml", skip_llm=False):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    llm_cfg = config.get("llm", {})
    model_name = llm_cfg.get("model", "gemini-3.5-flash-lite")
    allowed_aspects = llm_cfg.get("aspects", ["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn"])
    cache_path = llm_cfg.get("cache", "data/processed/llm_cache.jsonl")
    sample_size = llm_cfg.get("sample_size", 1000)

    cache = load_cache(cache_path)

    # 1. Đọc danh sách 1.000 review mẫu LLM
    df_filter_r = pd.read_csv(config["filter_files"]["reviews"], sep=None, engine="python")
    rev_id_col = "review_id" if "review_id" in df_filter_r.columns else "id"

    if "Mẫu LLM" in df_filter_r.columns:
        llm_ids = set(df_filter_r[df_filter_r["Mẫu LLM"].astype(str).str.lower() == "x"][rev_id_col].astype(int))
    else:
        llm_ids = set(df_filter_r[rev_id_col].astype(int).head(sample_size))

    first_snap = config["snapshots"][0]
    rev_path = f"data/clean/{first_snap}_reviews.csv" if os.path.exists(f"data/clean/{first_snap}_reviews.csv") else f"data/processed/{first_snap}_reviews.csv"
    df_rev = pd.read_csv(rev_path, low_memory=False)
    df_sample_1000 = df_rev[df_rev["id"].isin(llm_ids)].drop_duplicates(subset=["id"]).copy()

    # Lọc các review chưa có trong cache
    to_process = []
    for _, row in df_sample_1000.iterrows():
        rid = str(row["id"])
        comment = str(row.get("comments", "")).strip()
        if rid not in cache and comment and comment.lower() != "nan":
            to_process.append((rid, comment))

    print(f">>> Cache hiện có {len(cache)} mẫu. Số mẫu chưa có trong cache: {len(to_process)}.")

    # Chỉ khởi tạo và gọi Gemini API khi thực sự còn mẫu thiếu VÀ không bật --skip-llm
    if len(to_process) > 0 and not skip_llm:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("  [Cảnh báo] Không tìm thấy GEMINI_API_KEY. Chuyển sang chế độ đọc hoàn toàn từ cache.")
        else:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60000))
            batch_size = 30
            batches = [to_process[i:i + batch_size] for i in range(0, len(to_process), batch_size)]

            try:
                for batch in tqdm(batches, desc="Đang gọi Gemini API (30 reviews/batch)"):
                    try:
                        batch_res = analyze_batch(client, types, model_name, batch, allowed_aspects)
                        for rid, _ in batch:
                            if rid in batch_res:
                                cache[rid] = batch_res[rid]
                        save_cache(cache, cache_path)
                        time.sleep(4.2)
                    except RuntimeError:
                        raise
                    except Exception as e:
                        print(f"\n  [Lỗi Batch]: {e}")
            finally:
                save_cache(cache, cache_path)
    else:
        print(">>> Sử dụng 100% kết quả đã lưu trong cache (không gọi API).")

    # Tổng hợp kết quả cho toàn bộ 1.000 mẫu
    sentiments, aspects_list, languages = [], [], []
    for _, row in df_sample_1000.iterrows():
        rid = str(row["id"])
        res = cache.get(rid, {"sentiment": "neutral", "aspects": [], "language": "unknown"})
        sentiments.append(res.get("sentiment", "neutral").lower())
        aspects_list.append(", ".join(res.get("aspects", [])))
        languages.append(res.get("language", "unknown"))

    df_sample_1000["sentiment_pred"] = sentiments
    df_sample_1000["aspects_pred"] = aspects_list
    df_sample_1000["language_pred"] = languages

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    out_cols = ["id", "listing_id", "date", "sentiment_pred", "aspects_pred", "language_pred"]
    df_sample_1000[out_cols].to_csv("data/processed/llm_results.csv", index=False, encoding="utf-8")
    df_sample_1000[out_cols].to_csv("reports/llm_results.csv", index=False, encoding="utf-8")
    print(">>> Đã xuất kết quả LLM tại: data/processed/llm_results.csv và reports/llm_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất cảm xúc & khía cạnh bằng LLM (hỗ trợ cache)")
    parser.add_argument("--config", default="configs/berlin.yaml")
    parser.add_argument("--skip-llm", action="store_true", help="Chỉ đọc cache, không gọi API")
    args = parser.parse_args()
    run_llm_pipeline(args.config, skip_llm=args.skip_llm)