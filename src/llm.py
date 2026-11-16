import os
import json
import time
import argparse
import yaml
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel
from typing import Literal
from google import genai
from google.genai import types
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

class ReviewItemResult(BaseModel):
    id: str
    sentiment: Literal["positive", "neutral", "negative"]
    aspects: list[str]

class BatchReviewAnalysis(BaseModel):
    results: list[ReviewItemResult]

def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def analyze_batch(client, model_name, batch_items, allowed_aspects, max_retries=5):
    """Phân tích đồng thời một nhóm (batch) review để tiết kiệm RPM và tránh lỗi 429."""
    payload = [{"id": rid, "comment": text[:1200]} for rid, text in batch_items]
    prompt = f"""
    Phân tích danh sách các đánh giá khách thuê Airbnb sau đây (có thể viết bằng tiếng Anh, tiếng Đức hoặc ngôn ngữ khác).
    Với MỖI phần tử trong danh sách đầu vào, trả về đúng "id" tương ứng cùng:
    1. "sentiment":
       - "positive": hài lòng, khen ngợi (kể cả có góp ý nhỏ).
       - "neutral": trung tính, mô tả khách quan, khen chê ngang nhau, hoặc thông báo tự động của hệ thống.
       - "negative": phàn nàn, thất vọng, gặp sự cố nghiêm trọng.
    2. "aspects": danh sách các khía cạnh được nhắc tới, CHỈ chọn từ tập đúng các nhãn sau:
       {json.dumps(allowed_aspects, ensure_ascii=False)}
       Nếu không nhắc tới khía cạnh cụ thể nào, trả về [].

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
                    "aspects": valid_aspects
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

def run_llm_pipeline(config_path="configs/berlin.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Thiếu biến môi trường GEMINI_API_KEY.")

    llm_cfg = config.get("llm", {})
    model_name = llm_cfg.get("model", "gemini-3.5-flash-lite")
    allowed_aspects = llm_cfg.get("aspects", ["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn"])
    cache_path = llm_cfg.get("cache", "data/processed/llm_cache.json")
    sample_size = llm_cfg.get("sample_size", 1000)

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60000))
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

    # Lọc các review chưa có trong cache để gọi API theo batch 30 câu
    to_process = []
    for _, row in df_sample_1000.iterrows():
        rid = str(row["id"])
        comment = str(row.get("comments", "")).strip()
        if rid not in cache and comment and comment.lower() != "nan":
            to_process.append((rid, comment))

    print(f">>> Đã có sẵn {len(cache)} mẫu trong cache. Cần gọi API cho {len(to_process)} mẫu còn lại (Model: {model_name})...")

    batch_size = 30
    batches = [to_process[i:i + batch_size] for i in range(0, len(to_process), batch_size)]

    try:
        for batch in tqdm(batches, desc="Đang xử lý từng batch (30 reviews/lượt)"):
            try:
                batch_res = analyze_batch(client, model_name, batch, allowed_aspects)
                for rid, _ in batch:
                    if rid in batch_res:
                        cache[rid] = batch_res[rid]
                save_cache(cache, cache_path)
                time.sleep(4.2)  # Giữ nhịp ~14 requests/phút, an toàn dưới ngưỡng 15 RPM
            except RuntimeError:
                raise
            except Exception as e:
                print(f"\n  [Lỗi Batch]: {e}")
    finally:
        save_cache(cache, cache_path)

    # Tổng hợp kết quả cho toàn bộ 1.000 mẫu
    sentiments, aspects_list = [], []
    for _, row in df_sample_1000.iterrows():
        rid = str(row["id"])
        res = cache.get(rid, {"sentiment": "neutral", "aspects": []})
        sentiments.append(res.get("sentiment", "neutral").lower())
        aspects_list.append(", ".join(res.get("aspects", [])))

    df_sample_1000["sentiment_pred"] = sentiments
    df_sample_1000["aspects_pred"] = aspects_list

    os.makedirs("reports", exist_ok=True)
    out_results = "reports/llm_results.csv"
    df_sample_1000[["id", "listing_id", "date", "sentiment_pred", "aspects_pred"]].to_csv(
        out_results, index=False, encoding="utf-8"
    )
    print(f"\n>>> Đã lưu kết quả 1.000 mẫu tại: {out_results}")

    # 2. Đánh giá hiệu năng trên 100 mẫu gán nhãn tay
    labeled_path = "data/samples/labeled_100.csv"
    if os.path.exists(labeled_path):
        df_100 = pd.read_csv(labeled_path)
        if "sentiment_true" in df_100.columns and df_100["sentiment_true"].notna().any():
            df_eval = df_100.merge(
                df_sample_1000[["id", "sentiment_pred", "aspects_pred"]], on="id", how="inner"
            ).dropna(subset=["sentiment_true"])

            y_true_sent = df_eval["sentiment_true"].astype(str).str.strip().str.lower().replace({"positve": "positive"})
            y_pred_sent = df_eval["sentiment_pred"].astype(str).str.strip().str.lower()

            acc_sent = accuracy_score(y_true_sent, y_pred_sent)
            f1_macro_sent = f1_score(y_true_sent, y_pred_sent, average="macro", zero_division=0)

            report_dict = classification_report(y_true_sent, y_pred_sent, output_dict=True, zero_division=0)
            df_metrics = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "metric_target"})
            df_metrics["task"] = "sentiment"

            if "aspects_true" in df_eval.columns:
                parse_asp = lambda s: [x.strip() for x in str(s).split(",") if x.strip() in allowed_aspects] if pd.notna(s) else []
                y_true_asp = [parse_asp(x) for x in df_eval["aspects_true"]]
                y_pred_asp = [parse_asp(x) for x in df_eval["aspects_pred"]]

                mlb = MultiLabelBinarizer(classes=allowed_aspects)
                Y_t = mlb.fit_transform(y_true_asp)
                Y_p = mlb.transform(y_pred_asp)

                asp_report = classification_report(Y_t, Y_p, target_names=allowed_aspects, output_dict=True, zero_division=0)
                df_asp_metrics = pd.DataFrame(asp_report).transpose().reset_index().rename(columns={"index": "metric_target"})
                df_asp_metrics["task"] = "aspects"
                df_metrics = pd.concat([df_metrics, df_asp_metrics], ignore_index=True)
                f1_macro_asp = f1_score(Y_t, Y_p, average="macro", zero_division=0)
            else:
                f1_macro_asp = 0.0

            df_metrics.to_csv("reports/llm_eval.csv", index=False, encoding="utf-8")

            print(f"\n>>> KẾT QUẢ ĐÁNH GIÁ TRÊN {len(df_eval)} MẪU NHÃN TAY:")
            print(f"    - Sentiment Accuracy : {acc_sent:.4f}")
            print(f"    - Sentiment Macro-F1 : {f1_macro_sent:.4f}")
            print(f"    - Aspects Macro-F1   : {f1_macro_asp:.4f}")
            print(">>> Đã lưu báo cáo đánh giá chi tiết tại: reports/llm_eval.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phân tích cảm xúc & khía cạnh bằng LLM")
    parser.add_argument("--config", default="configs/berlin.yaml")
    run_llm_pipeline(parser.parse_args().config)