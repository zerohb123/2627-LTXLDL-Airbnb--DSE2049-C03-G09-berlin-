import os
import re
import argparse
import yaml
import pandas as pd

POS_KEYWORDS = {
    "great", "amazing", "wonderful", "excellent", "clean", "nice", "perfect",
    "friendly", "helpful", "recommend", "comfortable", "lovely", "best", "super",
    "fantastic", "cozy", "good", "beautiful", "enjoyed", "easy",
    # Tiếng Đức (Berlin)
    "super", "toll", "sehr gut", "sauber", "freundlich", "perfekt", "empfehlen", "schön", "gemütlich"
}

NEG_KEYWORDS = {
    "bad", "dirty", "noisy", "noise", "terrible", "awful", "poor", "rude",
    "broken", "smell", "cold", "uncomfortable", "disappointed", "worst", "problem",
    "issue", "late", "expensive", "small", "dark", "mold",
    # Tiếng Đức
    "schlecht", "schmutzig", "laut", "kalt", "problem", "enttäuscht", "teuer", "kaputt"
}

ASPECT_LEXICON = {
    "vị trí": ["location", "metro", "subway", "u-bahn", "s-bahn", "station", "center", "centre", "walk", "walking", "transport", "bus", "tram", "neighborhood", "neighbourhood", "lage", "zentrum", "bahnhof"],
    "sạch sẽ": ["clean", "cleanliness", "spotless", "dirty", "tidy", "hygiene", "dust", "bathroom", "towel", "sheets", "sauber", "sauberkeit", "schmutzig"],
    "chủ nhà": ["host", "communication", "responsive", "friendly", "helpful", "check-in", "checkin", "welcoming", "kind", "gastgeber", "freundlich", "kommunikation"],
    "giá trị": ["price", "value", "money", "worth", "affordable", "cheap", "expensive", "overpriced", "preis", "wert", "teuer", "günstig"],
    "tiện nghi": ["kitchen", "wifi", "wi-fi", "bed", "shower", "heating", "equipment", "amenities", "coffee", "washer", "tv", "küche", "bett", "dusche", "ausstattung"],
    "độ ồn": ["quiet", "noisy", "noise", "loud", "sound", "street", "sleep", "peaceful", "ruhig", "laut", "lärm", "ruhe"],
    "an toàn": ["safe", "safety", "secure", "security", "lock", "door", "area", "sicher", "sicherheit", "schloss"]
}

def predict_baseline_row(text, allowed_aspects):
    if not isinstance(text, str) or not text.strip():
        return "neutral", []

    text_lower = text.lower()
    tokens = set(re.findall(r"[a-zäöüß\-]+", text_lower))

    pos_hits = sum(1 for kw in POS_KEYWORDS if kw in tokens or (" " in kw and kw in text_lower))
    neg_hits = sum(1 for kw in NEG_KEYWORDS if kw in tokens or (" " in kw and kw in text_lower))

    if neg_hits > pos_hits:
        sentiment = "negative"
    elif pos_hits > neg_hits and pos_hits >= 1:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    matched_aspects = []
    for asp in allowed_aspects:
        kws = ASPECT_LEXICON.get(asp, [])
        if any((kw in tokens or (" " in kw and kw in text_lower)) for kw in kws):
            matched_aspects.append(asp)

    return sentiment, matched_aspects

def run_baseline(config_path="configs/berlin.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    llm_cfg = config.get("llm", {})
    allowed_aspects = llm_cfg.get("aspects", ["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn"])
    sample_size = llm_cfg.get("sample_size", 1000)

    df_filter_r = pd.read_csv(config["filter_files"]["reviews"], sep=None, engine="python")
    rev_id_col = "review_id" if "review_id" in df_filter_r.columns else "id"
    if "Mẫu LLM" in df_filter_r.columns:
        llm_ids = set(df_filter_r[df_filter_r["Mẫu LLM"].astype(str).str.lower() == "x"][rev_id_col].astype(int))
    else:
        llm_ids = set(df_filter_r[rev_id_col].astype(int).head(sample_size))

    first_snap = config["snapshots"][0]
    rev_path = f"data/clean/{first_snap}_reviews.csv" if os.path.exists(f"data/clean/{first_snap}_reviews.csv") else f"data/processed/{first_snap}_reviews.csv"
    df_rev = pd.read_csv(rev_path, low_memory=False)
    df_sample = df_rev[df_rev["id"].isin(llm_ids)].drop_duplicates(subset=["id"]).copy()

    preds_sent, preds_asp = [], []
    for _, row in df_sample.iterrows():
        s, a = predict_baseline_row(str(row.get("comments", "")), allowed_aspects)
        preds_sent.append(s)
        preds_asp.append(", ".join(a))

    df_sample["sentiment_baseline"] = preds_sent
    df_sample["aspects_baseline"] = preds_asp

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/baseline_results.csv"
    df_sample[["id", "listing_id", "date", "sentiment_baseline", "aspects_baseline"]].to_csv(
        out_path, index=False, encoding="utf-8"
    )
    print(f">>> HOÀN TẤT BASELINE: Đã lưu kết quả đối chứng tại {out_path} ({len(df_sample)} mẫu).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy phương pháp đối chứng không dùng LLM (Lexicon Baseline)")
    parser.add_argument("--config", default="configs/berlin.yaml")
    run_baseline(parser.parse_args().config)