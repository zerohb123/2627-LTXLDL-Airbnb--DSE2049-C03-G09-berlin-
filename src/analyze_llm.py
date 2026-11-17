import os
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt

def run_llm_analysis(config_path="configs/berlin.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    allowed_aspects = config.get("llm", {}).get(
        "aspects", ["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn"]
    )
    results_path = "reports/llm_results.csv"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Không tìm thấy {results_path}. Hãy chạy src/llm.py trước.")

    df = pd.read_csv(results_path)
    os.makedirs("reports/figures", exist_ok=True)

    # 1. Thống kê phân bố cảm xúc tổng thể
    sent_counts = df["sentiment_pred"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    sent_pct = (sent_counts / len(df) * 100).round(2)
    df_sent_summary = pd.DataFrame({"sentiment": sent_counts.index, "count": sent_counts.values, "percentage": sent_pct.values})
    df_sent_summary.to_csv("reports/llm_sentiment_summary.csv", index=False, encoding="utf-8")

    # 2. Thống kê đa nhãn theo 7 khía cạnh & phân bố cảm xúc từng khía cạnh
    aspect_rows = []
    for asp in allowed_aspects:
        mask = df["aspects_pred"].fillna("").apply(
            lambda s: asp in [x.strip() for x in str(s).split(",")]
        )
        sub = df[mask]
        total = len(sub)
        pos = (sub["sentiment_pred"] == "positive").sum()
        neu = (sub["sentiment_pred"] == "neutral").sum()
        neg = (sub["sentiment_pred"] == "negative").sum()
        aspect_rows.append({
            "aspect": asp,
            "total_mentions": total,
            "mention_rate_pct": round(total / len(df) * 100, 2),
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "positive_rate_pct": round(pos / total * 100, 2) if total > 0 else 0.0,
            "negative_rate_pct": round(neg / total * 100, 2) if total > 0 else 0.0,
        })

    df_aspect_summary = pd.DataFrame(aspect_rows).sort_values("total_mentions", ascending=False)
    df_aspect_summary.to_csv("reports/llm_aspect_summary.csv", index=False, encoding="utf-8")

    # 3. Vẽ Biểu đồ 1: Tần suất được nhắc tới của 7 khía cạnh
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
    plt.savefig("reports/figures/llm_aspect_frequency.png", dpi=200)
    plt.close()

    # 4. Vẽ Biểu đồ 2: Cơ cấu cảm xúc (Positive / Neutral / Negative) theo từng khía cạnh
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
    plt.savefig("reports/figures/llm_aspect_sentiment_breakdown.png", dpi=200)
    plt.close()

    print(">>> Đã xuất bảng tổng hợp và 2 biểu đồ tại:")
    print("    - reports/llm_sentiment_summary.csv")
    print("    - reports/llm_aspect_summary.csv")
    print("    - reports/figures/llm_aspect_frequency.png")
    print("    - reports/figures/llm_aspect_sentiment_breakdown.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tổng hợp & trực quan hóa kết quả phân tích LLM")
    parser.add_argument("--config", default="configs/berlin.yaml")
    run_llm_analysis(parser.parse_args().config)