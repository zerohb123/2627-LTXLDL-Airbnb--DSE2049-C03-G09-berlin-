import os
import argparse
import yaml
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

def evaluate_models(config_path="configs/berlin.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    allowed_aspects = config.get("llm", {}).get(
        "aspects", ["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn"]
    )
    labeled_path = "data/samples/labeled_100.csv"
    llm_path = "data/processed/llm_results.csv"
    base_path = "data/processed/baseline_results.csv"

    if not os.path.exists(labeled_path):
        print(f"[Cảnh báo] Không tìm thấy tệp nhãn tay {labeled_path}.")
        return

    df_100 = pd.read_csv(labeled_path)
    if "sentiment_true" not in df_100.columns or not df_100["sentiment_true"].notna().any():
        print("[Cảnh báo] Tệp labeled_100.csv chưa có dữ liệu cột sentiment_true.")
        return

    df_100["id"] = df_100["id"].astype(str).str.strip()
    df_100 = df_100.dropna(subset=["sentiment_true"]).copy()
    y_true_sent = df_100["sentiment_true"].astype(str).str.strip().str.lower().replace({"positve": "positive"})

    parse_asp = lambda s: [x.strip() for x in str(s).split(",") if x.strip() in allowed_aspects] if pd.notna(s) else []
    y_true_asp = [parse_asp(x) for x in df_100.get("aspects_true", [""] * len(df_100))]
    mlb = MultiLabelBinarizer(classes=allowed_aspects)
    Y_t = mlb.fit_transform(y_true_asp)

    all_reports = []
    summary_rows = []

    models_to_eval = [
        ("LLM (Gemini)", llm_path, "sentiment_pred", "aspects_pred"),
        ("Baseline (Lexicon)", base_path, "sentiment_baseline", "aspects_baseline"),
    ]

    for model_name, res_path, sent_col, asp_col in models_to_eval:
        if not os.path.exists(res_path):
            print(f"[Bỏ qua] Không tìm thấy kết quả của {model_name} tại {res_path}")
            continue

        df_pred = pd.read_csv(res_path)
        df_pred["id"] = df_pred["id"].astype(str).str.strip()
        df_m = df_100[["id"]].merge(df_pred[["id", sent_col, asp_col]], on="id", how="left")

        y_p_sent = df_m[sent_col].fillna("neutral").astype(str).str.strip().str.lower()
        acc = accuracy_score(y_true_sent, y_p_sent)
        f1_sent = f1_score(y_true_sent, y_p_sent, average="macro", zero_division=0)

        rep_sent = classification_report(y_true_sent, y_p_sent, output_dict=True, zero_division=0)
        df_rep_s = pd.DataFrame(rep_sent).transpose().reset_index().rename(columns={"index": "metric_target"})
        df_rep_s["model"] = model_name
        df_rep_s["task"] = "sentiment"
        all_reports.append(df_rep_s)

        y_p_asp = [parse_asp(x) for x in df_m[asp_col]]
        Y_p = mlb.transform(y_p_asp)
        f1_asp = f1_score(Y_t, Y_p, average="macro", zero_division=0)

        rep_asp = classification_report(Y_t, Y_p, target_names=allowed_aspects, output_dict=True, zero_division=0)
        df_rep_a = pd.DataFrame(rep_asp).transpose().reset_index().rename(columns={"index": "metric_target"})
        df_rep_a["model"] = model_name
        df_rep_a["task"] = "aspects"
        all_reports.append(df_rep_a)

        summary_rows.append({
            "model": model_name,
            "samples": len(df_100),
            "sentiment_accuracy": round(acc, 4),
            "sentiment_macro_f1": round(f1_sent, 4),
            "aspects_macro_f1": round(f1_asp, 4),
        })

    os.makedirs("reports", exist_ok=True)
    if all_reports:
        pd.concat(all_reports, ignore_index=True).to_csv("reports/llm_eval.csv", index=False, encoding="utf-8")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("reports/model_comparison.csv", index=False, encoding="utf-8")

    print("\n>>> BẢNG ĐỐI CHỨNG HIỆU NĂNG TRÊN TẬP NHÃN TAY:")
    print(df_summary.to_string(index=False))
    print(">>> Đã lưu báo cáo chi tiết tại: reports/llm_eval.csv và reports/model_comparison.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chấm điểm & đối chứng LLM với Baseline trên nhãn tay")
    parser.add_argument("--config", default="configs/berlin.yaml")
    evaluate_models(parser.parse_args().config)