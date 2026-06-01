from pathlib import Path
import sys
import pandas as pd


# ============================================================
# 12_evaluate_precision_at_k.py
#
# Purpose:
# Export Top-K retrieval results for manual relevance labeling,
# then calculate Precision@K after labels are filled.
#
# Usage:
# cd C:\Users\yujun\Desktop\fashion_rag_project
# .\.venv\Scripts\Activate.ps1
# python scripts\12_evaluate_precision_at_k.py
#
# Workflow:
# 1. First run:
#    Export outputs/eval_top5_to_label.csv
#
# 2. Manually fill:
#    is_relevant = 1 if the product is relevant to the query
#    is_relevant = 0 if the product is not relevant
#
# 3. Run again:
#    Calculate outputs/eval_precision_at_5.csv
# ============================================================


# =========================
# 1. Path setup
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

EXPORT_PATH = OUTPUT_DIR / "eval_top5_to_label.csv"
METRIC_PATH = OUTPUT_DIR / "eval_precision_at_5.csv"

TOP_K = 5


# Make project root importable, so we can reuse search logic from app.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================
# 2. Evaluation queries
# =========================
# You can edit this list later.

EVAL_QUERIES = [
    "白色高領上衣",
    "黑色短裙",
    "海邊度假風洋裝",
    "甜美短裙",
    "灰色大衣",
    "復古洋裝",
    "休閒短褲",
]


# =========================
# 3. Helper functions
# =========================

def normalize_label(value):
    """
    Convert manually filled relevance labels to 0/1.
    Accepts: 1, 0, yes, no, true, false, y, n, 是, 否
    """
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in ["1", "yes", "y", "true", "是", "相關", "relevant"]:
        return 1

    if text in ["0", "no", "n", "false", "否", "不相關", "irrelevant"]:
        return 0

    return None


def has_completed_labels(df: pd.DataFrame) -> bool:
    if "is_relevant" not in df.columns:
        return False

    labels = df["is_relevant"].apply(normalize_label)

    return labels.notna().all()


def export_topk_results():
    """
    Run current search pipeline and export Top-K products for manual labeling.
    """

    print("Loading search pipeline from app.py...")

    # Import current app search logic.
    # app.py should not execute Streamlit main() when imported because it uses if __name__ == "__main__".
    from app import load_products, get_image_collection, search_products

    df = load_products()
    collection = get_image_collection()

    rows = []

    print(f"Running evaluation queries, TOP_K = {TOP_K}")

    for query in EVAL_QUERIES:
        print(f"Query: {query}")

        results, diagnostics = search_products(df, collection, query)

        for rank, item in enumerate(results[:TOP_K], start=1):
            rows.append({
                "query": query,
                "rank": rank,
                "id": item.get("id", ""),
                "product_name": item.get("product_name", ""),
                "category_name": item.get("category_name", ""),
                "subcategory": item.get("subcategory", ""),
                "color": item.get("color", ""),
                "collar": item.get("collar", ""),
                "sleeve": item.get("sleeve", ""),
                "material": item.get("material", ""),
                "style_keywords": item.get("style_keywords", ""),
                "occasion_keywords": item.get("occasion_keywords", ""),
                "product_components": item.get("product_components", ""),
                "final_score": item.get("final_score", ""),
                "title_score": item.get("title_score", ""),
                "image_score": item.get("image_score", ""),
                "image_path": item.get("image_path", ""),
                "strict_candidate_count": diagnostics.get("strict_candidate_count", ""),
                "candidate_count": diagnostics.get("candidate_count", ""),
                "used_relaxed": diagnostics.get("used_relaxed", ""),
                "is_relevant": "",
                "notes": "",
            })

    OUTPUT_DIR.mkdir(exist_ok=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(EXPORT_PATH, index=False, encoding="utf-8-sig")

    print("")
    print("Export completed.")
    print("Please manually label this file:")
    print(EXPORT_PATH)
    print("")
    print("Fill is_relevant with:")
    print("1 = relevant")
    print("0 = not relevant")
    print("")
    print("Then run this script again to calculate Precision@5.")


def calculate_precision_at_k():
    """
    Calculate Precision@K after manual labeling.
    """

    df = pd.read_csv(EXPORT_PATH)

    df["label"] = df["is_relevant"].apply(normalize_label)

    if df["label"].isna().any():
        missing = df[df["label"].isna()][["query", "rank", "id", "product_name", "is_relevant"]]
        print("Some rows are not labeled correctly. Please fill is_relevant with 1 or 0.")
        print("")
        print(missing.to_string(index=False))
        return

    metric_rows = []

    for query, group in df.groupby("query", sort=False):
        group = group.sort_values("rank").head(TOP_K)

        relevant_count = int(group["label"].sum())
        retrieved_count = len(group)
        precision_at_k = relevant_count / TOP_K

        metric_rows.append({
            "query": query,
            "top_k": TOP_K,
            "retrieved_count": retrieved_count,
            "relevant_count": relevant_count,
            "precision_at_k": precision_at_k,
            "strict_candidate_count": group["strict_candidate_count"].iloc[0] if "strict_candidate_count" in group.columns else "",
            "candidate_count": group["candidate_count"].iloc[0] if "candidate_count" in group.columns else "",
            "used_relaxed": group["used_relaxed"].iloc[0] if "used_relaxed" in group.columns else "",
        })

    metric_df = pd.DataFrame(metric_rows)

    average_precision = metric_df["precision_at_k"].mean()

    summary_row = {
        "query": "AVERAGE",
        "top_k": TOP_K,
        "retrieved_count": "",
        "relevant_count": "",
        "precision_at_k": average_precision,
        "strict_candidate_count": "",
        "candidate_count": "",
        "used_relaxed": "",
    }

    metric_df = pd.concat([metric_df, pd.DataFrame([summary_row])], ignore_index=True)

    metric_df.to_csv(METRIC_PATH, index=False, encoding="utf-8-sig")

    print("")
    print("Precision@K calculation completed.")
    print("Metric file:")
    print(METRIC_PATH)
    print("")
    print(metric_df.to_string(index=False))


# =========================
# 4. Main
# =========================

def main():
    print("Evaluation: Precision@5")
    print("=======================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Export path: {EXPORT_PATH}")
    print(f"Metric path: {METRIC_PATH}")
    print("")

    if not EXPORT_PATH.exists():
        export_topk_results()
        return

    df = pd.read_csv(EXPORT_PATH)

    if not has_completed_labels(df):
        print("Existing export file found, but labels are incomplete.")
        print("Please open and label:")
        print(EXPORT_PATH)
        print("")
        print("Fill is_relevant with 1 or 0, then run this script again.")
        return

    calculate_precision_at_k()


if __name__ == "__main__":
    main()
