from pathlib import Path
import math
import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


# =========================
# 1. 路徑設定
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INPUT_CSV_PATH = DATA_DIR / "products_with_english_caption.csv"
REPORT_PATH = OUTPUT_DIR / "text_vector_db_report.txt"

COLLECTION_NAME = "fashion_product_text_v1"


# =========================
# 2. 建庫參數
# =========================

RESET_COLLECTION = True
BATCH_SIZE = 64


# =========================
# 3. 基本工具
# =========================

def is_valid_value(value) -> bool:
    if pd.isna(value):
        return False

    value = str(value).strip().lower()
    return value not in ["", "nan", "none", "null", "未知"]


def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def safe_int(value, default: int = -1) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_bool(value) -> bool:
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    return value in ["true", "1", "yes", "y"]


# =========================
# 4. 讀取商品英文 caption 資料
# =========================

def load_products(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到 {csv_path}，請先執行 scripts\\06_generate_english_caption.py"
        )

    df = pd.read_csv(csv_path)

    required_cols = [
        "id",
        "product_name",
        "image_path",
        "absolute_image_path",
        "category",
        "category_name",
        "subcategory",
        "color",
        "collar",
        "sleeve",
        "material",
        "style_keywords",
        "occasion_keywords",
        "product_components",
        "search_text",
        "english_caption",
        "english_search_text",
        "caption_quality",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"products_with_english_caption.csv 缺少欄位：{missing_cols}")

    df["id"] = df["id"].astype(str)
    df["english_caption"] = df["english_caption"].fillna("").astype(str)
    df["english_search_text"] = df["english_search_text"].fillna("").astype(str)

    # 這裡用 english_caption 建文字向量。
    # english_caption 比 english_search_text 乾淨，較不容易混入太多次要組件。
    df = df[df["english_caption"].apply(is_valid_value)].copy()
    df = df.reset_index(drop=True)

    return df


# =========================
# 5. 建立 ChromaDB Collection
# =========================

def create_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if RESET_COLLECTION:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"已刪除舊 collection：{COLLECTION_NAME}")
        except Exception:
            print(f"沒有找到舊 collection，將建立新的：{COLLECTION_NAME}")

    embedding_function = OpenCLIPEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={
            "description": "Fashion product English text caption collection built with OpenCLIP"
        },
    )

    return collection


# =========================
# 6. 建立 metadata
# =========================

def build_metadata(row: pd.Series) -> dict:
    """
    ChromaDB metadata 只能放簡單型別：
    str / int / float / bool
    """

    return {
        "product_name": safe_str(row.get("product_name", "")),
        "image_path": safe_str(row.get("image_path", "")),
        "absolute_image_path": safe_str(row.get("absolute_image_path", "")),

        "category": safe_int(row.get("category", -1)),
        "category_name": safe_str(row.get("category_name", "")),
        "subcategory": safe_str(row.get("subcategory", "")),

        "color": safe_str(row.get("color", "")),
        "collar": safe_str(row.get("collar", "")),
        "sleeve": safe_str(row.get("sleeve", "")),
        "material": safe_str(row.get("material", "")),

        "style_keywords": safe_str(row.get("style_keywords", "")),
        "occasion_keywords": safe_str(row.get("occasion_keywords", "")),
        "product_components": safe_str(row.get("product_components", "")),

        "search_text": safe_str(row.get("search_text", "")),
        "english_caption": safe_str(row.get("english_caption", "")),
        "english_search_text": safe_str(row.get("english_search_text", "")),
        "caption_quality": safe_str(row.get("caption_quality", "")),

        "attribute_quality_score": safe_float(row.get("attribute_quality_score", 0.0)),

        "is_office": safe_bool(row.get("is_office", False)),
        "is_minimal": safe_bool(row.get("is_minimal", False)),
        "is_casual": safe_bool(row.get("is_casual", False)),
        "is_vacation": safe_bool(row.get("is_vacation", False)),
        "is_sweet": safe_bool(row.get("is_sweet", False)),
        "is_sexy": safe_bool(row.get("is_sexy", False)),
    }


# =========================
# 7. 分批匯入文字向量
# =========================

def build_text_vector_db(df: pd.DataFrame, collection) -> None:
    total_rows = len(df)
    total_batches = math.ceil(total_rows / BATCH_SIZE)

    print(f"準備匯入商品文字數量：{total_rows}")
    print(f"Batch size：{BATCH_SIZE}")
    print(f"總批次數：{total_batches}")

    for start_idx in tqdm(range(0, total_rows, BATCH_SIZE), desc="建立英文文字向量資料庫"):
        batch_df = df.iloc[start_idx:start_idx + BATCH_SIZE].copy()

        ids = []
        documents = []
        metadatas = []

        for _, row in batch_df.iterrows():
            product_id = safe_str(row["id"])
            english_caption = safe_str(row["english_caption"])

            if not is_valid_value(product_id):
                continue

            if not is_valid_value(english_caption):
                continue

            ids.append(product_id)

            # 注意：
            # 這裡只提供 documents。
            # ChromaDB 新版規則是 documents / images / uris 只能選一種。
            documents.append(english_caption)

            metadatas.append(build_metadata(row))

        if len(ids) == 0:
            continue

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    print("英文文字向量資料庫建立完成。")
    print(f"目前 collection 筆數：{collection.count()}")


# =========================
# 8. 輸出報告
# =========================

def write_report(df: pd.DataFrame, collection_count: int):
    OUTPUT_DIR.mkdir(exist_ok=True)

    lines = []
    lines.append("Text Vector DB Report")
    lines.append("=====================")
    lines.append("")
    lines.append(f"input_csv: {INPUT_CSV_PATH}")
    lines.append(f"collection_name: {COLLECTION_NAME}")
    lines.append(f"chroma_dir: {CHROMA_DIR}")
    lines.append("")
    lines.append(f"input_rows_after_filter: {len(df)}")
    lines.append(f"collection_count: {collection_count}")
    lines.append("")

    lines.append("caption_quality counts:")
    lines.append(str(df["caption_quality"].value_counts(dropna=False)))
    lines.append("")

    lines.append("sample documents:")
    sample_cols = [
        "id",
        "product_name",
        "subcategory",
        "color",
        "collar",
        "sleeve",
        "english_caption",
    ]

    lines.append(df[sample_cols].head(20).to_string(index=False))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 9. 主程式
# =========================

def main():
    print("專案根目錄：", PROJECT_ROOT)
    print("讀取資料：", INPUT_CSV_PATH)
    print("ChromaDB 儲存位置：", CHROMA_DIR)
    print("文字 collection：", COLLECTION_NAME)

    df = load_products(INPUT_CSV_PATH)

    print(f"\n可建文字向量商品數：{len(df)}")

    print("\n資料預覽：")
    preview_cols = [
        "id",
        "product_name",
        "subcategory",
        "color",
        "collar",
        "sleeve",
        "english_caption",
    ]
    print(df[preview_cols].head(10))

    collection = create_collection()
    build_text_vector_db(df, collection)

    write_report(df, collection.count())

    print("\n完成。")
    print(f"Collection name：{COLLECTION_NAME}")
    print(f"Collection count：{collection.count()}")
    print(f"Report：{REPORT_PATH}")


if __name__ == "__main__":
    main()