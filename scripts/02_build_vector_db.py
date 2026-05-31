from pathlib import Path
import math
import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader


# =========================
# 1. 路徑設定
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

PRODUCT_CSV_PATH = DATA_DIR / "products_clean.csv"

COLLECTION_NAME = "fashion_products_v1"


# =========================
# 2. 建庫參數
# =========================
# RESET_COLLECTION = True：
# 每次重跑都會刪掉舊資料庫 collection，重新建立。
# 第一版建議先用 True，避免重複匯入。
#
# BATCH_SIZE：
# 一次送幾張圖片給 ChromaDB / OpenCLIP 處理。
# 電腦普通的話先用 32 或 64，比較穩。

RESET_COLLECTION = True
BATCH_SIZE = 32


# =========================
# 3. 讀取乾淨商品資料
# =========================

def load_products(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 products_clean.csv：{csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "id",
        "product_name",
        "image_path",
        "week",
        "category",
        "category_name",
        "rank",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"products_clean.csv 缺少欄位：{missing_cols}")

    df["id"] = df["id"].astype(str)
    df["product_name"] = df["product_name"].astype(str)
    df["image_path"] = df["image_path"].astype(str)

    return df


# =========================
# 4. 建立 ChromaDB Collection
# =========================

def create_collection():
    """
    建立 ChromaDB 本地資料庫。

    PersistentClient：
    代表資料會真的存在 chroma_db 資料夾裡，
    下次執行搜尋時不用重新建立全部向量。

    OpenCLIPEmbeddingFunction：
    負責把圖片或文字轉成向量。

    ImageLoader：
    負責讓 ChromaDB 可以讀取本地圖片。
    """

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if RESET_COLLECTION:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"已刪除舊 collection：{COLLECTION_NAME}")
        except Exception:
            print(f"沒有找到舊 collection，將建立新的：{COLLECTION_NAME}")

    embedding_function = OpenCLIPEmbeddingFunction()
    image_loader = ImageLoader()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        data_loader=image_loader,
        metadata={
            "description": "Fashion product image collection built with OpenCLIP"
        },
    )

    return collection


# =========================
# 5. 分批匯入商品圖片
# =========================

def build_vector_db(df: pd.DataFrame, collection) -> None:
    total_rows = len(df)
    total_batches = math.ceil(total_rows / BATCH_SIZE)

    print(f"準備匯入商品數量：{total_rows}")
    print(f"Batch size：{BATCH_SIZE}")
    print(f"總批次數：{total_batches}")

    for start_idx in tqdm(range(0, total_rows, BATCH_SIZE), desc="建立向量資料庫"):
        batch_df = df.iloc[start_idx:start_idx + BATCH_SIZE].copy()

        ids = []
        uris = []
        metadatas = []

        for _, row in batch_df.iterrows():
            product_id = str(row["id"])
            product_name = str(row["product_name"])
            relative_image_path = str(row["image_path"])

            absolute_image_path = PROJECT_ROOT / relative_image_path

            if not absolute_image_path.exists():
                raise FileNotFoundError(f"找不到圖片：{absolute_image_path}")

            ids.append(product_id)

            # ChromaDB 的 ImageLoader 需要圖片實際路徑
            uris.append(str(absolute_image_path))

            # metadata 只能放簡單型別：str / int / float / bool
            metadatas.append({
                "product_name": product_name,
                "image_path": relative_image_path,
                "absolute_image_path": str(absolute_image_path),
                "week": int(row["week"]),
                "category": int(row["category"]),
                "category_name": str(row["category_name"]),
                "rank": int(row["rank"]),
            })

        collection.add(
            ids=ids,
            uris=uris,
            metadatas=metadatas,
        )

    print("向量資料庫建立完成。")
    print(f"目前 collection 筆數：{collection.count()}")


# =========================
# 6. 主程式
# =========================

def main():
    print("專案根目錄：", PROJECT_ROOT)
    print("讀取商品資料：", PRODUCT_CSV_PATH)
    print("ChromaDB 儲存位置：", CHROMA_DIR)

    df = load_products(PRODUCT_CSV_PATH)

    print("\n商品資料預覽：")
    print(df.head())

    collection = create_collection()
    build_vector_db(df, collection)

    print("\n完成。")
    print(f"Collection name：{COLLECTION_NAME}")
    print(f"ChromaDB path：{CHROMA_DIR}")


if __name__ == "__main__":
    main()