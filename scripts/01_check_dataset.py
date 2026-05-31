from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


# =========================
# 1. 專案路徑設定
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = PROJECT_ROOT / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_EXCEL_PATH = DATA_DIR / "raw_products.xlsx"
CLEAN_CSV_PATH = DATA_DIR / "products_clean.csv"

BAD_ROWS_PATH = OUTPUT_DIR / "bad_rows.csv"
REPORT_PATH = OUTPUT_DIR / "dataset_check_report.txt"


# =========================
# 2. 類別對照表
# =========================
# 你的檔名是像 1_上衣_001.jpg
# 但 Excel 裡的「類別」可能是 1, 2, 3...
# 這裡先保留一份對照，之後可以當 metadata 使用。

CATEGORY_MAP = {
    1: "上衣",
    2: "套裝",
    3: "褲子",
    4: "裙子",
    5: "外套",
    6: "內著",
}


# =========================
# 3. 讀取原始 Excel
# =========================

def load_raw_excel(excel_path: Path) -> pd.DataFrame:
    """
    讀取 raw_products.xlsx。

    你的 Excel 第一列看起來不是 pandas 可以直接辨識的標準欄位，
    所以這裡用 header=None 讀進來，再自動找出包含「檔名」「標題」的那一列。
    """

    if not excel_path.exists():
        raise FileNotFoundError(f"找不到 Excel 檔案：{excel_path}")

    xls = pd.ExcelFile(excel_path)

    if "合併檔案_純值" in xls.sheet_names:
        sheet_name = "合併檔案_純值"
    else:
        sheet_name = xls.sheet_names[0]

    raw_df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    header_row_index = None

    for i in range(min(10, len(raw_df))):
        row_values = raw_df.iloc[i].astype(str).tolist()
        if "檔名" in row_values and "標題" in row_values:
            header_row_index = i
            break

    if header_row_index is None:
        raise ValueError("找不到包含「檔名」與「標題」的欄位列，請檢查 Excel 格式。")

    columns = raw_df.iloc[header_row_index].astype(str).tolist()
    df = raw_df.iloc[header_row_index + 1:].copy()
    df.columns = columns

    df = df.reset_index(drop=True)

    return df


# =========================
# 4. 整理成 RAG 系統需要的格式
# =========================

def clean_product_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    將原始欄位：
    檔名、週次、類別、排名、標題

    轉成第一版 RAG 需要的欄位：
    id、product_name、image_path、week、category、category_name、rank
    """

    required_cols = ["檔名", "週次", "類別", "排名", "標題"]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Excel 缺少必要欄位：{missing_cols}")

    clean_df = pd.DataFrame()

    clean_df["id"] = df["檔名"].astype(str).str.strip()
    clean_df["product_name"] = df["標題"].astype(str).str.strip()

    clean_df["week"] = pd.to_numeric(df["週次"], errors="coerce")
    clean_df["category"] = pd.to_numeric(df["類別"], errors="coerce")
    clean_df["rank"] = pd.to_numeric(df["排名"], errors="coerce")

    clean_df["category_name"] = clean_df["category"].map(CATEGORY_MAP)

    # 你的圖片都是 jpg，且檔名對應 id
    clean_df["image_path"] = clean_df["id"].apply(lambda x: f"images/{x}.jpg")

    # 移除明顯空白列
    clean_df = clean_df[
        (clean_df["id"].notna()) &
        (clean_df["id"] != "") &
        (clean_df["id"].str.lower() != "nan")
    ].copy()

    clean_df = clean_df.reset_index(drop=True)

    return clean_df


# =========================
# 5. 檢查圖片是否存在與可讀取
# =========================

def check_image_file(relative_image_path: str) -> tuple[bool, bool, str]:
    """
    回傳：
    image_exists: 圖片檔案是否存在
    image_readable: 圖片是否能被 PIL 正常開啟
    error_message: 錯誤原因
    """

    image_path = PROJECT_ROOT / relative_image_path

    if not image_path.exists():
        return False, False, "圖片檔案不存在"

    try:
        with Image.open(image_path) as img:
            img.verify()
        return True, True, ""
    except Exception as e:
        return True, False, str(e)


def validate_dataset(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    檢查資料品質：
    1. id 是否重複
    2. 商品名稱是否缺失
    3. 圖片是否存在
    4. 圖片是否可以開啟
    """

    checked_df = clean_df.copy()

    checked_df["id_duplicated"] = checked_df["id"].duplicated(keep=False)

    checked_df["name_missing"] = (
        checked_df["product_name"].isna() |
        (checked_df["product_name"].astype(str).str.strip() == "") |
        (checked_df["product_name"].astype(str).str.lower() == "nan")
    )

    image_exists_list = []
    image_readable_list = []
    image_error_list = []

    for image_path in tqdm(checked_df["image_path"], desc="檢查圖片"):
        image_exists, image_readable, error_message = check_image_file(image_path)

        image_exists_list.append(image_exists)
        image_readable_list.append(image_readable)
        image_error_list.append(error_message)

    checked_df["image_exists"] = image_exists_list
    checked_df["image_readable"] = image_readable_list
    checked_df["image_error"] = image_error_list

    checked_df["is_valid"] = (
        (~checked_df["id_duplicated"]) &
        (~checked_df["name_missing"]) &
        (checked_df["image_exists"]) &
        (checked_df["image_readable"])
    )

    bad_rows = checked_df[~checked_df["is_valid"]].copy()

    report = {
        "total_rows": len(checked_df),
        "valid_rows": int(checked_df["is_valid"].sum()),
        "bad_rows": int((~checked_df["is_valid"]).sum()),
        "duplicated_id_rows": int(checked_df["id_duplicated"].sum()),
        "missing_name_rows": int(checked_df["name_missing"].sum()),
        "missing_image_rows": int((~checked_df["image_exists"]).sum()),
        "unreadable_image_rows": int(
            (checked_df["image_exists"] & ~checked_df["image_readable"]).sum()
        ),
    }

    return checked_df, bad_rows, report


# =========================
# 6. 輸出檢查結果
# =========================

def save_outputs(checked_df: pd.DataFrame, bad_rows: pd.DataFrame, report: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    valid_df = checked_df[checked_df["is_valid"]].copy()

    export_cols = [
        "id",
        "product_name",
        "image_path",
        "week",
        "category",
        "category_name",
        "rank",
    ]

    valid_df[export_cols].to_csv(CLEAN_CSV_PATH, index=False, encoding="utf-8-sig")

    if len(bad_rows) > 0:
        bad_rows.to_csv(BAD_ROWS_PATH, index=False, encoding="utf-8-sig")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Dataset Check Report\n")
        f.write("====================\n\n")

        for key, value in report.items():
            f.write(f"{key}: {value}\n")

        f.write("\nOutput Files\n")
        f.write("============\n")
        f.write(f"clean csv: {CLEAN_CSV_PATH}\n")
        f.write(f"bad rows: {BAD_ROWS_PATH}\n")
        f.write(f"report: {REPORT_PATH}\n")


# =========================
# 7. 主程式
# =========================

def main():
    print("專案根目錄：", PROJECT_ROOT)
    print("讀取 Excel：", RAW_EXCEL_PATH)

    raw_df = load_raw_excel(RAW_EXCEL_PATH)
    print(f"原始資料筆數：{len(raw_df)}")

    clean_df = clean_product_table(raw_df)
    print(f"整理後資料筆數：{len(clean_df)}")

    checked_df, bad_rows, report = validate_dataset(clean_df)

    save_outputs(checked_df, bad_rows, report)

    print("\n檢查完成")
    print("==========")
    print(f"總筆數：{report['total_rows']}")
    print(f"有效筆數：{report['valid_rows']}")
    print(f"問題筆數：{report['bad_rows']}")
    print(f"重複 id 筆數：{report['duplicated_id_rows']}")
    print(f"商品名稱缺失筆數：{report['missing_name_rows']}")
    print(f"圖片不存在筆數：{report['missing_image_rows']}")
    print(f"圖片無法讀取筆數：{report['unreadable_image_rows']}")

    print("\n輸出檔案：")
    print(f"- {CLEAN_CSV_PATH}")
    print(f"- {REPORT_PATH}")

    if report["bad_rows"] > 0:
        print(f"- {BAD_ROWS_PATH}")


if __name__ == "__main__":
    main()