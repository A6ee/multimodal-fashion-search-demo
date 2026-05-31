from pathlib import Path
import pandas as pd


# =========================
# 1. 路徑設定
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INPUT_CSV_PATH = DATA_DIR / "products_with_attributes.csv"
OUTPUT_CSV_PATH = DATA_DIR / "products_with_english_caption.csv"
REPORT_PATH = OUTPUT_DIR / "english_caption_report.txt"


# =========================
# 2. 基本工具
# =========================

UNKNOWN_VALUES = {"", "未知", "nan", "none", "null"}


def is_valid_value(value) -> bool:
    if pd.isna(value):
        return False

    value = str(value).strip().lower()
    return value not in UNKNOWN_VALUES


def split_pipe_text(value) -> list[str]:
    if not is_valid_value(value):
        return []

    return [x.strip() for x in str(value).split("|") if is_valid_value(x)]


def add_unique(items: list[str], value: str):
    if is_valid_value(value) and value not in items:
        items.append(value)


def add_many_unique(items: list[str], values: list[str]):
    for value in values:
        add_unique(items, value)


# =========================
# 3. 中文屬性 → 英文詞彙
# =========================

CATEGORY_EN = {
    "上衣": "top",
    "套裝": "two-piece outfit set",
    "褲子": "pants",
    "裙子": "skirt or dress",
    "外套": "outerwear jacket",
    "內著": "innerwear lingerie",
}


SUBCATEGORY_EN = {
    "上衣": "top",
    "襯衫": "blouse shirt",
    "高領上衣": "turtleneck top",
    "針織衫": "knit top",
    "毛衣": "sweater",
    "T恤": "t-shirt",
    "衛衣": "hoodie sweatshirt",
    "背心": "tank top camisole",
    "Polo衫": "polo shirt",
    "露肩上衣": "off-shoulder top",

    "套裝": "two-piece outfit set",

    "洋裝": "dress",
    "短裙": "mini skirt",
    "長裙": "long skirt",
    "裙子": "skirt",

    "牛仔褲": "denim jeans",
    "短褲": "shorts",
    "長褲": "long pants trousers",
    "褲子": "pants",

    "西裝外套": "blazer",
    "針織外套": "knit cardigan",
    "牛仔外套": "denim jacket",
    "背心外套": "vest outerwear",
    "夾克": "jacket",
    "大衣": "coat",
    "外套": "outerwear jacket",

    "內衣": "bra lingerie",
    "內褲": "panty underwear",
    "睡衣": "pajamas loungewear",
    "內著": "innerwear lingerie",
}


COLOR_EN = {
    "白色": "white",
    "黑色": "black",
    "灰色": "gray",
    "米色": "beige cream",
    "粉色": "pink",
    "藍色": "blue",
    "紅色": "red",
    "黃色": "yellow",
    "綠色": "green",
    "棕色": "brown",
    "紫色": "purple",
}


COLLAR_EN = {
    "高領": "turtleneck high neck",
    "翻領": "collared lapel",
    "V領": "v-neck",
    "圓領": "round neck crew neck",
    "方領": "square neck",
    "U領": "u-neck",
    "一字領": "off-shoulder neckline",
}


SLEEVE_EN = {
    "長袖": "long sleeve",
    "短袖": "short sleeve",
    "無袖": "sleeveless",
    "七分袖": "three-quarter sleeve",
    "細肩帶": "spaghetti strap",
}


MATERIAL_EN = {
    "針織": "knit fabric",
    "棉": "cotton",
    "羊毛": "wool",
    "雪紡": "chiffon",
    "牛仔": "denim",
    "皮革": "leather",
    "蕾絲": "lace",
    "毛呢": "woolen fabric",
    "絨": "velvet corduroy fleece",
}


STYLE_EN = {
    "簡約": "minimal simple clean",
    "通勤": "office workwear commuter",
    "氣質": "elegant refined feminine",
    "甜美": "sweet cute feminine",
    "辣妹": "sexy trendy slim fit",
    "休閒": "casual daily",
    "街頭": "street style edgy cool",
    "復古": "vintage retro",
    "韓系": "korean style",
    "法式": "french chic",
    "度假": "resort vacation beach",
}


OCCASION_EN = {
    "上班": "office work",
    "日常": "daily casual wear",
    "約會": "date outfit",
    "度假": "vacation resort beach",
    "派對": "party",
    "正式場合": "formal occasion",
}


COMPONENT_EN = {
    "襯衫": "shirt blouse",
    "T恤": "t-shirt",
    "高領": "turtleneck",
    "針織": "knit",
    "毛衣": "sweater",
    "背心": "tank top vest",
    "外套": "jacket outerwear",
    "短裙": "mini skirt",
    "長裙": "long skirt",
    "裙子": "skirt",
    "洋裝": "dress",
    "長褲": "long pants",
    "短褲": "shorts",
    "牛仔": "denim",
}


# =========================
# 4. Caption 生成邏輯
# =========================

def map_single(value, mapping: dict) -> str:
    if not is_valid_value(value):
        return ""

    value = str(value).strip()
    return mapping.get(value, "")


def map_pipe_values(value, mapping: dict) -> list[str]:
    output = []

    for item in split_pipe_text(value):
        mapped = mapping.get(item, "")
        if is_valid_value(mapped):
            output.append(mapped)

    return output


def build_english_caption(row: pd.Series) -> str:
    """
    根據商品屬性生成英文 caption。
    這不是直接翻譯商品名稱，而是建立適合 OpenCLIP 文字向量的商品描述。
    """

    terms = []

    # 1. 顏色
    color_en = map_single(row.get("color", ""), COLOR_EN)
    add_unique(terms, color_en)

    # 2. 領口與袖長
    collar_en = map_single(row.get("collar", ""), COLLAR_EN)
    sleeve_en = map_single(row.get("sleeve", ""), SLEEVE_EN)

    add_unique(terms, collar_en)
    add_unique(terms, sleeve_en)

    # 3. 細分類是最重要的商品品類描述
    subcategory_en = map_single(row.get("subcategory", ""), SUBCATEGORY_EN)
    category_en = map_single(row.get("category_name", ""), CATEGORY_EN)

    if is_valid_value(subcategory_en):
        add_unique(terms, subcategory_en)
    else:
        add_unique(terms, category_en)

    # 4. 材質
    material_en = map_single(row.get("material", ""), MATERIAL_EN)
    add_unique(terms, material_en)

    # 5. 風格與場合
    style_terms = map_pipe_values(row.get("style_keywords", ""), STYLE_EN)
    occasion_terms = map_pipe_values(row.get("occasion_keywords", ""), OCCASION_EN)
    # component_terms = map_pipe_values(row.get("product_components", ""), COMPONENT_EN)

    add_many_unique(terms, style_terms)
    add_many_unique(terms, occasion_terms)

    # components 不要放太前面，避免干擾主品類
    # add_many_unique(terms, component_terms)

    # 6. 加上固定語境，讓 OpenCLIP 更知道這是服飾商品圖
    add_unique(terms, "fashion e-commerce clothing product photo")
    add_unique(terms, "women's clothing")

    # 7. 組成 caption
    caption = ", ".join([t for t in terms if is_valid_value(t)])

    if not is_valid_value(caption):
        caption = "women's fashion e-commerce clothing product photo"

    return caption


def build_english_search_text(row: pd.Series) -> str:
    """
    給文字向量庫使用的較長英文文字。
    caption 偏自然描述，search_text 偏檢索用，把更多英文屬性展開。
    """

    parts = []

    caption = row.get("english_caption", "")
    add_unique(parts, caption)

    # 加入各欄位英文展開
    add_unique(parts, map_single(row.get("category_name", ""), CATEGORY_EN))
    add_unique(parts, map_single(row.get("subcategory", ""), SUBCATEGORY_EN))
    add_unique(parts, map_single(row.get("color", ""), COLOR_EN))
    add_unique(parts, map_single(row.get("collar", ""), COLLAR_EN))
    add_unique(parts, map_single(row.get("sleeve", ""), SLEEVE_EN))
    add_unique(parts, map_single(row.get("material", ""), MATERIAL_EN))

    add_many_unique(parts, map_pipe_values(row.get("style_keywords", ""), STYLE_EN))
    add_many_unique(parts, map_pipe_values(row.get("occasion_keywords", ""), OCCASION_EN))
    add_many_unique(parts, map_pipe_values(row.get("product_components", ""), COMPONENT_EN))

    return " ".join([p for p in parts if is_valid_value(p)])


def caption_quality_level(row: pd.Series) -> str:
    """
    粗略判斷 caption 資訊量。
    """

    score = 0

    for col in [
        "subcategory",
        "color",
        "collar",
        "sleeve",
        "material",
        "style_keywords",
        "occasion_keywords",
        "product_components",
    ]:
        if is_valid_value(row.get(col, "")):
            score += 1

    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


# =========================
# 5. 報告輸出
# =========================

def write_report(df: pd.DataFrame):
    lines = []

    lines.append("English Caption Report")
    lines.append("======================")
    lines.append("")
    lines.append(f"total_rows: {len(df)}")
    lines.append(f"empty_english_caption: {(~df['english_caption'].apply(is_valid_value)).sum()}")
    lines.append("")

    lines.append("caption_quality counts:")
    lines.append(str(df["caption_quality"].value_counts(dropna=False)))
    lines.append("")

    lines.append("caption length summary:")
    lines.append(str(df["english_caption"].astype(str).str.len().describe()))
    lines.append("")

    lines.append("sample captions:")
    sample_cols = [
        "id",
        "product_name",
        "category_name",
        "subcategory",
        "color",
        "collar",
        "sleeve",
        "style_keywords",
        "occasion_keywords",
        "english_caption",
    ]

    sample_df = df[sample_cols].head(20)

    lines.append(sample_df.to_string(index=False))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 6. 主程式
# =========================

def main():
    print("專案根目錄：", PROJECT_ROOT)
    print("讀取資料：", INPUT_CSV_PATH)

    if not INPUT_CSV_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {INPUT_CSV_PATH}，請先執行 scripts\\04_prepare_product_attributes.py"
        )

    df = pd.read_csv(INPUT_CSV_PATH)

    required_cols = [
        "id",
        "product_name",
        "image_path",
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
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"products_with_attributes.csv 缺少欄位：{missing_cols}")

    print(f"商品筆數：{len(df)}")

    df["english_caption"] = df.apply(build_english_caption, axis=1)
    df["english_search_text"] = df.apply(build_english_search_text, axis=1)
    df["caption_quality"] = df.apply(caption_quality_level, axis=1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    write_report(df)

    print("\n英文 caption 生成完成")
    print("======================")
    print(f"輸出資料：{OUTPUT_CSV_PATH}")
    print(f"輸出報告：{REPORT_PATH}")
    print(f"總筆數：{len(df)}")
    print(f"空 caption 筆數：{(~df['english_caption'].apply(is_valid_value)).sum()}")

    print("\n資料預覽：")
    preview_cols = [
        "id",
        "product_name",
        "subcategory",
        "color",
        "collar",
        "sleeve",
        "style_keywords",
        "occasion_keywords",
        "english_caption",
        "caption_quality",
    ]

    print(df[preview_cols].head(15))


if __name__ == "__main__":
    main()