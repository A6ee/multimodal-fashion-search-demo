from pathlib import Path
import re
import pandas as pd


# =========================
# 1. 路徑設定
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INPUT_CSV_PATH = DATA_DIR / "products_clean.csv"
OUTPUT_CSV_PATH = DATA_DIR / "products_with_attributes.csv"
REPORT_PATH = OUTPUT_DIR / "attribute_extract_report.txt"


# =========================
# 2. 基本工具
# =========================

UNKNOWN_VALUES = {"", "未知", "nan", "none", "null"}


def normalize_text(text) -> str:
    """
    將商品名稱標準化，方便規則比對。
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("　", " ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace("/", " ")
    text = text.replace("／", " ")
    text = text.replace("，", " ")
    text = text.replace(",", " ")
    text = text.replace("。", " ")
    text = text.replace("、", " ")

    # 常見中英混合服飾詞標準化
    text = re.sub(r"t\s*恤", "t恤", text)
    text = re.sub(r"t\s*-?\s*shirt", "t-shirt", text)

    # 空白壓縮
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_valid_value(value) -> bool:
    if pd.isna(value):
        return False
    value = str(value).strip().lower()
    return value not in UNKNOWN_VALUES


def match_first(text: str, rules: list[tuple[str, list[str]]], default="未知") -> str:
    for label, keywords in rules:
        for kw in keywords:
            if kw.lower() in text:
                return label
    return default


def match_all(text: str, rules: list[tuple[str, list[str]]]) -> list[str]:
    matched = []

    for label, keywords in rules:
        for kw in keywords:
            if kw.lower() in text:
                matched.append(label)
                break

    return matched


def list_to_str(items: list[str]) -> str:
    if not items:
        return ""
    return "|".join(items)


def unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    output = []

    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)

    return output


# =========================
# 3. 規則表
# =========================

COLOR_RULES = [
    ("白色", ["白色", "米白", "乳白", "奶白", "象牙白", "本白", "珍珠白", "米白色", "white"]),
    ("黑色", ["黑色", "黑", "black"]),
    ("灰色", ["灰色", "浅灰", "淺灰", "深灰", "灰", "gray", "grey"]),
    ("米色", ["米色", "杏色", "卡其", "奶油色", "燕麦色", "燕麥色", "beige", "cream"]),
    ("粉色", ["粉色", "粉紅", "粉红", "藕粉", "玫粉", "pink"]),
    ("藍色", ["藍色", "蓝色", "天蓝", "天藍", "浅蓝", "淺藍", "牛仔蓝", "牛仔藍", "blue"]),
    ("紅色", ["紅色", "红色", "酒紅", "酒红", "正红", "正紅", "red"]),
    ("黃色", ["黃色", "黄色", "鹅黄", "鵝黃", "奶黄", "奶黃", "yellow"]),
    ("綠色", ["綠色", "绿色", "軍綠", "军绿", "墨绿", "墨綠", "green"]),
    ("棕色", ["棕色", "咖啡", "褐色", "摩卡", "巧克力", "brown"]),
    ("紫色", ["紫色", "薰衣草", "香芋", "purple"]),
]


# 上衣類細分
TOP_SUBCATEGORY_RULES = [
    ("襯衫", ["襯衫", "衬衫", "襯衣", "衬衣", "shirt", "blouse"]),
    ("高領上衣", ["高領", "高领", "半高領", "半高领", "堆堆領", "堆堆领", "turtleneck"]),
    ("針織衫", ["針織", "针织", "knit"]),
    ("毛衣", ["毛衣", "sweater"]),
    ("T恤", ["t恤", "t-shirt", "tee", "短袖t", "短袖 t"]),
    ("衛衣", ["衛衣", "卫衣", "帽t", "帽 t", "hoodie", "sweatshirt"]),
    ("背心", ["背心", "吊帶", "吊带", "細肩帶", "细肩带", "tank", "camisole"]),
    ("Polo衫", ["polo"]),
    ("露肩上衣", ["露肩", "一字肩", "斜肩", "off shoulder"]),
]


# 裙子類細分
SKIRT_SUBCATEGORY_RULES = [
    ("洋裝", ["洋裝", "洋装", "連衣裙", "连衣裙", "衬衫裙", "襯衫裙", "dress"]),
    ("短裙", ["短裙", "mini skirt", "迷你裙"]),
    ("長裙", ["長裙", "长裙", "半身裙", "中長裙", "中长裙", "skirt"]),
    ("裙子", ["裙"]),
]


# 褲子類細分
PANTS_SUBCATEGORY_RULES = [
    ("牛仔褲", ["牛仔褲", "牛仔裤", "jeans"]),
    ("短褲", ["短褲", "短裤", "shorts"]),
    ("長褲", ["長褲", "长裤", "西裝褲", "西装裤", "闊腿褲", "阔腿裤", "trousers", "pants"]),
]


# 外套類細分
OUTER_SUBCATEGORY_RULES = [
    ("西裝外套", ["西裝外套", "西装外套", "西服外套", "blazer"]),
    ("針織外套", ["針織外套", "针织外套", "開衫", "开衫", "cardigan"]),
    ("牛仔外套", ["牛仔外套", "denim jacket"]),
    ("背心外套", ["背心外套", "馬甲", "马甲", "vest"]),
    ("夾克", ["夾克", "夹克", "jacket"]),
    ("大衣", ["大衣", "風衣", "风衣", "coat", "trench"]),
]


# 套裝類細分
SET_SUBCATEGORY_RULES = [
    ("套裝", ["套裝", "套装", "兩件套", "两件套", "三件套", "set", "two piece", "two-piece"]),
]


# 內著類細分
INNER_SUBCATEGORY_RULES = [
    ("內衣", ["內衣", "内衣", "bra"]),
    ("內褲", ["內褲", "内裤", "panty", "brief"]),
    ("睡衣", ["睡衣", "家居服", "pajamas"]),
    ("內著", ["內著", "内著", "underwear", "lingerie"]),
]


COLLAR_RULES = [
    ("高領", ["高領", "高领", "半高領", "半高领", "堆堆領", "堆堆领", "turtleneck"]),
    ("翻領", ["翻領", "翻领", "polo領", "polo领", "襯衫領", "衬衫领", "lapel"]),
    ("V領", ["v領", "v领", "v-neck"]),
    ("圓領", ["圓領", "圆领", "crew neck", "round neck"]),
    ("方領", ["方領", "方领", "square neck"]),
    ("U領", ["u領", "u领", "u-neck"]),
    ("一字領", ["一字領", "一字领", "露肩", "off shoulder"]),
]


SLEEVE_RULES = [
    ("長袖", ["長袖", "长袖", "long sleeve"]),
    ("短袖", ["短袖", "short sleeve"]),
    ("無袖", ["無袖", "无袖", "sleeveless"]),
    ("七分袖", ["七分袖", "中袖"]),
    ("細肩帶", ["細肩帶", "细肩带", "吊帶", "吊带", "spaghetti strap"]),
]


MATERIAL_RULES = [
    ("針織", ["針織", "针织", "knit"]),
    ("棉", ["棉", "cotton"]),
    ("羊毛", ["羊毛", "wool"]),
    ("雪紡", ["雪紡", "雪纺", "chiffon"]),
    ("牛仔", ["牛仔", "denim"]),
    ("皮革", ["皮革", "皮衣", "leather"]),
    ("蕾絲", ["蕾絲", "蕾丝", "lace"]),
    ("毛呢", ["毛呢", "呢料"]),
    ("絨", ["絨", "绒", "velvet", "corduroy"]),
]


STYLE_RULES = [
    ("簡約", ["簡約", "简约", "極簡", "极简", "純色", "纯色", "素色", "basic", "minimal"]),
    ("通勤", ["通勤", "上班", "職場", "职场", "ol", "office"]),
    ("氣質", ["氣質", "气质", "優雅", "优雅", "elegant"]),
    ("甜美", ["甜美", "可愛", "可爱", "少女", "甜", "sweet"]),
    ("辣妹", ["辣妹", "性感", "修身", "緊身", "紧身", "顯身材", "显身材", "sexy"]),
    ("休閒", ["休閒", "休闲", "日常", "casual"]),
    ("街頭", ["街頭", "街头", "酷", "帥氣", "帅气", "street"]),
    ("復古", ["復古", "复古", "vintage"]),
    ("韓系", ["韓", "韩系", "韓系", "korean"]),
    ("法式", ["法式", "french"]),
    ("度假", ["度假", "海邊", "海边", "沙灘", "沙滩", "resort", "beach"]),
]


OCCASION_RULES = [
    ("上班", ["上班", "通勤", "職場", "职场", "ol", "office"]),
    ("日常", ["日常", "休閒", "休闲", "百搭", "casual"]),
    ("約會", ["約會", "约会", "date"]),
    ("度假", ["度假", "海邊", "海边", "沙灘", "沙滩", "resort", "beach"]),
    ("派對", ["派對", "派对", "party"]),
    ("正式場合", ["正式", "宴會", "宴会", "formal"]),
]


COMPONENT_RULES = [
    ("襯衫", ["襯衫", "衬衫", "襯衣", "衬衣", "shirt", "blouse"]),
    ("T恤", ["t恤", "t-shirt", "tee"]),
    ("高領", ["高領", "高领", "半高領", "半高领", "turtleneck"]),
    ("針織", ["針織", "针织", "knit"]),
    ("毛衣", ["毛衣", "sweater"]),
    ("背心", ["背心", "吊帶", "吊带", "細肩帶", "细肩带", "vest", "tank"]),
    ("外套", ["外套", "夾克", "夹克", "大衣", "jacket", "coat"]),
    ("短裙", ["短裙", "mini skirt"]),
    ("長裙", ["長裙", "长裙", "半身裙", "skirt"]),
    ("洋裝", ["洋裝", "洋装", "連衣裙", "连衣裙", "衬衫裙", "襯衫裙", "dress"]),
    ("長褲", ["長褲", "长裤", "西裝褲", "西装裤", "pants", "trousers"]),
    ("短褲", ["短褲", "短裤", "shorts"]),
    ("牛仔", ["牛仔", "denim", "jeans"]),
]


# =========================
# 4. 依大類抽 subcategory
# =========================

def infer_subcategory_by_category(text: str, category_name: str) -> str:
    """
    根據大類分流判斷細分類，避免跨類誤判。
    """

    category_name = str(category_name).strip()

    if category_name == "上衣":
        return match_first(text, TOP_SUBCATEGORY_RULES, default="上衣")

    if category_name == "套裝":
        return match_first(text, SET_SUBCATEGORY_RULES, default="套裝")

    if category_name == "褲子":
        return match_first(text, PANTS_SUBCATEGORY_RULES, default="褲子")

    if category_name == "裙子":
        return match_first(text, SKIRT_SUBCATEGORY_RULES, default="裙子")

    if category_name == "外套":
        return match_first(text, OUTER_SUBCATEGORY_RULES, default="外套")

    if category_name == "內著":
        return match_first(text, INNER_SUBCATEGORY_RULES, default="內著")

    return "未知"


def extract_product_components(text: str) -> list[str]:
    """
    抽出商品名稱中出現的組成元素。
    例如：襯衫裙可以同時有 襯衫、洋裝。
    """
    components = match_all(text, COMPONENT_RULES)
    return unique_keep_order(components)


def infer_special_cases(text: str, category_name: str, subcategory: str, components: list[str]) -> tuple[str, list[str]]:
    """
    修正一些常見混合品類。
    """

    # 襯衫裙：大類若為裙子，subcategory 應偏向洋裝，而不是襯衫
    if category_name == "裙子" and any(kw in text for kw in ["襯衫裙", "衬衫裙"]):
        subcategory = "洋裝"
        components = unique_keep_order(components + ["襯衫", "洋裝"])

    # 背心外套 / 馬甲：大類若為外套，不要標成一般背心
    if category_name == "外套" and any(kw in text for kw in ["背心外套", "馬甲", "马甲", "vest"]):
        subcategory = "背心外套"
        components = unique_keep_order(components + ["背心", "外套"])

    # 套裝保留組件，不讓 subcategory 被單一品類吃掉
    if category_name == "套裝":
        subcategory = "套裝"

    return subcategory, components


# =========================
# 5. 屬性抽取
# =========================

def extract_attributes(row: pd.Series) -> dict:
    product_name = row.get("product_name", "")
    category_name = row.get("category_name", "")

    text = normalize_text(product_name)

    color = match_first(text, COLOR_RULES, default="未知")
    subcategory = infer_subcategory_by_category(text, category_name)

    collar = match_first(text, COLLAR_RULES, default="")
    sleeve = match_first(text, SLEEVE_RULES, default="")
    material = match_first(text, MATERIAL_RULES, default="")

    styles = match_all(text, STYLE_RULES)
    occasions = match_all(text, OCCASION_RULES)
    components = extract_product_components(text)

    subcategory, components = infer_special_cases(
        text=text,
        category_name=category_name,
        subcategory=subcategory,
        components=components,
    )

    is_office = "上班" in occasions or "通勤" in styles
    is_minimal = "簡約" in styles
    is_casual = "休閒" in styles or "日常" in occasions
    is_vacation = "度假" in styles or "度假" in occasions
    is_sweet = "甜美" in styles
    is_sexy = "辣妹" in styles

    # 粗略屬性完整度：不是品質真值，只是方便你檢查規則抽取覆蓋率
    quality_items = [
        subcategory if subcategory != "未知" else "",
        color if color != "未知" else "",
        collar,
        sleeve,
        material,
        list_to_str(styles),
        list_to_str(occasions),
        list_to_str(components),
    ]
    attribute_quality_score = sum(1 for x in quality_items if is_valid_value(x)) / len(quality_items)

    return {
        "subcategory": subcategory,
        "color": color,
        "collar": collar,
        "sleeve": sleeve,
        "material": material,
        "style_keywords": list_to_str(styles),
        "occasion_keywords": list_to_str(occasions),
        "product_components": list_to_str(components),
        "is_office": is_office,
        "is_minimal": is_minimal,
        "is_casual": is_casual,
        "is_vacation": is_vacation,
        "is_sweet": is_sweet,
        "is_sexy": is_sexy,
        "attribute_quality_score": round(attribute_quality_score, 4),
    }


def build_search_text(row: pd.Series) -> str:
    """
    建立 Hybrid Search 使用的文字欄位。
    v2 重點：不放入 未知 / nan / 空值。
    """

    candidate_parts = [
        row.get("product_name", ""),
        row.get("category_name", ""),
        row.get("subcategory", ""),
        row.get("color", ""),
        row.get("collar", ""),
        row.get("sleeve", ""),
        row.get("material", ""),
        row.get("style_keywords", ""),
        row.get("occasion_keywords", ""),
        row.get("product_components", ""),
    ]

    parts = []

    for value in candidate_parts:
        if not is_valid_value(value):
            continue

        value = str(value).strip()

        if value == "未知":
            continue

        # 把 pipe 也換成空格，讓文字檢索比較好用
        value = value.replace("|", " ")

        parts.append(value)

    return " ".join(parts)


# =========================
# 6. 報告輸出
# =========================

def add_value_counts_to_report(report_lines: list[str], df: pd.DataFrame, col: str):
    report_lines.append(f"{col} counts:")
    report_lines.append(str(df[col].value_counts(dropna=False).head(50)))
    report_lines.append("")


def write_report(output_df: pd.DataFrame):
    report_lines = []
    report_lines.append("Attribute Extract Report v2")
    report_lines.append("===========================")
    report_lines.append("")
    report_lines.append(f"total_rows: {len(output_df)}")
    report_lines.append("")

    for col in [
        "category_name",
        "subcategory",
        "color",
        "collar",
        "sleeve",
        "material",
        "style_keywords",
        "occasion_keywords",
        "product_components",
    ]:
        add_value_counts_to_report(report_lines, output_df, col)

    report_lines.append("boolean summary:")
    bool_cols = [
        "is_office",
        "is_minimal",
        "is_casual",
        "is_vacation",
        "is_sweet",
        "is_sexy",
    ]

    for col in bool_cols:
        report_lines.append(f"{col}: {int(output_df[col].sum())}")

    report_lines.append("")
    report_lines.append("attribute_quality_score summary:")
    report_lines.append(str(output_df["attribute_quality_score"].describe()))

    report_lines.append("")
    report_lines.append("search_text contains unknown:")
    report_lines.append(str(output_df["search_text"].astype(str).str.contains("未知", na=False).sum()))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


# =========================
# 7. 主程式
# =========================

def main():
    print("專案根目錄：", PROJECT_ROOT)
    print("讀取資料：", INPUT_CSV_PATH)

    if not INPUT_CSV_PATH.exists():
        raise FileNotFoundError(f"找不到檔案：{INPUT_CSV_PATH}")

    df = pd.read_csv(INPUT_CSV_PATH)

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
        raise ValueError(f"products_clean.csv 缺少必要欄位：{missing_cols}")

    print(f"原始商品筆數：{len(df)}")

    attribute_rows = []

    for _, row in df.iterrows():
        attrs = extract_attributes(row)
        attribute_rows.append(attrs)

    attr_df = pd.DataFrame(attribute_rows)

    output_df = pd.concat([df, attr_df], axis=1)

    # 補上圖片絕對路徑，後面 05 顯示圖片會更穩
    output_df["absolute_image_path"] = output_df["image_path"].apply(
        lambda p: str(PROJECT_ROOT / str(p))
    )

    output_df["search_text"] = output_df.apply(build_search_text, axis=1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    output_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    write_report(output_df)

    print("\n屬性抽取完成 v2")
    print("================")
    print(f"輸出資料：{OUTPUT_CSV_PATH}")
    print(f"輸出報告：{REPORT_PATH}")
    print(f"總筆數：{len(output_df)}")
    print(f"search_text 含有「未知」筆數：{output_df['search_text'].astype(str).str.contains('未知', na=False).sum()}")

    print("\n資料預覽：")
    preview_cols = [
        "id",
        "product_name",
        "category_name",
        "subcategory",
        "color",
        "collar",
        "sleeve",
        "material",
        "style_keywords",
        "occasion_keywords",
        "product_components",
        "attribute_quality_score",
    ]

    print(output_df[preview_cols].head(15))


if __name__ == "__main__":
    main()