from pathlib import Path
import re
import pandas as pd


# ============================================================
# 10_dataset_query_audit.py
# 目的：
# 先不要再調搜尋排序，而是檢查「資料本身」是否有足夠候選商品。
#
# 使用方式：
# cd C:\Users\yujun\Desktop\fashion_rag_project
# .\.venv\Scripts\Activate.ps1
# python scripts\10_dataset_query_audit.py
# ============================================================


# =========================
# 1. 路徑設定
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INPUT_CSV_PATH = DATA_DIR / "products_with_english_caption.csv"

REPORT_CSV_PATH = OUTPUT_DIR / "query_audit_report.csv"
REPORT_TXT_PATH = OUTPUT_DIR / "query_audit_report.txt"


# =========================
# 2. 基本工具
# =========================

UNKNOWN_VALUES = {"", "未知", "nan", "none", "null"}


def normalize_text(text) -> str:
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
    text = re.sub(r"t\s*恤", "t恤", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_valid_value(value) -> bool:
    if pd.isna(value):
        return False

    return str(value).strip().lower() not in UNKNOWN_VALUES


def contains_any(text: str, terms: list[str]) -> bool:
    text = normalize_text(text)

    for term in terms:
        if normalize_text(term) in text:
            return True

    return False


def split_pipe_text(value) -> list[str]:
    if not is_valid_value(value):
        return []

    return [x.strip() for x in str(value).split("|") if is_valid_value(x)]


def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


# =========================
# 3. 查詢解析規則
# =========================

COLOR_RULES = {
    "白色": ["白色", "白", "米白", "乳白", "奶白", "象牙白", "本白", "珍珠白"],
    "黑色": ["黑色", "黑"],
    "灰色": ["灰色", "淺灰", "浅灰", "深灰", "灰"],
    "米色": ["米色", "杏色", "卡其", "奶油色", "燕麥色", "燕麦色"],
    "粉色": ["粉色", "粉紅", "粉红", "藕粉"],
    "藍色": ["藍色", "蓝色", "淺藍", "浅蓝", "天藍", "天蓝", "牛仔藍", "牛仔蓝"],
    "紅色": ["紅色", "红色", "酒紅", "酒红"],
    "黃色": ["黃色", "黄色", "鵝黃", "鹅黄", "奶黃", "奶黄"],
    "綠色": ["綠色", "绿色", "軍綠", "军绿", "墨綠", "墨绿"],
    "棕色": ["棕色", "咖啡", "褐色", "摩卡", "巧克力"],
    "紫色": ["紫色", "薰衣草", "香芋"],
}

SUBCATEGORY_RULES = {
    "襯衫": ["襯衫", "衬衫", "襯衣", "衬衣", "shirt", "blouse"],
    "高領上衣": ["高領", "高领", "半高領", "半高领", "堆堆領", "堆堆领", "turtleneck"],
    "針織衫": ["針織", "针织", "knit"],
    "毛衣": ["毛衣", "sweater"],
    "T恤": ["t恤", "t-shirt", "tee"],
    "衛衣": ["衛衣", "卫衣", "帽t", "hoodie", "sweatshirt"],
    "背心": ["背心", "吊帶", "吊带", "細肩帶", "细肩带", "tank", "camisole"],
    "Polo衫": ["polo"],
    "露肩上衣": ["露肩", "一字肩", "斜肩", "off shoulder"],
    "洋裝": ["洋裝", "洋装", "連衣裙", "连衣裙", "dress"],
    "短裙": ["短裙", "mini skirt"],
    "長裙": ["長裙", "长裙", "半身裙", "skirt"],
    "牛仔褲": ["牛仔褲", "牛仔裤", "jeans"],
    "短褲": ["短褲", "短裤", "shorts"],
    "長褲": ["長褲", "长裤", "西裝褲", "西装裤", "trousers", "pants"],
    "西裝外套": ["西裝外套", "西装外套", "西服外套", "blazer"],
    "針織外套": ["針織外套", "针织外套", "開衫", "开衫", "cardigan"],
    "牛仔外套": ["牛仔外套", "denim jacket"],
    "夾克": ["夾克", "夹克", "jacket"],
    "大衣": ["大衣", "風衣", "风衣", "coat", "trench"],
}

COLLAR_RULES = {
    "高領": ["高領", "高领", "半高領", "半高领", "堆堆領", "堆堆领", "turtleneck"],
    "翻領": ["翻領", "翻领", "polo領", "polo领", "襯衫領", "衬衫领", "lapel"],
    "V領": ["v領", "v领", "v-neck"],
    "圓領": ["圓領", "圆领", "crew neck", "round neck"],
    "方領": ["方領", "方领", "square neck"],
    "U領": ["u領", "u领", "u-neck"],
    "一字領": ["一字領", "一字领", "露肩", "off shoulder"],
}

SLEEVE_RULES = {
    "長袖": ["長袖", "长袖", "long sleeve"],
    "短袖": ["短袖", "short sleeve"],
    "無袖": ["無袖", "无袖", "sleeveless"],
    "七分袖": ["七分袖", "中袖"],
    "細肩帶": ["細肩帶", "细肩带", "吊帶", "吊带", "spaghetti strap"],
}

STYLE_RULES = {
    "簡約": ["簡約", "简约", "極簡", "极简", "純色", "纯色", "素色", "basic", "minimal", "simple", "clean"],
    "通勤": ["通勤", "上班", "職場", "职场", "ol", "office", "workwear"],
    "氣質": ["氣質", "气质", "優雅", "优雅", "elegant", "refined"],
    "甜美": ["甜美", "可愛", "可爱", "少女", "sweet"],
    "辣妹": ["辣妹", "性感", "修身", "緊身", "紧身", "sexy"],
    "休閒": ["休閒", "休闲", "日常", "casual"],
    "街頭": ["街頭", "街头", "酷", "帥氣", "帅气", "street"],
    "復古": ["復古", "复古", "vintage"],
    "韓系": ["韓系", "韩系", "韓", "korean"],
    "法式": ["法式", "french"],
    "度假": ["度假", "海邊", "海边", "沙灘", "沙滩", "resort", "beach"],
}

OCCASION_RULES = {
    "上班": ["上班", "通勤", "職場", "职场", "ol", "office", "workwear"],
    "日常": ["日常", "休閒", "休闲", "百搭", "casual"],
    "約會": ["約會", "约会", "date"],
    "度假": ["度假", "海邊", "海边", "沙灘", "沙滩", "resort", "beach"],
    "派對": ["派對", "派对", "party"],
    "正式場合": ["正式", "宴會", "宴会", "formal"],
}


BAD_SHIRT_TERMS = [
    "t恤", "t-shirt", "tee",
    "打底", "底襯衫", "底衬衫",
    "疊穿", "叠穿", "分層", "分层",
    "二合一", "假兩件", "假两件",
    "拼接", "露肩", "一字肩",
    "性感", "辣妹", "海灘", "海滩", "度假",
    "蕾絲", "蕾丝"
]


OFFICE_POSITIVE_TERMS = [
    "通勤", "上班", "職場", "职场", "ol",
    "氣質", "气质", "優雅", "优雅",
    "翻領", "翻领", "襯衫領", "衬衫领",
    "紐扣", "纽扣", "扣", "office", "workwear", "commuter", "elegant"
]


MINIMAL_POSITIVE_TERMS = [
    "簡約", "简约", "純色", "纯色", "素色", "basic", "minimal", "simple", "clean"
]


# =========================
# 4. 查詢解析
# =========================

def match_one(q: str, rules: dict) -> str:
    for label, terms in rules.items():
        if contains_any(q, terms):
            return label
    return ""


def match_many(q: str, rules: dict) -> list[str]:
    matched = []

    for label, terms in rules.items():
        if contains_any(q, terms):
            matched.append(label)

    return matched


def infer_category_from_query(query_text: str):
    q = normalize_text(query_text)

    if contains_any(q, [
        "上衣", "襯衫", "衬衫", "襯衣", "衬衣", "t恤", "t-shirt",
        "高領", "高领", "毛衣", "針織", "针织", "背心",
        "shirt", "blouse", "top", "turtleneck", "sweater", "knit"
    ]):
        return 1

    if contains_any(q, ["套裝", "套装", "set", "two piece", "two-piece"]):
        return 2

    if contains_any(q, [
        "褲", "裤", "長褲", "长裤", "短褲", "短裤",
        "牛仔褲", "牛仔裤", "pants", "jeans", "trousers", "shorts"
    ]):
        return 3

    if contains_any(q, [
        "裙", "短裙", "長裙", "长裙", "洋裝", "洋装",
        "連衣裙", "连衣裙", "skirt", "dress"
    ]):
        return 4

    if contains_any(q, [
        "外套", "夾克", "夹克", "大衣", "西裝外套", "西装外套",
        "jacket", "coat", "blazer"
    ]):
        return 5

    if contains_any(q, ["內衣", "内衣", "內著", "内著", "bra", "underwear", "lingerie"]):
        return 6

    return None


def parse_query(query_text: str) -> dict:
    q = normalize_text(query_text)

    return {
        "raw_query": query_text,
        "category": infer_category_from_query(q),
        "subcategory": match_one(q, SUBCATEGORY_RULES),
        "color": match_one(q, COLOR_RULES),
        "collar": match_one(q, COLLAR_RULES),
        "sleeve": match_one(q, SLEEVE_RULES),
        "styles": match_many(q, STYLE_RULES),
        "occasions": match_many(q, OCCASION_RULES),
    }


# =========================
# 5. 資料讀取
# =========================

def load_products() -> pd.DataFrame:
    if not INPUT_CSV_PATH.exists():
        raise FileNotFoundError(f"找不到檔案：{INPUT_CSV_PATH}")

    df = pd.read_csv(INPUT_CSV_PATH)

    required_cols = [
        "id",
        "product_name",
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
        "english_caption",
        "image_path",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"缺少必要欄位：{missing_cols}")

    df["id"] = df["id"].astype(str)
    df["product_name"] = df["product_name"].fillna("").astype(str)
    df["english_caption"] = df["english_caption"].fillna("").astype(str)

    return df


# =========================
# 6. 篩選邏輯
# =========================

def mask_category(df: pd.DataFrame, profile: dict) -> pd.Series:
    if profile["category"] is None:
        return pd.Series(True, index=df.index)

    return df["category"].astype(int) == int(profile["category"])


def mask_subcategory_keyword(df: pd.DataFrame, profile: dict) -> pd.Series:
    query_sub = profile.get("subcategory", "")

    if not query_sub:
        return pd.Series(True, index=df.index)

    terms = SUBCATEGORY_RULES.get(query_sub, [])

    name_mask = df["product_name"].apply(lambda x: contains_any(x, terms))
    sub_mask = df["subcategory"].astype(str) == query_sub

    return name_mask | sub_mask


def mask_color(df: pd.DataFrame, profile: dict) -> pd.Series:
    query_color = profile.get("color", "")

    if not query_color:
        return pd.Series(True, index=df.index)

    # 欄位顏色已知且正確，或商品名稱有該顏色
    terms = COLOR_RULES.get(query_color, [])
    name_has_target = df["product_name"].apply(lambda x: contains_any(x, terms))
    field_target = df["color"].astype(str) == query_color

    # 欄位顏色已知但不同者排除
    field_known_wrong = df["color"].apply(is_valid_value) & (df["color"].astype(str) != query_color)

    # 商品名稱明確含其他顏色者排除
    name_has_other_color = pd.Series(False, index=df.index)

    for color_label, color_terms in COLOR_RULES.items():
        if color_label == query_color:
            continue
        name_has_other_color = name_has_other_color | df["product_name"].apply(lambda x: contains_any(x, color_terms))

    return (field_target | name_has_target | ~df["color"].apply(is_valid_value)) & (~field_known_wrong) & (~name_has_other_color)


def mask_collar(df: pd.DataFrame, profile: dict) -> pd.Series:
    query_collar = profile.get("collar", "")

    if not query_collar:
        return pd.Series(True, index=df.index)

    terms = COLLAR_RULES.get(query_collar, [])

    name_has_target = df["product_name"].apply(lambda x: contains_any(x, terms))
    field_target = df["collar"].astype(str) == query_collar
    field_unknown = ~df["collar"].apply(is_valid_value)

    return name_has_target | field_target | field_unknown


def mask_sleeve(df: pd.DataFrame, profile: dict) -> pd.Series:
    query_sleeve = profile.get("sleeve", "")

    if not query_sleeve:
        return pd.Series(True, index=df.index)

    terms = SLEEVE_RULES.get(query_sleeve, [])

    name_has_target = df["product_name"].apply(lambda x: contains_any(x, terms))
    field_target = df["sleeve"].astype(str) == query_sleeve
    field_unknown = ~df["sleeve"].apply(is_valid_value)

    return name_has_target | field_target | field_unknown


def mask_remove_bad_shirt(df: pd.DataFrame, profile: dict) -> pd.Series:
    query_sub = profile.get("subcategory", "")

    if query_sub != "襯衫":
        return pd.Series(True, index=df.index)

    return ~df["product_name"].apply(lambda x: contains_any(x, BAD_SHIRT_TERMS))


def mask_office_or_minimal(df: pd.DataFrame, profile: dict, mode: str) -> pd.Series:
    """
    mode:
    - "relaxed": 只要命中任一相關語意
    - "strict": 需要更接近上班 / 通勤 / 簡約
    """

    query_styles = set(profile.get("styles", []))
    query_occasions = set(profile.get("occasions", []))

    is_office_query = ("通勤" in query_styles) or ("上班" in query_occasions)
    is_minimal_query = "簡約" in query_styles
    is_elegant_query = "氣質" in query_styles

    if not is_office_query and not is_minimal_query and not is_elegant_query:
        return pd.Series(True, index=df.index)

    def row_match(row) -> bool:
        name = normalize_text(row.get("product_name", ""))
        caption = normalize_text(row.get("english_caption", ""))
        combined = name + " " + caption

        row_styles = set(split_pipe_text(row.get("style_keywords", "")))
        row_occasions = set(split_pipe_text(row.get("occasion_keywords", "")))

        hits = 0

        if is_office_query:
            if (
                "通勤" in row_styles
                or "氣質" in row_styles
                or "上班" in row_occasions
                or contains_any(combined, OFFICE_POSITIVE_TERMS)
            ):
                hits += 1

        if is_minimal_query:
            if "簡約" in row_styles or contains_any(combined, MINIMAL_POSITIVE_TERMS):
                hits += 1

        if is_elegant_query:
            if "氣質" in row_styles or contains_any(combined, ["氣質", "气质", "優雅", "优雅", "elegant", "refined"]):
                hits += 1

        if mode == "strict":
            required = 0
            if is_office_query:
                required += 1
            if is_minimal_query:
                required += 1
            if is_elegant_query:
                required += 1

            return hits >= max(1, required)

        return hits >= 1

    return df.apply(row_match, axis=1)


# =========================
# 7. 稽核流程
# =========================

def audit_query(df: pd.DataFrame, query_text: str) -> tuple[list[dict], pd.DataFrame]:
    profile = parse_query(query_text)

    stages = []

    def add_stage(stage_name: str, current_df: pd.DataFrame):
        stages.append({
            "query": query_text,
            "stage": stage_name,
            "count": len(current_df),
            "profile": str(profile),
        })

    current = df.copy()
    add_stage("00_all_rows", current)

    current = current[mask_category(current, profile)]
    add_stage("01_category_filter", current)

    current = current[mask_subcategory_keyword(current, profile)]
    add_stage("02_subcategory_or_keyword_filter", current)

    current = current[mask_color(current, profile)]
    add_stage("03_color_filter", current)

    current = current[mask_collar(current, profile)]
    add_stage("04_collar_filter", current)

    current = current[mask_sleeve(current, profile)]
    add_stage("05_sleeve_filter", current)

    current = current[mask_remove_bad_shirt(current, profile)]
    add_stage("06_remove_bad_shirt_terms", current)

    relaxed = current[mask_office_or_minimal(current, profile, mode="relaxed")]
    add_stage("07_style_occasion_relaxed", relaxed)

    strict = current[mask_office_or_minimal(current, profile, mode="strict")]
    add_stage("08_style_occasion_strict", strict)

    return stages, strict


def sample_candidates(df: pd.DataFrame, max_rows: int = 20) -> pd.DataFrame:
    cols = [
        "id",
        "product_name",
        "category_name",
        "subcategory",
        "color",
        "collar",
        "sleeve",
        "style_keywords",
        "occasion_keywords",
        "product_components",
        "image_path",
    ]

    existing_cols = [col for col in cols if col in df.columns]

    return df[existing_cols].head(max_rows).copy()


# =========================
# 8. 主程式
# =========================

def main():
    print("啟動 Dataset Query Audit")
    print("========================")
    print(f"讀取資料：{INPUT_CSV_PATH}")

    df = load_products()

    print(f"商品總筆數：{len(df)}")

    default_queries = [
        "白色高領上衣",
        "適合上班的簡約襯衫",
        "氣質通勤翻領長袖襯衫",
        "黑色短裙",
        "海邊度假風洋裝",
    ]

    all_stage_rows = []
    txt_lines = []

    for query in default_queries:
        stages, final_candidates = audit_query(df, query)
        all_stage_rows.extend(stages)

        profile = parse_query(query)

        txt_lines.append("=" * 100)
        txt_lines.append(f"QUERY: {query}")
        txt_lines.append(f"PROFILE: {profile}")
        txt_lines.append("-" * 100)

        stage_df = pd.DataFrame(stages)
        txt_lines.append(stage_df[["stage", "count"]].to_string(index=False))
        txt_lines.append("")
        txt_lines.append("FINAL CANDIDATE SAMPLES:")
        txt_lines.append(sample_candidates(final_candidates, max_rows=20).to_string(index=False))
        txt_lines.append("")

        print("\n" + "=" * 80)
        print(f"查詢：{query}")
        print(f"解析：{profile}")
        print(pd.DataFrame(stages)[["stage", "count"]].to_string(index=False))

        print("\n最終候選樣本：")
        print(sample_candidates(final_candidates, max_rows=10).to_string(index=False))

    OUTPUT_DIR.mkdir(exist_ok=True)

    report_df = pd.DataFrame(all_stage_rows)
    report_df.to_csv(REPORT_CSV_PATH, index=False, encoding="utf-8-sig")

    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    print("\n完成。")
    print(f"CSV 報告：{REPORT_CSV_PATH}")
    print(f"TXT 報告：{REPORT_TXT_PATH}")


if __name__ == "__main__":
    main()
