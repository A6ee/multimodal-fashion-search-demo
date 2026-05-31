from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd
import streamlit as st

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from scripts.llm_recommender import generate_grounded_recommendation

# ============================================================
# app.py
# 第一版 Demo：Title-first Fashion Product Search
#
# 執行方式：
# cd C:\Users\yujun\Desktop\fashion_rag_project
# .\.venv\Scripts\Activate.ps1
# streamlit run app.py
# ============================================================


# =========================
# 1. 路徑設定
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

PRODUCT_CSV_PATH = DATA_DIR / "products_with_english_caption.csv"
IMAGE_COLLECTION_NAME = "fashion_products_v1"


# =========================
# 2. 基本設定
# =========================

FINAL_TOP_K = 8
IMAGE_TOP_K = 500

DEMO_QUERIES = [
    "白色高領上衣",
    "黑色短裙",
    "海邊度假風洋裝",
    "甜美短裙",
    "針織外套",
    "灰色大衣",
    "復古洋裝",
    "休閒短褲",
]


# =========================
# 3. 基本工具
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
    return any(normalize_text(term) in text for term in terms)


def split_pipe_text(value) -> list[str]:
    if not is_valid_value(value):
        return []

    return [x.strip() for x in str(value).split("|") if is_valid_value(x)]


def safe_text(value) -> str:
    if not is_valid_value(value):
        return ""
    return str(value)


def display_value(value, default="—") -> str:
    if not is_valid_value(value):
        return default
    return str(value)


def normalize_name_for_dedup(name: str) -> str:
    name = normalize_text(name)
    name = re.sub(r"\s+", "", name)
    return name


# =========================
# 4. 查詢規則
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
    "翻領": ["翻領", "翻领", "polo領", "polo领", "襯衫領", "衬衫领", "lapel", "領", "领"],
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
    "簡約": ["簡約", "简约", "極簡", "极简", "純色", "纯色", "素色", "basic", "minimal", "simple"],
    "通勤": ["通勤", "上班", "職場", "职场", "ol", "office", "workwear"],
    "氣質": ["氣質", "气质", "優雅", "优雅", "elegant"],
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

SUBCATEGORY_EN = {
    "襯衫": "blouse shirt",
    "高領上衣": "turtleneck top",
    "針織衫": "knit top",
    "毛衣": "sweater",
    "T恤": "t-shirt",
    "衛衣": "hoodie sweatshirt",
    "背心": "tank top camisole",
    "Polo衫": "polo shirt",
    "露肩上衣": "off-shoulder top",
    "洋裝": "dress",
    "短裙": "mini skirt",
    "長裙": "long skirt",
    "牛仔褲": "denim jeans",
    "短褲": "shorts",
    "長褲": "long pants trousers",
    "西裝外套": "blazer",
    "針織外套": "knit cardigan",
    "牛仔外套": "denim jacket",
    "夾克": "jacket",
    "大衣": "coat",
}

COLLAR_EN = {
    "高領": "turtleneck high neck",
    "翻領": "collared lapel button shirt",
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
    "襯衫", "衬衫", "翻領", "翻领", "紐扣", "纽扣", "扣"
]

OFFICE_NEGATIVE_TERMS = [
    "性感", "辣妹", "露肩", "吊帶", "吊带",
    "海灘", "海滩", "度假", "派對", "派对"
]


# =========================
# 5. 查詢解析
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


def build_english_image_query(query_text: str, profile: dict) -> str:
    parts = []

    if profile.get("color"):
        parts.append(COLOR_EN.get(profile["color"], ""))

    if profile.get("collar"):
        parts.append(COLLAR_EN.get(profile["collar"], ""))

    if profile.get("sleeve"):
        parts.append(SLEEVE_EN.get(profile["sleeve"], ""))

    if profile.get("subcategory"):
        parts.append(SUBCATEGORY_EN.get(profile["subcategory"], ""))

    for s in profile.get("styles", []):
        parts.append(STYLE_EN.get(s, ""))

    for o in profile.get("occasions", []):
        parts.append(OCCASION_EN.get(o, ""))

    parts.append("fashion e-commerce clothing product photo")
    parts.append("women's clothing")

    parts = [p for p in parts if is_valid_value(p)]

    if not parts:
        return query_text

    return ", ".join(parts)


# =========================
# 6. Streamlit Cache：資料與 ChromaDB
# =========================

@st.cache_data(show_spinner=False)
def load_products() -> pd.DataFrame:
    if not PRODUCT_CSV_PATH.exists():
        raise FileNotFoundError(f"找不到資料檔：{PRODUCT_CSV_PATH}")

    df = pd.read_csv(PRODUCT_CSV_PATH)

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
        "english_caption",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"products_with_english_caption.csv 缺少欄位：{missing_cols}")

    df["id"] = df["id"].astype(str)
    df["product_name"] = df["product_name"].fillna("").astype(str)
    df["english_caption"] = df["english_caption"].fillna("").astype(str)

    return df


@st.cache_resource(show_spinner=False)
def get_image_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_function = OpenCLIPEmbeddingFunction()
    image_loader = ImageLoader()

    collection = client.get_collection(
        name=IMAGE_COLLECTION_NAME,
        embedding_function=embedding_function,
        data_loader=image_loader,
    )

    return collection


# =========================
# 7. 搜尋核心
# =========================

def image_search_scores(collection, english_query: str, profile: dict) -> dict:
    query_kwargs = {
        "query_texts": [english_query],
        "n_results": IMAGE_TOP_K,
        "include": ["metadatas", "distances"],
    }

    if profile["category"] is not None:
        query_kwargs["where"] = {"category": int(profile["category"])}

    try:
        results = collection.query(**query_kwargs)
    except Exception:
        query_kwargs.pop("where", None)
        results = collection.query(**query_kwargs)

    if not results["ids"] or len(results["ids"][0]) == 0:
        return {}

    ids = results["ids"][0]
    distances = np.array(results["distances"][0], dtype=float)

    d_min = distances.min()
    d_max = distances.max()

    if d_max > d_min:
        scores = 1.0 - ((distances - d_min) / (d_max - d_min))
    else:
        scores = np.ones_like(distances)

    output = {}

    for product_id, distance, score in zip(ids, distances, scores):
        output[str(product_id)] = {
            "image_distance": float(distance),
            "image_score": float(score),
        }

    return output


def has_category_match(row: pd.Series, profile: dict) -> bool:
    if profile["category"] is None:
        return True

    try:
        return int(row["category"]) == int(profile["category"])
    except Exception:
        return False


def has_subcategory_match(row: pd.Series, profile: dict, strict: bool = True) -> bool:
    query_sub = profile.get("subcategory", "")

    if not query_sub:
        return True

    name = normalize_text(row.get("product_name", ""))
    row_sub = str(row.get("subcategory", ""))
    components = split_pipe_text(row.get("product_components", ""))

    if query_sub == "襯衫":
        shirt_terms = SUBCATEGORY_RULES["襯衫"]

        if not contains_any(name, shirt_terms):
            return False

        if strict and contains_any(name, BAD_SHIRT_TERMS):
            return False

        return True

    if query_sub == "高領上衣":
        high_neck_terms = SUBCATEGORY_RULES["高領上衣"]
        return (
            row_sub == "高領上衣"
            or "高領" in components
            or contains_any(name, high_neck_terms)
        )

    terms = SUBCATEGORY_RULES.get(query_sub, [])

    return (
        row_sub == query_sub
        or query_sub in components
        or contains_any(name, terms)
    )


def has_color_match_or_unknown(row: pd.Series, profile: dict) -> bool:
    query_color = profile.get("color", "")

    if not query_color:
        return True

    name = normalize_text(row.get("product_name", ""))
    row_color = str(row.get("color", ""))

    if is_valid_value(row_color) and row_color != query_color:
        return False

    for color_label, terms in COLOR_RULES.items():
        if color_label == query_color:
            continue

        if contains_any(name, terms):
            return False

    return True


def has_collar_match_or_unknown(row: pd.Series, profile: dict) -> bool:
    query_collar = profile.get("collar", "")

    if not query_collar:
        return True

    name = normalize_text(row.get("product_name", ""))
    row_collar = str(row.get("collar", ""))

    if is_valid_value(row_collar) and row_collar != query_collar:
        return False

    terms = COLLAR_RULES.get(query_collar, [])

    if contains_any(name, terms):
        return True

    if not is_valid_value(row_collar):
        return True

    return False


def has_sleeve_match_or_unknown(row: pd.Series, profile: dict) -> bool:
    query_sleeve = profile.get("sleeve", "")

    if not query_sleeve:
        return True

    name = normalize_text(row.get("product_name", ""))
    row_sleeve = str(row.get("sleeve", ""))

    if is_valid_value(row_sleeve) and row_sleeve != query_sleeve:
        return False

    terms = SLEEVE_RULES.get(query_sleeve, [])

    if contains_any(name, terms):
        return True

    if not is_valid_value(row_sleeve):
        return True

    return False


def passes_title_filter(row: pd.Series, profile: dict, strict: bool = True) -> bool:
    if not has_category_match(row, profile):
        return False

    if not has_subcategory_match(row, profile, strict=strict):
        return False

    if not has_color_match_or_unknown(row, profile):
        return False

    if not has_collar_match_or_unknown(row, profile):
        return False

    if not has_sleeve_match_or_unknown(row, profile):
        return False

    return True


def get_candidates(df: pd.DataFrame, profile: dict, strict: bool = True) -> pd.DataFrame:
    mask = df.apply(lambda row: passes_title_filter(row, profile, strict=strict), axis=1)
    return df[mask].copy().reset_index(drop=True)


def raw_title_score(row: pd.Series, profile: dict) -> float:
    name = normalize_text(row.get("product_name", ""))
    caption = normalize_text(row.get("english_caption", ""))

    row_sub = str(row.get("subcategory", ""))
    row_color = str(row.get("color", ""))
    row_collar = str(row.get("collar", ""))
    row_sleeve = str(row.get("sleeve", ""))

    row_styles = split_pipe_text(row.get("style_keywords", ""))
    row_occasions = split_pipe_text(row.get("occasion_keywords", ""))

    score = 0.0

    if profile["category"] is not None:
        score += 1.0

    query_sub = profile.get("subcategory", "")
    if query_sub:
        terms = SUBCATEGORY_RULES.get(query_sub, [])
        if row_sub == query_sub:
            score += 4.0
        if contains_any(name, terms):
            score += 3.0

    query_color = profile.get("color", "")
    if query_color:
        terms = COLOR_RULES.get(query_color, [])
        if row_color == query_color:
            score += 2.0
        if contains_any(name, terms):
            score += 2.0

    query_collar = profile.get("collar", "")
    if query_collar:
        terms = COLLAR_RULES.get(query_collar, [])
        if row_collar == query_collar:
            score += 1.5
        if contains_any(name, terms):
            score += 1.5

    query_sleeve = profile.get("sleeve", "")
    if query_sleeve:
        terms = SLEEVE_RULES.get(query_sleeve, [])
        if row_sleeve == query_sleeve:
            score += 1.2
        if contains_any(name, terms):
            score += 1.2

    for style in profile.get("styles", []):
        terms = STYLE_RULES.get(style, [])
        if style in row_styles:
            score += 1.5
        if contains_any(name, terms) or contains_any(caption, terms):
            score += 1.0

    for occasion in profile.get("occasions", []):
        terms = OCCASION_RULES.get(occasion, [])
        if occasion in row_occasions:
            score += 1.5
        if contains_any(name, terms) or contains_any(caption, terms):
            score += 1.0

    is_office_query = (
        "通勤" in profile.get("styles", [])
        or "上班" in profile.get("occasions", [])
    )

    if is_office_query:
        if contains_any(name, OFFICE_POSITIVE_TERMS):
            score += 2.0

        if contains_any(name, OFFICE_NEGATIVE_TERMS):
            score -= 4.0

    return score


def normalize_scores(items: list[dict], score_key: str, output_key: str) -> list[dict]:
    if not items:
        return items

    scores = np.array([item[score_key] for item in items], dtype=float)

    s_min = scores.min()
    s_max = scores.max()

    if s_max > s_min:
        normalized = (scores - s_min) / (s_max - s_min)
    else:
        normalized = np.ones_like(scores)

    for item, score in zip(items, normalized):
        item[output_key] = float(score)

    return items


def get_dynamic_weights(profile: dict) -> dict:
    has_subcategory = bool(profile.get("subcategory"))
    has_visual_condition = bool(profile.get("color") or profile.get("collar") or profile.get("sleeve"))

    if has_subcategory and has_visual_condition:
        return {
            "title": 0.65,
            "image": 0.35,
        }

    if has_subcategory:
        return {
            "title": 0.80,
            "image": 0.20,
        }

    return {
        "title": 0.55,
        "image": 0.45,
    }


def deduplicate(items: list[dict], top_k: int = FINAL_TOP_K) -> list[dict]:
    deduped = []
    seen_names = set()

    for item in items:
        name_key = normalize_name_for_dedup(item["product_name"])

        if name_key in seen_names:
            continue

        seen_names.add(name_key)
        deduped.append(item)

        if len(deduped) >= top_k:
            break

    return deduped


def rank_candidates(candidates: pd.DataFrame, image_scores: dict, profile: dict) -> list[dict]:
    items = []

    for _, row in candidates.iterrows():
        product_id = str(row["id"])

        image_score = image_scores.get(product_id, {}).get("image_score", 0.0)
        title_raw_score = raw_title_score(row, profile)
        # 如果查詢明確指定顏色，圖片相似度太低的商品先排除。
        # 這可以避免商品標題或標籤寫白色，但圖片視覺上明顯不符合的商品排到前面。
        if profile.get("color") and image_score < 0.5:
            continue

        items.append({
            "id": product_id,
            "product_name": row["product_name"],
            "image_path": row["image_path"],
            "absolute_image_path": row["absolute_image_path"],
            "category_name": row["category_name"],
            "subcategory": row["subcategory"],
            "color": row["color"],
            "collar": row["collar"],
            "sleeve": row["sleeve"],
            "material": row["material"],
            "style_keywords": row["style_keywords"],
            "occasion_keywords": row["occasion_keywords"],
            "product_components": row["product_components"],
            "english_caption": row["english_caption"],
            "title_raw_score": title_raw_score,
            "title_score": 0.0,
            "image_score": image_score,
            "final_score": 0.0,
        })

    items = normalize_scores(items, score_key="title_raw_score", output_key="title_score")

    weights = get_dynamic_weights(profile)

    for item in items:
        item["final_score"] = (
            weights["title"] * item["title_score"]
            + weights["image"] * item["image_score"]
        )

    items = sorted(items, key=lambda x: x["final_score"], reverse=True)

    return deduplicate(items, top_k=FINAL_TOP_K)


def search_products(df: pd.DataFrame, collection, query_text: str) -> tuple[list[dict], dict]:
    profile = parse_query(query_text)
    english_query = build_english_image_query(query_text, profile)

    strict_candidates = get_candidates(df, profile, strict=True)

    used_relaxed = False
    candidates = strict_candidates

    if len(candidates) == 0:
        used_relaxed = True
        candidates = get_candidates(df, profile, strict=False)

    image_scores = image_search_scores(collection, english_query, profile)

    results = rank_candidates(candidates, image_scores, profile)

    diagnostics = {
        "profile": profile,
        "english_query": english_query,
        "strict_candidate_count": len(strict_candidates),
        "candidate_count": len(candidates),
        "used_relaxed": used_relaxed,
        "weights": get_dynamic_weights(profile),
    }

    return results, diagnostics


# =========================
# 8. 導購文字
# =========================

def build_recommendation_text(query: str, results: list[dict], diagnostics: dict) -> str:
    if not results:
        return "目前找不到符合條件的商品。建議改用較寬的查詢，例如「高領上衣」「黑色短裙」「度假洋裝」。"

    if diagnostics["strict_candidate_count"] < 5:
        prefix = (
            f"這個需求在目前資料中完全符合的商品偏少"
            f"（嚴格候選 {diagnostics['strict_candidate_count']} 筆），"
            "以下先提供最接近的款式："
        )
    else:
        prefix = "以下是依商品名稱條件與圖片相似度排序後，較符合需求的款式："

    names = []
    for item in results[:3]:
        sub = display_value(item.get("subcategory"))
        color = display_value(item.get("color"))
        names.append(f"{sub}（{color}）")

    return prefix + " " + "、".join(names) + "。"


# =========================
# 9. Streamlit UI
# =========================

def render_result_card(item: dict):
    image_path = Path(item["absolute_image_path"])

    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning("圖片不存在")

    st.markdown(f"**{item['product_name']}**")
    st.caption(f"ID：{item['id']}")

    tags = []
    for col in ["category_name", "subcategory", "color", "collar", "sleeve", "style_keywords", "occasion_keywords"]:
        value = display_value(item.get(col))
        if value != "—":
            tags.append(value)

    if tags:
        st.caption(" / ".join(tags[:6]))

    st.progress(min(max(float(item["final_score"]), 0.0), 1.0))
    st.caption(
        f"總分 {item['final_score']:.3f}｜"
        f"名稱 {item['title_score']:.2f}｜"
        f"圖片 {item['image_score']:.2f}"
    )


def main():
    st.set_page_config(
        page_title="Fashion RAG ",
        page_icon="👗",
        layout="wide",
    )

    st.title("多模態時尚商品搜尋 ")
    st.write("第一版以商品名稱與圖片相似度為主，適合展示白色高領上衣、黑色短裙、度假洋裝等資料量充足的查詢。")

    try:
        df = load_products()
        collection = get_image_collection()
    except Exception as e:
        st.error(f"初始化失敗：{e}")
        st.stop()

    with st.sidebar:
        st.header("資料狀態")
        st.write(f"商品數量：{len(df)}")
        st.write(f"圖片向量庫：{collection.count()}")

        st.divider()
        st.header("建議展示查詢")
        selected_demo = st.selectbox("選一個示範查詢", [""] + DEMO_QUERIES)

        st.divider()
        show_debug = st.checkbox("顯示除錯資訊", value=True)

    default_query = selected_demo if selected_demo else "白色高領上衣"

    query = st.text_input(
        "請輸入穿搭或商品需求",
        value=default_query,
        placeholder="例如：白色高領上衣、黑色短裙、海邊度假風洋裝",
    )

    search_clicked = st.button("搜尋", type="primary")

    if search_clicked or query:
        with st.spinner("搜尋中..."):
            results, diagnostics = search_products(df, collection, query)

        st.subheader("AI 導購建議")

        use_llm = st.checkbox("使用 Gemini 生成導購回答", value=True)

        if use_llm:
            with st.spinner("Gemini 正在生成導購建議..."):
                llm_text = generate_grounded_recommendation(
                    user_query=query,
                    retrieved_products=results[:6],
                    diagnostics=diagnostics,
                )
            st.info(llm_text)
        else:
            st.info(build_recommendation_text(query, results, diagnostics))

        if show_debug:
            with st.expander("查詢解析與候選數", expanded=False):
                st.json(diagnostics)

        st.subheader("推薦商品")

        if not results:
            st.warning("沒有找到結果。")
            return

        cols = st.columns(min(4, len(results)))

        for idx, item in enumerate(results):
            with cols[idx % len(cols)]:
                render_result_card(item)


if __name__ == "__main__":
    main()
