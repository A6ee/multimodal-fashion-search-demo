from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

from PIL import Image
import matplotlib.pyplot as plt


# =========================
# 1. 路徑設定
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

PRODUCT_CSV_PATH = DATA_DIR / "products_with_english_caption.csv"

IMAGE_COLLECTION_NAME = "fashion_products_v1"
TEXT_COLLECTION_NAME = "fashion_product_text_v1"


# =========================
# 2. 搜尋參數
# =========================

IMAGE_TOP_K = 120
TEXT_TOP_K = 120
ATTR_TOP_K = 200
FINAL_TOP_K = 5


# =========================
# 3. 基本工具
# =========================

UNKNOWN_VALUES = {"", "未知", "nan", "none", "null"}


def normalize_text(text) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("　", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_value(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() not in UNKNOWN_VALUES


def split_pipe_text(text) -> list[str]:
    if not is_valid_value(text):
        return []
    return [x.strip() for x in str(text).split("|") if is_valid_value(x)]


def normalize_name_for_dedup(name: str) -> str:
    name = normalize_text(name)
    name = re.sub(r"\s+", "", name)
    return name


# =========================
# 4. 查詢解析規則
# =========================

COLOR_RULES = [
    ("白色", ["白色", "米白", "乳白", "奶白", "象牙白", "本白", "珍珠白"]),
    ("黑色", ["黑色", "黑"]),
    ("灰色", ["灰色", "淺灰", "浅灰", "深灰", "灰"]),
    ("米色", ["米色", "杏色", "卡其", "奶油色", "燕麥色", "燕麦色"]),
    ("粉色", ["粉色", "粉紅", "粉红", "藕粉"]),
    ("藍色", ["藍色", "蓝色", "淺藍", "浅蓝", "天藍", "天蓝", "牛仔藍", "牛仔蓝"]),
    ("紅色", ["紅色", "红色", "酒紅", "酒红"]),
    ("黃色", ["黃色", "黄色", "鵝黃", "鹅黄", "奶黃", "奶黄"]),
    ("綠色", ["綠色", "绿色", "軍綠", "军绿", "墨綠", "墨绿"]),
    ("棕色", ["棕色", "咖啡", "褐色", "摩卡", "巧克力"]),
    ("紫色", ["紫色", "薰衣草", "香芋"]),
]

SUBCATEGORY_RULES = [
    ("襯衫", ["襯衫", "衬衫", "襯衣", "衬衣", "shirt", "blouse"]),
    ("高領上衣", ["高領", "高领", "半高領", "半高领", "堆堆領", "堆堆领", "turtleneck"]),
    ("針織衫", ["針織", "针织", "knit"]),
    ("毛衣", ["毛衣", "sweater"]),
    ("T恤", ["t恤", "t-shirt", "tee"]),
    ("衛衣", ["衛衣", "卫衣", "帽t", "hoodie", "sweatshirt"]),
    ("背心", ["背心", "吊帶", "吊带", "細肩帶", "细肩带", "tank", "camisole"]),
    ("Polo衫", ["polo"]),
    ("露肩上衣", ["露肩", "一字肩", "斜肩", "off shoulder"]),
    ("洋裝", ["洋裝", "洋装", "連衣裙", "连衣裙", "dress"]),
    ("短裙", ["短裙", "mini skirt"]),
    ("長裙", ["長裙", "长裙", "半身裙", "skirt"]),
    ("牛仔褲", ["牛仔褲", "牛仔裤", "jeans"]),
    ("短褲", ["短褲", "短裤", "shorts"]),
    ("長褲", ["長褲", "长裤", "西裝褲", "西装裤", "trousers", "pants"]),
    ("西裝外套", ["西裝外套", "西装外套", "西服外套", "blazer"]),
    ("針織外套", ["針織外套", "针织外套", "開衫", "开衫", "cardigan"]),
    ("牛仔外套", ["牛仔外套", "denim jacket"]),
    ("夾克", ["夾克", "夹克", "jacket"]),
    ("大衣", ["大衣", "風衣", "风衣", "coat", "trench"]),
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

STYLE_RULES = [
    ("簡約", ["簡約", "简约", "極簡", "极简", "純色", "纯色", "素色", "basic", "minimal", "simple"]),
    ("通勤", ["通勤", "上班", "職場", "职场", "ol", "office", "workwear"]),
    ("氣質", ["氣質", "气质", "優雅", "优雅", "elegant"]),
    ("甜美", ["甜美", "可愛", "可爱", "少女", "sweet"]),
    ("辣妹", ["辣妹", "性感", "修身", "緊身", "紧身", "sexy"]),
    ("休閒", ["休閒", "休闲", "日常", "casual"]),
    ("街頭", ["街頭", "街头", "酷", "帥氣", "帅气", "street"]),
    ("復古", ["復古", "复古", "vintage"]),
    ("韓系", ["韓系", "韩系", "korean"]),
    ("法式", ["法式", "french"]),
    ("度假", ["度假", "海邊", "海边", "沙灘", "沙滩", "resort", "beach"]),
]

OCCASION_RULES = [
    ("上班", ["上班", "通勤", "職場", "职场", "ol", "office", "workwear"]),
    ("日常", ["日常", "休閒", "休闲", "百搭", "casual"]),
    ("約會", ["約會", "约会", "date"]),
    ("度假", ["度假", "海邊", "海边", "沙灘", "沙滩", "resort", "beach"]),
    ("派對", ["派對", "派对", "party"]),
    ("正式場合", ["正式", "宴會", "宴会", "formal"]),
]


def match_first(text: str, rules: list[tuple[str, list[str]]], default="") -> str:
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


def infer_category_from_query(query_text: str):
    q = normalize_text(query_text)

    if any(w in q for w in [
        "上衣", "襯衫", "衬衫", "襯衣", "衬衣", "t恤", "t-shirt",
        "高領", "高领", "毛衣", "針織", "针织", "背心",
        "shirt", "blouse", "top", "turtleneck", "sweater", "knit"
    ]):
        return 1

    if any(w in q for w in ["套裝", "套装", "set", "two piece", "two-piece"]):
        return 2

    if any(w in q for w in [
        "褲", "裤", "長褲", "长裤", "短褲", "短裤",
        "牛仔褲", "牛仔裤", "pants", "jeans", "trousers", "shorts"
    ]):
        return 3

    if any(w in q for w in [
        "裙", "短裙", "長裙", "长裙", "洋裝", "洋装",
        "連衣裙", "连衣裙", "skirt", "dress"
    ]):
        return 4

    if any(w in q for w in [
        "外套", "夾克", "夹克", "大衣", "西裝外套", "西装外套",
        "jacket", "coat", "blazer"
    ]):
        return 5

    if any(w in q for w in ["內衣", "内衣", "內著", "内著", "bra", "underwear", "lingerie"]):
        return 6

    return None


def parse_query(query_text: str) -> dict:
    q = normalize_text(query_text)

    profile = {
        "raw_query": query_text,
        "category": infer_category_from_query(q),
        "subcategory": match_first(q, SUBCATEGORY_RULES, default=""),
        "color": match_first(q, COLOR_RULES, default=""),
        "collar": match_first(q, COLLAR_RULES, default=""),
        "sleeve": match_first(q, SLEEVE_RULES, default=""),
        "styles": match_all(q, STYLE_RULES),
        "occasions": match_all(q, OCCASION_RULES),
    }

    return profile


# =========================
# 5. 中文查詢 → 英文查詢
# =========================

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


def build_english_query(query_text: str, profile: dict) -> str:
    parts = []

    raw_q = normalize_text(query_text)

    if re.search(r"[a-zA-Z]", raw_q):
        parts.append(query_text)

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

    if len(parts) == 0:
        return query_text

    return ", ".join(parts)


# =========================
# 6. 搜尋引擎
# =========================

class MultivectorSearchEngine:
    def __init__(self):
        self.df = self.load_product_table()
        self.image_collection = self.get_image_collection()
        self.text_collection = self.get_text_collection()

    def load_product_table(self) -> pd.DataFrame:
        if not PRODUCT_CSV_PATH.exists():
            raise FileNotFoundError(
                f"找不到 {PRODUCT_CSV_PATH}，請先執行 scripts\\06_generate_english_caption.py"
            )

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
            "search_text",
            "english_caption",
            "english_search_text",
            "caption_quality",
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"products_with_english_caption.csv 缺少欄位：{missing_cols}")

        df["id"] = df["id"].astype(str)
        df["product_name"] = df["product_name"].fillna("").astype(str)
        df["search_text"] = df["search_text"].fillna("").astype(str)
        df["english_caption"] = df["english_caption"].fillna("").astype(str)
        df["product_components"] = df["product_components"].fillna("").astype(str)

        return df

    def get_image_collection(self):
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embedding_function = OpenCLIPEmbeddingFunction()
        image_loader = ImageLoader()

        collection = client.get_collection(
            name=IMAGE_COLLECTION_NAME,
            embedding_function=embedding_function,
            data_loader=image_loader,
        )
        return collection

    def get_text_collection(self):
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embedding_function = OpenCLIPEmbeddingFunction()

        collection = client.get_collection(
            name=TEXT_COLLECTION_NAME,
            embedding_function=embedding_function,
        )
        return collection

    def get_filtered_indices(self, profile: dict) -> np.ndarray:
        if profile["category"] is None:
            return self.df.index.to_numpy()

        mask = self.df["category"].astype(int) == int(profile["category"])
        indices = self.df[mask].index.to_numpy()

        if len(indices) == 0:
            return self.df.index.to_numpy()

        return indices

    def image_search(self, english_query: str, profile: dict) -> dict:
        query_kwargs = {
            "query_texts": [english_query],
            "n_results": IMAGE_TOP_K,
            "include": ["metadatas", "distances"],
        }

        if profile["category"] is not None:
            query_kwargs["where"] = {"category": int(profile["category"])}

        try:
            results = self.image_collection.query(**query_kwargs)
        except Exception:
            query_kwargs.pop("where", None)
            results = self.image_collection.query(**query_kwargs)

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

    def text_search(self, english_query: str, profile: dict) -> dict:
        query_kwargs = {
            "query_texts": [english_query],
            "n_results": TEXT_TOP_K,
            "include": ["metadatas", "distances"],
        }

        if profile["category"] is not None:
            query_kwargs["where"] = {"category": int(profile["category"])}

        try:
            results = self.text_collection.query(**query_kwargs)
        except Exception:
            query_kwargs.pop("where", None)
            results = self.text_collection.query(**query_kwargs)

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
                "text_distance": float(distance),
                "text_score": float(score),
            }

        return output

    def compute_attribute_score(self, row: pd.Series, profile: dict) -> float:
        matched_weight = 0.0
        total_weight = 0.0

        if profile["category"] is not None:
            total_weight += 1.0
            if int(row["category"]) == int(profile["category"]):
                matched_weight += 1.0

        if profile["subcategory"]:
            total_weight += 2.8
            if str(row.get("subcategory", "")) == profile["subcategory"]:
                matched_weight += 2.8
            else:
                row_components = split_pipe_text(row.get("product_components", ""))
                if profile["subcategory"] in row_components:
                    matched_weight += 1.0

        if profile["color"]:
            total_weight += 1.5
            if str(row.get("color", "")) == profile["color"]:
                matched_weight += 1.5

        if profile["collar"]:
            total_weight += 1.8
            if str(row.get("collar", "")) == profile["collar"]:
                matched_weight += 1.8

        if profile["sleeve"]:
            total_weight += 1.2
            if str(row.get("sleeve", "")) == profile["sleeve"]:
                matched_weight += 1.2

        row_styles = split_pipe_text(row.get("style_keywords", ""))
        for style in profile.get("styles", []):
            total_weight += 1.0
            if style in row_styles:
                matched_weight += 1.0

        row_occasions = split_pipe_text(row.get("occasion_keywords", ""))
        for occasion in profile.get("occasions", []):
            total_weight += 1.0
            if occasion in row_occasions:
                matched_weight += 1.0

        if total_weight == 0:
            return 0.0

        return matched_weight / total_weight

    def attribute_search(self, profile: dict) -> dict:
        filtered_indices = self.get_filtered_indices(profile)
        scored = []

        for df_idx in filtered_indices:
            row = self.df.loc[df_idx]
            score = self.compute_attribute_score(row, profile)

            if score > 0:
                scored.append((df_idx, score))

        scored = sorted(scored, key=lambda x: x[1], reverse=True)[:ATTR_TOP_K]

        output = {}
        for df_idx, score in scored:
            product_id = str(self.df.loc[df_idx, "id"])
            output[product_id] = {"attribute_score": float(score)}

        return output

    def get_dynamic_weights(self, profile: dict) -> dict:
        """
        根據查詢明確程度動態調整權重。

        這版比上一版提高圖片權重，因為目前結果顯示：
        text_score / attribute_score 太容易把圖片不合理的商品拉到前面。
        """

        explicit_count = 0

        for key in ["subcategory", "color", "collar", "sleeve"]:
            if profile.get(key):
                explicit_count += 1

        explicit_count += len(profile.get("styles", []))
        explicit_count += len(profile.get("occasions", []))

        has_visual_condition = any([
            profile.get("color"),
            profile.get("collar"),
            profile.get("sleeve"),
            profile.get("subcategory"),
        ])

        # 很具體且有視覺條件，例如：
        # 白色高領上衣、氣質通勤翻領長袖襯衫
        if has_visual_condition:
            return {
                "image": 0.30,
                "text": 0.35,
                "attribute": 0.35,
            }

        # 中度具體，但不一定是視覺條件
        if explicit_count >= 3:
            return {
                "image": 0.30,
                "text": 0.35,
                "attribute": 0.35,
            }

        # 偏風格、偏情境，例如：
        # 海邊度假風、甜美約會風
        return {
            "image": 0.50,
            "text": 0.25,
            "attribute": 0.25,
        }

    def passes_hard_constraints(
        self,
        row: pd.Series,
        profile: dict,
        image_score: float = 0.0,
        text_score: float = 0.0,
        strict: bool = True
    ) -> bool:
        """
        強條件過濾。

        目的：
        1. 使用者明確指定顏色時，不讓明顯錯色商品進入 Top-K。
        2. 使用者明確指定襯衫時，不讓 T恤、外套、一般上衣混進來。
        3. 使用者指定領口、袖長時，排除明顯衝突商品。
        4. 對於屬性未知的商品，不直接殺掉，但要求圖片或文字分數有一定支持。
        """

        if profile["category"] is not None:
            try:
                if int(row["category"]) != int(profile["category"]):
                    return False
            except Exception:
                return False

        row_subcategory = str(row.get("subcategory", ""))
        row_color = str(row.get("color", ""))
        row_collar = str(row.get("collar", ""))
        row_sleeve = str(row.get("sleeve", ""))
        row_components = split_pipe_text(row.get("product_components", ""))

        product_name = normalize_text(row.get("product_name", ""))

        query_subcategory = profile.get("subcategory", "")

        if query_subcategory:
            if query_subcategory == "襯衫":
                if row_subcategory != "襯衫":
                    return False

                # 查襯衫時，不能只靠文字屬性。
                # 圖片完全不支持的商品不要進來。
                if image_score < 0.05 and text_score < 0.30:
                    return False

            elif query_subcategory == "高領上衣":
                is_turtleneck_like = (
                    row_subcategory == "高領上衣"
                    or "高領" in row_components
                    or any(term in product_name for term in ["高領", "高领", "半高領", "半高领", "turtleneck"])
                )

                if not is_turtleneck_like:
                    return False

            else:
                if row_subcategory != query_subcategory and query_subcategory not in row_components:
                    return False

        query_color = profile.get("color", "")

        query_color = profile.get("color", "")

        if query_color:
            # 商品顏色已知，而且明確不同：直接排除
            if is_valid_value(row_color) and row_color != query_color:
                return False

            # 只要使用者指定顏色，就要求圖片也要有一定支持。
            # 因為商品名稱可能寫白色，但圖片實際看起來不是白色。
            if image_score < 0.25:
                return False

        query_collar = profile.get("collar", "")

        if query_collar:
            if is_valid_value(row_collar) and row_collar != query_collar:
                return False

            # 有明確領口條件時，圖片不能完全不支持
            if image_score < 0.15 and text_score < 0.25:
                return False

        query_sleeve = profile.get("sleeve", "")

        if query_sleeve:
            if is_valid_value(row_sleeve) and row_sleeve != query_sleeve:
                return False

            # 有明確袖長條件時，圖片不能完全不支持
            if image_score < 0.10 and text_score < 0.25:
                return False

        return True

    def mismatch_penalty(self, row: pd.Series, profile: dict) -> float:
        penalty = 0.0

        if profile["subcategory"]:
            row_sub = str(row.get("subcategory", ""))
            row_components = split_pipe_text(row.get("product_components", ""))

            if (
                row_sub not in ["", "未知"]
                and row_sub != profile["subcategory"]
                and profile["subcategory"] not in row_components
            ):
                penalty += 0.10

        if profile["color"]:
            row_color = str(row.get("color", ""))
            if row_color not in ["", "未知"] and row_color != profile["color"]:
                penalty += 0.05

        if profile["collar"]:
            row_collar = str(row.get("collar", ""))
            if row_collar not in ["", "未知"] and row_collar != profile["collar"]:
                penalty += 0.08

        if profile["sleeve"]:
            row_sleeve = str(row.get("sleeve", ""))
            if row_sleeve not in ["", "未知"] and row_sleeve != profile["sleeve"]:
                penalty += 0.05

        return penalty

    def merge_and_rank(self, image_result: dict, text_result: dict, attr_result: dict, profile: dict, strict: bool = True) -> list[dict]:
        candidate_ids = set()
        candidate_ids.update(image_result.keys())
        candidate_ids.update(text_result.keys())
        candidate_ids.update(attr_result.keys())

        weights = self.get_dynamic_weights(profile)
        ranked_items = []

        for product_id in candidate_ids:
            matched_rows = self.df[self.df["id"] == product_id]

            if matched_rows.empty:
                continue

            row = matched_rows.iloc[0]

            image_score = image_result.get(product_id, {}).get("image_score", 0.0)
            image_distance = image_result.get(product_id, {}).get("image_distance", None)

            text_score = text_result.get(product_id, {}).get("text_score", 0.0)
            text_distance = text_result.get(product_id, {}).get("text_distance", None)

            attribute_score = attr_result.get(product_id, {}).get("attribute_score", 0.0)

            if not self.passes_hard_constraints(
                row=row,
                profile=profile,
                image_score=image_score,
                text_score=text_score,
                strict=strict,
            ):
                continue

            penalty = self.mismatch_penalty(row, profile)

            final_score = (
                weights["image"] * image_score
                + weights["text"] * text_score
                + weights["attribute"] * attribute_score
                - penalty
            )

            ranked_items.append({
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
                "image_distance": image_distance,
                "image_score": image_score,
                "text_distance": text_distance,
                "text_score": text_score,
                "attribute_score": attribute_score,
                "penalty": penalty,
                "final_score": final_score,
            })

        ranked_items = sorted(ranked_items, key=lambda x: x["final_score"], reverse=True)
        return ranked_items

    def deduplicate(self, ranked_items: list[dict], top_k: int = FINAL_TOP_K) -> list[dict]:
        deduped = []
        seen_names = set()

        for item in ranked_items:
            name_key = normalize_name_for_dedup(item["product_name"])

            if name_key in seen_names:
                continue

            seen_names.add(name_key)
            deduped.append(item)

            if len(deduped) >= top_k:
                break

        return deduped

    def search(self, query_text: str):
        profile = parse_query(query_text)
        english_query = build_english_query(query_text, profile)

        print("\n查詢解析")
        print("========")
        print(f"原始查詢：{query_text}")
        print(f"英文查詢：{english_query}")
        print(f"解析結果：{profile}")
        print(f"動態權重：{self.get_dynamic_weights(profile)}")

        image_result = self.image_search(english_query, profile)
        text_result = self.text_search(english_query, profile)
        attr_result = self.attribute_search(profile)

        print("\n候選商品來源")
        print("============")
        print(f"圖片候選數：{len(image_result)}")
        print(f"文字候選數：{len(text_result)}")
        print(f"屬性候選數：{len(attr_result)}")

        strict_ranked = self.merge_and_rank(
            image_result=image_result,
            text_result=text_result,
            attr_result=attr_result,
            profile=profile,
            strict=True,
        )

        strict_results = self.deduplicate(strict_ranked, top_k=FINAL_TOP_K)

        return strict_results, profile, english_query


# =========================
# 7. 顯示結果
# =========================

def print_results(query_text: str, results: list[dict]):
    print("\n" + "=" * 100)
    print(f"查詢：{query_text}")
    print("=" * 100)

    if not results:
        print("沒有找到符合強條件的結果。")
        return

    for i, item in enumerate(results, start=1):
        print(f"\n[{i}]")
        print(f"id: {item['id']}")
        print(f"product_name: {item['product_name']}")
        print(f"category_name: {item['category_name']}")
        print(f"subcategory: {item['subcategory']}")
        print(f"color: {item['color']}")
        print(f"collar: {item['collar']}")
        print(f"sleeve: {item['sleeve']}")
        print(f"material: {item['material']}")
        print(f"style_keywords: {item['style_keywords']}")
        print(f"occasion_keywords: {item['occasion_keywords']}")
        print(f"product_components: {item['product_components']}")
        print(f"english_caption: {item['english_caption']}")
        print(f"image_score: {item['image_score']:.4f}")
        print(f"text_score: {item['text_score']:.4f}")
        print(f"attribute_score: {item['attribute_score']:.4f}")
        print(f"penalty: {item['penalty']:.4f}")
        print(f"final_score: {item['final_score']:.4f}")
        print(f"image_path: {item['image_path']}")


def show_images(query_text: str, results: list[dict]):
    if not results:
        return

    cols = len(results)
    plt.figure(figsize=(4 * cols, 5))

    for i, item in enumerate(results, start=1):
        image_path = Path(item["absolute_image_path"])

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"圖片無法開啟：{image_path}，錯誤：{e}")
            continue

        plt.subplot(1, cols, i)
        plt.imshow(img)
        plt.axis("off")

        title = (
            f"{i}. {item['id']}\n"
            f"final: {item['final_score']:.3f}\n"
            f"img: {item['image_score']:.2f}  "
            f"txt: {item['text_score']:.2f}\n"
            f"attr: {item['attribute_score']:.2f}"
        )

        title = "\n".join(textwrap.wrap(title, width=20))
        plt.title(title, fontsize=9)

    plt.suptitle(f"Query: {query_text}", fontsize=16)
    plt.tight_layout()
    plt.show()


# =========================
# 8. 主程式
# =========================

def main():
    print("啟動 Multivector Hybrid Search Demo")
    print("===================================")
    print(f"商品資料：{PRODUCT_CSV_PATH}")
    print(f"ChromaDB path：{CHROMA_DIR}")
    print(f"Image collection：{IMAGE_COLLECTION_NAME}")
    print(f"Text collection：{TEXT_COLLECTION_NAME}")

    engine = MultivectorSearchEngine()

    print(f"\n商品筆數：{len(engine.df)}")
    print(f"Image collection 筆數：{engine.image_collection.count()}")
    print(f"Text collection 筆數：{engine.text_collection.count()}")

    print("\n輸入 q 離開。")
    print("\n建議測試：")
    print("- 白色高領上衣")
    print("- 氣質通勤翻領長袖襯衫")
    print("- 黑色短裙")
    print("- 海邊度假風洋裝")
    print("- 酷一點的街頭風外套")
    print("- 適合上班的簡約襯衫")

    while True:
        query_text = input("\n請輸入穿搭需求：").strip()

        if query_text.lower() in ["q", "quit", "exit"]:
            print("結束查詢。")
            break

        if query_text == "":
            print("請輸入查詢文字。")
            continue

        results, profile, english_query = engine.search(query_text)

        print_results(query_text, results)
        show_images(query_text, results)


if __name__ == "__main__":
    main()