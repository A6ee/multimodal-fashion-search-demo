from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

from sklearn.feature_extraction.text import TfidfVectorizer

from PIL import Image
import matplotlib.pyplot as plt


# =========================
# 1. 路徑設定
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

PRODUCT_ATTR_CSV_PATH = DATA_DIR / "products_with_attributes.csv"

COLLECTION_NAME = "fashion_products_v1"


# =========================
# 2. 搜尋參數
# =========================

IMAGE_TOP_K = 120
TEXT_TOP_K = 180
ATTR_TOP_K = 180
FINAL_TOP_K = 5


# =========================
# 3. 基本工具
# =========================

def normalize_text(text) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("　", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_name_for_dedup(name: str) -> str:
    name = normalize_text(name)
    name = re.sub(r"\s+", "", name)
    return name


def split_pipe_text(text) -> list[str]:
    if pd.isna(text) or str(text).strip() == "":
        return []

    return [x.strip() for x in str(text).split("|") if x.strip() != ""]


def append_unique(items: list[str], value: str):
    if value and value not in items:
        items.append(value)


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

    if any(w in q for w in ["褲", "裤", "長褲", "长裤", "短褲", "短裤", "牛仔褲", "牛仔裤", "pants", "jeans", "trousers", "shorts"]):
        return 3

    if any(w in q for w in ["裙", "短裙", "長裙", "长裙", "洋裝", "洋装", "連衣裙", "连衣裙", "skirt", "dress"]):
        return 4

    if any(w in q for w in ["外套", "夾克", "夹克", "大衣", "西裝外套", "西装外套", "jacket", "coat", "blazer"]):
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
# 5. OpenCLIP 英文查詢增強
# =========================

def expand_query_to_english(query_text: str) -> str:
    q = normalize_text(query_text)
    terms = []

    zh_to_en = [
        ("白色", "white"),
        ("黑色", "black"),
        ("灰色", "gray"),
        ("米色", "beige"),
        ("杏色", "cream beige"),
        ("粉色", "pink"),
        ("藍色", "blue"),
        ("蓝色", "blue"),
        ("紅色", "red"),
        ("红色", "red"),
        ("黃色", "yellow"),
        ("黄色", "yellow"),
        ("綠色", "green"),
        ("绿色", "green"),
        ("棕色", "brown"),
        ("咖啡色", "brown"),

        ("高領", "turtleneck"),
        ("高领", "turtleneck"),
        ("半高領", "mock neck"),
        ("半高领", "mock neck"),
        ("襯衫", "blouse shirt"),
        ("衬衫", "blouse shirt"),
        ("上衣", "top"),
        ("長袖", "long sleeve"),
        ("长袖", "long sleeve"),
        ("短袖", "short sleeve"),
        ("針織", "knit"),
        ("针织", "knit"),
        ("毛衣", "sweater"),
        ("外套", "jacket coat"),
        ("短裙", "mini skirt"),
        ("長裙", "long skirt"),
        ("长裙", "long skirt"),
        ("洋裝", "dress"),
        ("洋装", "dress"),
        ("褲", "pants"),
        ("裤", "pants"),
        ("牛仔", "denim"),

        ("上班", "office workwear"),
        ("通勤", "office workwear"),
        ("簡約", "minimal simple"),
        ("简约", "minimal simple"),
        ("氣質", "elegant"),
        ("气质", "elegant"),
        ("甜美", "sweet feminine"),
        ("約會", "date outfit"),
        ("约会", "date outfit"),
        ("海邊", "beach vacation"),
        ("海边", "beach vacation"),
        ("度假", "resort vacation"),
        ("街頭", "street style"),
        ("街头", "street style"),
        ("酷", "cool edgy"),
        ("正式", "formal"),
        ("休閒", "casual"),
        ("休闲", "casual"),
    ]

    for zh, en in zh_to_en:
        if zh in q:
            append_unique(terms, en)

    if len(terms) == 0:
        return query_text

    return " ".join(terms) + " fashion product photo, e-commerce clothing item"


# =========================
# 6. Hybrid Search Engine
# =========================

class HybridSearchEngine:
    def __init__(self):
        self.df = self.load_product_table()
        self.collection = self.get_collection()
        self.vectorizer, self.tfidf_matrix = self.build_text_index()

    def load_product_table(self) -> pd.DataFrame:
        if not PRODUCT_ATTR_CSV_PATH.exists():
            raise FileNotFoundError(
                f"找不到 {PRODUCT_ATTR_CSV_PATH}，請先執行 scripts\\04_prepare_product_attributes.py"
            )

        df = pd.read_csv(PRODUCT_ATTR_CSV_PATH)

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
            "search_text",
        ]

        # products_with_attributes.csv 原本不一定有 absolute_image_path，這裡補上
        if "absolute_image_path" not in df.columns:
            df["absolute_image_path"] = df["image_path"].apply(
                lambda p: str(PROJECT_ROOT / str(p))
            )

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"products_with_attributes.csv 缺少欄位：{missing_cols}")

        df["id"] = df["id"].astype(str)
        df["search_text"] = df["search_text"].fillna("").astype(str)
        df["product_name"] = df["product_name"].fillna("").astype(str)

        return df

    def get_collection(self):
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        embedding_function = OpenCLIPEmbeddingFunction()
        image_loader = ImageLoader()

        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            data_loader=image_loader,
        )

        return collection

    def build_text_index(self):
        """
        使用 TF-IDF 建立商品名稱與屬性的文字搜尋索引。

        analyzer='char'：
        對中文比較友善，因為中文沒有空白分詞。
        """
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=1,
            max_df=0.95,
            norm="l2",
        )

        tfidf_matrix = vectorizer.fit_transform(self.df["search_text"])

        return vectorizer, tfidf_matrix

    def get_filtered_indices(self, profile: dict) -> np.ndarray:
        """
        如果查詢能推測出大類，就先限制在該 category。
        """
        if profile["category"] is None:
            return self.df.index.to_numpy()

        mask = self.df["category"].astype(int) == int(profile["category"])
        indices = self.df[mask].index.to_numpy()

        if len(indices) == 0:
            return self.df.index.to_numpy()

        return indices

    def image_search(self, expanded_query: str, profile: dict) -> dict:
        query_kwargs = {
            "query_texts": [expanded_query],
            "n_results": IMAGE_TOP_K,
            "include": ["metadatas", "distances"],
        }

        if profile["category"] is not None:
            query_kwargs["where"] = {"category": int(profile["category"])}

        results = self.collection.query(**query_kwargs)

        if not results["ids"] or len(results["ids"][0]) == 0:
            return {}

        ids = results["ids"][0]
        distances = np.array(results["distances"][0], dtype=float)

        d_min = distances.min()
        d_max = distances.max()

        if d_max > d_min:
            image_scores = 1.0 - ((distances - d_min) / (d_max - d_min))
        else:
            image_scores = np.ones_like(distances)

        image_result = {}

        for product_id, distance, image_score in zip(ids, distances, image_scores):
            image_result[str(product_id)] = {
                "image_distance": float(distance),
                "image_score": float(image_score),
            }

        return image_result

    def build_text_query(self, query_text: str, profile: dict) -> str:
        parts = [query_text]

        for key in ["subcategory", "color", "collar", "sleeve"]:
            if profile.get(key):
                parts.append(profile[key])

        parts.extend(profile.get("styles", []))
        parts.extend(profile.get("occasions", []))

        return " ".join(parts)

    def text_search(self, text_query: str, profile: dict) -> dict:
        query_vec = self.vectorizer.transform([text_query])
        filtered_indices = self.get_filtered_indices(profile)

        if len(filtered_indices) == 0:
            return {}

        sub_matrix = self.tfidf_matrix[filtered_indices]
        sims = (sub_matrix @ query_vec.T).toarray().ravel()

        if np.all(sims == 0):
            return {}

        top_local_indices = np.argsort(-sims)[:TEXT_TOP_K]

        text_result = {}

        for local_idx in top_local_indices:
            score = float(sims[local_idx])

            if score <= 0:
                continue

            df_idx = filtered_indices[local_idx]
            product_id = str(self.df.loc[df_idx, "id"])

            text_result[product_id] = {
                "text_score": score,
            }

        return text_result

    def compute_attribute_score(self, row: pd.Series, profile: dict) -> float:
        """
        根據 query profile 和商品屬性算匹配分數。
        分數範圍大致為 0~1。
        """
        matched_weight = 0.0
        total_weight = 0.0

        # 大類
        if profile["category"] is not None:
            total_weight += 1.0
            if int(row["category"]) == int(profile["category"]):
                matched_weight += 1.0

        # 細分類，例如 襯衫、高領上衣
        if profile["subcategory"]:
            total_weight += 2.5
            if str(row.get("subcategory", "")) == profile["subcategory"]:
                matched_weight += 2.5

        # 顏色
        if profile["color"]:
            total_weight += 1.5
            if str(row.get("color", "")) == profile["color"]:
                matched_weight += 1.5

        # 領口
        if profile["collar"]:
            total_weight += 1.8
            if str(row.get("collar", "")) == profile["collar"]:
                matched_weight += 1.8

        # 袖長
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

        attr_scores = []

        for df_idx in filtered_indices:
            row = self.df.loc[df_idx]
            score = self.compute_attribute_score(row, profile)

            if score > 0:
                attr_scores.append((df_idx, score))

        attr_scores = sorted(attr_scores, key=lambda x: x[1], reverse=True)[:ATTR_TOP_K]

        attr_result = {}

        for df_idx, score in attr_scores:
            product_id = str(self.df.loc[df_idx, "id"])
            attr_result[product_id] = {
                "attribute_score": float(score),
            }

        return attr_result

    def get_dynamic_weights(self, profile: dict) -> dict:
        """
        查詢越具體，文字與屬性權重越高；
        查詢越抽象，圖片權重稍微提高。
        """
        specific_count = 0

        for key in ["subcategory", "color", "collar", "sleeve"]:
            if profile.get(key):
                specific_count += 1

        specific_count += len(profile.get("styles", []))
        specific_count += len(profile.get("occasions", []))

        if specific_count >= 3:
            return {
                "image": 0.25,
                "text": 0.45,
                "attribute": 0.30,
            }

        return {
            "image": 0.40,
            "text": 0.40,
            "attribute": 0.20,
        }

    def mismatch_penalty(self, row: pd.Series, profile: dict) -> float:
        """
        對明顯不符合使用者具體條件的商品扣分。
        """
        penalty = 0.0

        if profile["subcategory"]:
            row_sub = str(row.get("subcategory", ""))
            if row_sub not in ["", "未知"] and row_sub != profile["subcategory"]:
                penalty += 0.10

        if profile["color"]:
            row_color = str(row.get("color", ""))
            if row_color not in ["", "未知"] and row_color != profile["color"]:
                penalty += 0.06

        if profile["collar"]:
            row_collar = str(row.get("collar", ""))
            if row_collar not in ["", "未知"] and row_collar != profile["collar"]:
                penalty += 0.08

        if profile["sleeve"]:
            row_sleeve = str(row.get("sleeve", ""))
            if row_sleeve not in ["", "未知"] and row_sleeve != profile["sleeve"]:
                penalty += 0.05

        return penalty

    def merge_and_rank(self, image_result: dict, text_result: dict, attr_result: dict, profile: dict) -> list[dict]:
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
            attribute_score = attr_result.get(product_id, {}).get("attribute_score", 0.0)

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
                "image_distance": image_distance,
                "image_score": image_score,
                "text_score": text_score,
                "attribute_score": attribute_score,
                "penalty": penalty,
                "final_score": final_score,
            })

        ranked_items = sorted(ranked_items, key=lambda x: x["final_score"], reverse=True)

        return self.deduplicate(ranked_items)

    def deduplicate(self, ranked_items: list[dict]) -> list[dict]:
        deduped = []
        seen_names = set()

        for item in ranked_items:
            name_key = normalize_name_for_dedup(item["product_name"])

            if name_key in seen_names:
                continue

            seen_names.add(name_key)
            deduped.append(item)

            if len(deduped) >= FINAL_TOP_K:
                break

        return deduped

    def search(self, query_text: str):
        profile = parse_query(query_text)
        expanded_query = expand_query_to_english(query_text)
        text_query = self.build_text_query(query_text, profile)

        print("\n查詢解析")
        print("========")
        print(f"原始查詢：{query_text}")
        print(f"OpenCLIP 查詢：{expanded_query}")
        print(f"文字搜尋查詢：{text_query}")
        print(f"解析結果：{profile}")
        print(f"動態權重：{self.get_dynamic_weights(profile)}")

        image_result = self.image_search(expanded_query, profile)
        text_result = self.text_search(text_query, profile)
        attr_result = self.attribute_search(profile)

        print("\n候選商品來源")
        print("============")
        print(f"圖片候選數：{len(image_result)}")
        print(f"文字候選數：{len(text_result)}")
        print(f"屬性候選數：{len(attr_result)}")

        final_results = self.merge_and_rank(
            image_result=image_result,
            text_result=text_result,
            attr_result=attr_result,
            profile=profile,
        )

        return final_results, profile, expanded_query, text_query


# =========================
# 7. 顯示結果
# =========================

def print_results(query_text: str, results: list[dict]):
    print("\n" + "=" * 80)
    print(f"查詢：{query_text}")
    print("=" * 80)

    if not results:
        print("沒有找到結果。")
        return

    for i, item in enumerate(results, start=1):
        print(f"\n[{i}]")
        print(f"id: {item['id']}")
        print(f"product_name: {item['product_name']}")
        print(f"category: {item['category_name']}")
        print(f"subcategory: {item['subcategory']}")
        print(f"color: {item['color']}")
        print(f"collar: {item['collar']}")
        print(f"sleeve: {item['sleeve']}")
        print(f"material: {item['material']}")
        print(f"style_keywords: {item['style_keywords']}")
        print(f"occasion_keywords: {item['occasion_keywords']}")
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
            f"img: {item['image_score']:.2f} "
            f"txt: {item['text_score']:.2f} "
            f"attr: {item['attribute_score']:.2f}"
        )

        title = "\n".join(textwrap.wrap(title, width=18))
        plt.title(title, fontsize=9)

    plt.suptitle(f"Query: {query_text}", fontsize=14)
    plt.tight_layout()
    plt.show()


# =========================
# 8. 主程式
# =========================

def main():
    print("啟動 Hybrid Search Demo")
    print("=======================")
    print(f"商品屬性資料：{PRODUCT_ATTR_CSV_PATH}")
    print(f"ChromaDB path：{CHROMA_DIR}")
    print(f"Collection name：{COLLECTION_NAME}")

    engine = HybridSearchEngine()

    print(f"\n商品筆數：{len(engine.df)}")
    print(f"ChromaDB collection 筆數：{engine.collection.count()}")
    print("\n輸入 q 離開。")
    print("\n建議測試：")
    print("- 白色高領上衣")
    print("- 適合上班的簡約襯衫")
    print("- 黑色短裙")
    print("- 海邊度假風洋裝")
    print("- 酷一點的街頭風外套")

    while True:
        query_text = input("\n請輸入穿搭需求：").strip()

        if query_text.lower() in ["q", "quit", "exit"]:
            print("結束查詢。")
            break

        if query_text == "":
            print("請輸入查詢文字。")
            continue

        results, profile, expanded_query, text_query = engine.search(query_text)

        print_results(query_text, results)
        show_images(query_text, results)


if __name__ == "__main__":
    main()