from pathlib import Path
import textwrap
import re

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

from PIL import Image
import matplotlib.pyplot as plt


# =========================
# 1. 路徑與 Collection 設定
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "fashion_products_v1"


# =========================
# 2. 查詢參數
# =========================
# RAW_TOP_K：
# 先從 ChromaDB 取回較多候選商品。
# 因為你的資料有分週重複商品，也因為我們後面要做 rerank，
# 所以不能只取 5 筆。
#
# FINAL_TOP_K：
# 最後實際顯示幾筆商品。

RAW_TOP_K = 80
FINAL_TOP_K = 5


# =========================
# 3. 連接 ChromaDB
# =========================

def get_collection():
    """
    連接已建立好的 ChromaDB collection。

    這裡要使用和建庫時相同的 OpenCLIPEmbeddingFunction。
    因為查詢文字也要被 OpenCLIP 轉成向量。
    """

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embedding_function = OpenCLIPEmbeddingFunction()
    image_loader = ImageLoader()

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        data_loader=image_loader,
    )

    return collection


# =========================
# 4. 查詢文字增強
# =========================

def expand_query_to_english(query_text: str) -> str:
    """
    第一版簡單查詢增強：
    把常見中文穿搭需求轉成 OpenCLIP 比較容易理解的英文描述。

    注意：
    這不是完整翻譯器，而是針對服飾檢索做關鍵詞轉換。
    """

    q = query_text.lower()

    color_map = {
        "白色": "white",
        "白": "white",
        "黑色": "black",
        "黑": "black",
        "灰色": "gray",
        "灰": "gray",
        "米色": "beige",
        "杏色": "cream beige",
        "奶油色": "cream",
        "粉色": "pink",
        "粉紅": "pink",
        "藍色": "blue",
        "蓝色": "blue",
        "紅色": "red",
        "红色": "red",
        "黃色": "yellow",
        "黄色": "yellow",
        "綠色": "green",
        "绿色": "green",
        "咖啡色": "brown",
        "棕色": "brown",
    }

    item_map = {
        "高領": "turtleneck",
        "高领": "turtleneck",
        "半高領": "mock neck",
        "半高领": "mock neck",
        "襯衫": "blouse shirt",
        "衬衫": "blouse shirt",
        "襯衣": "shirt",
        "衬衣": "shirt",
        "上衣": "top",
        "t恤": "t-shirt",
        "t-shirt": "t-shirt",
        "短版": "cropped",
        "長袖": "long sleeve",
        "长袖": "long sleeve",
        "短袖": "short sleeve",
        "針織": "knit",
        "针织": "knit",
        "毛衣": "sweater",
        "衛衣": "hoodie sweatshirt",
        "卫衣": "hoodie sweatshirt",
        "外套": "jacket coat",
        "夾克": "jacket",
        "夹克": "jacket",
        "大衣": "coat",
        "西裝外套": "blazer",
        "西装外套": "blazer",
        "短裙": "mini skirt",
        "長裙": "long skirt",
        "长裙": "long skirt",
        "洋裝": "dress",
        "连衣裙": "dress",
        "連衣裙": "dress",
        "褲": "pants",
        "裤": "pants",
        "牛仔": "denim",
        "polo": "polo shirt",
    }

    style_map = {
        "上班": "office workwear",
        "通勤": "office workwear",
        "辦公室": "office",
        "办公室": "office",
        "簡約": "minimal simple",
        "简约": "minimal simple",
        "氣質": "elegant",
        "气质": "elegant",
        "甜美": "sweet feminine",
        "約會": "date outfit",
        "约会": "date outfit",
        "海邊": "beach vacation",
        "海边": "beach vacation",
        "度假": "resort vacation",
        "街頭": "street style",
        "街头": "street style",
        "酷": "cool edgy",
        "拍照": "photogenic",
        "正式": "formal",
        "休閒": "casual",
        "休闲": "casual",
        "辣妹": "sexy trendy",
        "溫柔": "soft feminine",
        "温柔": "soft feminine",
    }

    english_terms = []

    for zh, en in color_map.items():
        if zh in q:
            english_terms.append(en)

    for zh, en in item_map.items():
        if zh in q:
            english_terms.append(en)

    for zh, en in style_map.items():
        if zh in q:
            english_terms.append(en)

    # 如果使用者本來就輸入英文，而且沒有命中中文詞，就直接使用原始查詢
    if len(english_terms) == 0:
        return query_text

    english_query = " ".join(english_terms)
    english_query += " fashion product photo, e-commerce clothing item"

    return english_query


# =========================
# 5. 推測商品大類
# =========================

def infer_category_filter(query_text: str):
    """
    根據查詢文字推測商品類別。

    目前你的類別是：
    1 上衣
    2 套裝
    3 褲子
    4 裙子
    5 外套
    6 內著
    """

    q = query_text.lower()

    # 上衣
    if any(word in q for word in [
        "上衣", "襯衫", "衬衫", "襯衣", "衬衣", "t恤", "t-shirt",
        "高領", "高领", "毛衣", "針織", "针织", "背心",
        "shirt", "blouse", "top", "turtleneck", "sweater", "knit"
    ]):
        return {"category": 1}

    # 套裝
    if any(word in q for word in [
        "套裝", "套装", "set", "two piece", "two-piece"
    ]):
        return {"category": 2}

    # 褲子
    if any(word in q for word in [
        "褲", "裤", "長褲", "长裤", "短褲", "短裤", "牛仔褲", "牛仔裤",
        "pants", "jeans", "trousers", "shorts"
    ]):
        return {"category": 3}

    # 裙子、洋裝
    if any(word in q for word in [
        "裙", "短裙", "長裙", "长裙", "洋裝", "洋装", "連衣裙", "连衣裙",
        "skirt", "dress"
    ]):
        return {"category": 4}

    # 外套
    if any(word in q for word in [
        "外套", "夾克", "夹克", "大衣", "西裝外套", "西装外套",
        "jacket", "coat", "blazer"
    ]):
        return {"category": 5}

    # 內著
    if any(word in q for word in [
        "內衣", "内衣", "內著", "内著", "bra", "underwear", "lingerie"
    ]):
        return {"category": 6}

    return None


# =========================
# 6. 商品名稱 rerank 規則
# =========================

def get_keyword_adjustment(query_text: str, product_name: str) -> float:
    """
    根據查詢文字與商品名稱做簡單加權。

    ChromaDB 的 distance 是越小越相似。
    所以：
    - 符合關鍵詞：score 減少，排名往前
    - 明顯不符合：score 增加，排名往後
    """

    q = query_text.lower()
    name = str(product_name).lower()

    adjustment = 0.0

    # =========================
    # 襯衫 / blouse / shirt
    # =========================
    if any(word in q for word in ["襯衫", "衬衫", "襯衣", "衬衣", "shirt", "blouse"]):
        positive_terms = [
            "襯衫", "衬衫", "襯衣", "衬衣",
            "shirt", "blouse",
            "翻領", "翻领", "排扣", "扣", "領口", "领口"
        ]
        negative_terms = [
            "毛衣", "針織", "针织", "衛衣", "卫衣",
            "t恤", "t 恤", "高領", "高领", "polo", "背心", "吊帶", "吊带"
        ]

        if any(term in name for term in positive_terms):
            adjustment -= 0.035

        if any(term in name for term in negative_terms):
            adjustment += 0.025

    # =========================
    # 高領 / turtleneck
    # =========================
    if any(word in q for word in ["高領", "高领", "turtleneck"]):
        positive_terms = [
            "高領", "高领", "半高領", "半高领",
            "堆堆領", "堆堆领", "turtleneck"
        ]
        negative_terms = [
            "v領", "v领", "方領", "方领", "圓領", "圆领",
            "露肩", "低領", "低领", "吊帶", "吊带"
        ]

        if any(term in name for term in positive_terms):
            adjustment -= 0.035

        if any(term in name for term in negative_terms):
            adjustment += 0.025

    # =========================
    # 白色 / white
    # =========================
    if any(word in q for word in ["白色", "white"]):
        positive_terms = [
            "白", "米白", "乳白", "奶白", "杏色", "cream", "white"
        ]
        negative_terms = [
            "黑", "深灰", "咖啡", "棕", "藍", "蓝", "紅", "红", "綠", "绿"
        ]

        if any(term in name for term in positive_terms):
            adjustment -= 0.015

        if any(term in name for term in negative_terms):
            adjustment += 0.015

    # =========================
    # 黑色 / black
    # =========================
    if any(word in q for word in ["黑色", "black"]):
        positive_terms = ["黑", "black"]
        negative_terms = ["白", "米白", "乳白", "奶白", "杏色", "粉", "紅", "红"]

        if any(term in name for term in positive_terms):
            adjustment -= 0.015

        if any(term in name for term in negative_terms):
            adjustment += 0.015

    # =========================
    # 上班 / office / workwear
    # =========================
    if any(word in q for word in ["上班", "通勤", "office", "workwear"]):
        positive_terms = [
            "通勤", "上班", "襯衫", "衬衫", "西裝", "西装",
            "簡約", "简约", "氣質", "气质", "翻領", "翻领"
        ]
        negative_terms = [
            "辣妹", "露肩", "性感", "吊帶", "吊带",
            "短版", "街頭", "街头", "甜酷"
        ]

        if any(term in name for term in positive_terms):
            adjustment -= 0.020

        if any(term in name for term in negative_terms):
            adjustment += 0.020

    # =========================
    # 簡約 / minimal
    # =========================
    if any(word in q for word in ["簡約", "简约", "minimal", "simple"]):
        positive_terms = ["簡約", "简约", "純色", "纯色", "素色", "百搭", "氣質", "气质"]
        negative_terms = ["印花", "圖案", "图案", "亮片", "露肩", "辣妹"]

        if any(term in name for term in positive_terms):
            adjustment -= 0.015

        if any(term in name for term in negative_terms):
            adjustment += 0.015

    return adjustment


# =========================
# 7. 搜尋結果去重與 rerank
# =========================

def normalize_product_name(name: str) -> str:
    """
    將商品名稱簡單標準化，用於去重。
    """

    name = str(name).strip().lower()
    name = re.sub(r"\s+", "", name)
    return name


def rerank_and_deduplicate_results(results, final_top_k: int, query_text: str):
    """
    先根據 ChromaDB distance 取得候選商品，
    再用商品名稱做簡單 rerank，
    最後依 product_name 去重。

    final_score 越小，代表越應該排前面。
    """

    if not results["ids"] or len(results["ids"][0]) == 0:
        return []

    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []

    for product_id, metadata, distance in zip(ids, metadatas, distances):
        product_name = metadata.get("product_name", "")

        keyword_adjustment = get_keyword_adjustment(query_text, product_name)
        final_score = distance + keyword_adjustment

        candidates.append({
            "id": product_id,
            "product_name": product_name,
            "image_path": metadata.get("image_path", ""),
            "absolute_image_path": metadata.get("absolute_image_path", ""),
            "category_name": metadata.get("category_name", ""),
            "week": metadata.get("week", ""),
            "rank": metadata.get("rank", ""),
            "distance": distance,
            "keyword_adjustment": keyword_adjustment,
            "final_score": final_score,
        })

    # final_score 越小越前面
    candidates = sorted(candidates, key=lambda x: x["final_score"])

    deduped = []
    seen_names = set()

    for item in candidates:
        normalized_name = normalize_product_name(item["product_name"])

        if normalized_name in seen_names:
            continue

        seen_names.add(normalized_name)
        deduped.append(item)

        if len(deduped) >= final_top_k:
            break

    return deduped


# =========================
# 8. 執行文字查詢
# =========================

def search_products(collection, query_text: str):
    """
    用文字查詢商品圖片。

    這版會做：
    1. 中文查詢轉英文關鍵詞
    2. 根據查詢推測商品類別並篩選
    3. 從 ChromaDB 取較多候選
    4. 根據商品名稱做 rerank
    5. 去重後回傳 Top-K
    """

    expanded_query = expand_query_to_english(query_text)
    category_filter = infer_category_filter(query_text)

    print(f"\n原始查詢：{query_text}")
    print(f"增強查詢：{expanded_query}")
    print(f"類別篩選：{category_filter}")

    query_kwargs = {
        "query_texts": [expanded_query],
        "n_results": RAW_TOP_K,
        "include": ["metadatas", "distances"],
    }

    if category_filter is not None:
        query_kwargs["where"] = category_filter

    results = collection.query(**query_kwargs)

    final_results = rerank_and_deduplicate_results(
        results=results,
        final_top_k=FINAL_TOP_K,
        query_text=query_text,
    )

    return final_results, expanded_query, category_filter


# =========================
# 9. 顯示文字結果
# =========================

def print_results(query_text: str, expanded_query: str, category_filter, results: list[dict]):
    print("\n" + "=" * 70)
    print(f"原始查詢：{query_text}")
    print(f"增強查詢：{expanded_query}")
    print(f"類別篩選：{category_filter}")
    print("=" * 70)

    if len(results) == 0:
        print("沒有找到結果。")
        return

    for i, item in enumerate(results, start=1):
        print(f"\n[{i}]")
        print(f"id: {item['id']}")
        print(f"product_name: {item['product_name']}")
        print(f"category: {item['category_name']}")
        print(f"week: {item['week']}")
        print(f"rank: {item['rank']}")
        print(f"image_path: {item['image_path']}")
        print(f"distance: {item['distance']:.4f}")
        print(f"keyword_adjustment: {item['keyword_adjustment']:.4f}")
        print(f"final_score: {item['final_score']:.4f}")


# =========================
# 10. 顯示圖片結果
# =========================

def show_images(query_text: str, expanded_query: str, results: list[dict]):
    """
    用 matplotlib 顯示搜尋結果圖片。

    distance 越小，通常代表 OpenCLIP 原始向量越相似。
    final_score 則是加入商品名稱 rerank 後的排序分數。
    """

    if len(results) == 0:
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
            f"{i}. {item['product_name']}\n"
            f"distance: {item['distance']:.4f}\n"
            f"final: {item['final_score']:.4f}"
        )
        title = "\n".join(textwrap.wrap(title, width=16))
        plt.title(title, fontsize=9)

    # 圖片視窗標題用 expanded_query，避免中文顯示成方塊時太難看
    plt.suptitle(f"Query: {expanded_query}", fontsize=14)
    plt.tight_layout()
    plt.show()


# =========================
# 11. 主程式
# =========================

def main():
    print("連接 ChromaDB...")
    print(f"ChromaDB path: {CHROMA_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")

    collection = get_collection()

    print(f"目前 collection 筆數：{collection.count()}")
    print("\n輸入穿搭需求後按 Enter。")
    print("輸入 q 離開。")
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

        results, expanded_query, category_filter = search_products(
            collection=collection,
            query_text=query_text,
        )

        print_results(
            query_text=query_text,
            expanded_query=expanded_query,
            category_filter=category_filter,
            results=results,
        )

        show_images(
            query_text=query_text,
            expanded_query=expanded_query,
            results=results,
        )


if __name__ == "__main__":
    main()