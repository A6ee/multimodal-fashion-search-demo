import os
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


def _load_env() -> None:
    """
    Load environment variables from .env if python-dotenv is installed.
    """
    if load_dotenv is not None:
        load_dotenv()


def _clean_value(value: Any) -> str:
    """
    Clean values before sending them to Gemini.
    Avoid sending unknown / NaN-like values.
    """
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in ["", "nan", "none", "null", "未知", "—"]:
        return ""

    return value


def _format_float(value: Any, digits: int = 3) -> str:
    """
    Format numeric scores in a readable way.
    """
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return _clean_value(value)


def _format_product_for_prompt(item: Dict[str, Any], index: int) -> str:
    """
    Convert one retrieved product result into a compact text block for Gemini.
    Only include fields available from the retrieval result.

    Important:
    Gemini v0.2 does not directly inspect images.
    Image information is represented through image_score from OpenCLIP retrieval.
    """

    fields = {
        "product_name": _clean_value(item.get("product_name")),
        "category_name": _clean_value(item.get("category_name")),
        "subcategory": _clean_value(item.get("subcategory")),
        "color": _clean_value(item.get("color")),
        "collar": _clean_value(item.get("collar")),
        "sleeve": _clean_value(item.get("sleeve")),
        "material": _clean_value(item.get("material")),
        "style_keywords": _clean_value(item.get("style_keywords")),
        "occasion_keywords": _clean_value(item.get("occasion_keywords")),
        "product_components": _clean_value(item.get("product_components")),
        "final_score": _format_float(item.get("final_score")),
        "title_score": _format_float(item.get("title_score")),
        "image_score": _format_float(item.get("image_score")),
    }

    lines = [f"Product {index}:"]

    for key, value in fields.items():
        if value != "":
            lines.append(f"- {key}: {value}")

    return "\n".join(lines)


def build_recommendation_prompt(
    user_query: str,
    retrieved_products: List[Dict[str, Any]],
    diagnostics: Dict[str, Any] | None = None,
) -> str:
    """
    Build a grounded prompt for Gemini.

    Design:
    - The retrieval system finds candidate products.
    - Gemini only summarizes and explains the retrieved products.
    - Gemini must not invent unsupported product facts.
    - Gemini v0.2 does not directly inspect product images.
    """

    product_blocks = []

    for i, item in enumerate(retrieved_products, start=1):
        product_blocks.append(_format_product_for_prompt(item, i))

    products_text = "\n\n".join(product_blocks) if product_blocks else "No retrieved products."

    diagnostics = diagnostics or {}
    strict_candidate_count = diagnostics.get("strict_candidate_count", "")
    candidate_count = diagnostics.get("candidate_count", "")
    used_relaxed = diagnostics.get("used_relaxed", "")
    weights = diagnostics.get("weights", "")

    prompt = f"""
你是一位謹慎的時尚電商導購助理。請根據使用者查詢與系統檢索出的商品資料，生成自然、簡潔、可信的中文導購建議。

重要規則：
1. 你只能根據「已提供的商品名稱、系統標籤與檢索分數」回答，不可以編造沒有提供的資訊。
2. 頁面會顯示商品圖片，但你目前沒有直接判讀圖片內容；因此不要說「圖片看起來」、「視覺上一定是」、「實際穿起來」。
3. 商品圖片已經透過 OpenCLIP 圖片向量參與檢索與排序，因此可以提到「圖片相似度分數較高」，但不要自行描述圖片內容。
4. 描述商品時，請使用謹慎語氣，例如「商品名稱顯示」、「系統標籤顯示」、「較接近使用者需求」。
5. 不可以自行假設商品價格、品牌、庫存、實際尺寸、實際材質或實際穿著效果。
6. 如果某些商品只是相近款，而不是完全符合需求，要誠實說明。
7. 請根據檢索結果推薦 4 到 6 件商品，可以分成「最優先推薦」與「可作為替代款」。
8. 推薦時請標明 Product 1、Product 2、Product 3 等，方便對應畫面上的商品順序。
9. 如果某件商品有明顯差異，例如無袖、背心、半高領、非長袖、非指定顏色，請用「如果可以接受……」的方式說明。
10. 回答語氣要像自然的電商導購，不要太像機器人。
11. 請控制在 3 到 5 段內，不要過度條列。
12. 最後請補一句提醒：本段 AI 導購主要根據商品名稱、系統標籤與檢索分數生成；實際顏色與細節仍建議以頁面商品圖為準。

使用者查詢：
{user_query}

檢索診斷：
- strict_candidate_count: {strict_candidate_count}
- candidate_count: {candidate_count}
- used_relaxed: {used_relaxed}
- ranking_weights: {weights}

檢索商品：
{products_text}

請輸出：
- 先用一句話總結整體搜尋結果
- 說明 Product 1 到 Product 6 中哪些最符合需求
- 若某些商品只是相近款，請明確說明它適合在哪種情境下考慮
- 最後補一句提醒：實際顏色與細節仍建議以頁面商品圖為準
""".strip()

    return prompt


def generate_grounded_recommendation(
    user_query: str,
    retrieved_products: List[Dict[str, Any]],
    diagnostics: Dict[str, Any] | None = None,
) -> str:
    """
    Generate grounded shopping recommendation using Gemini API.

    Environment variables:
    - GEMINI_API_KEY: required
    - GEMINI_MODEL: optional, default = gemini-1.5-flash

    If API key or SDK is missing, return a clear fallback message instead of crashing.
    """

    _load_env()

    if genai is None:
        return (
            "尚未安裝 Gemini Python SDK。請先執行："
            "python -m pip install google-generativeai python-dotenv"
        )

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

    if not api_key:
        return (
            "尚未設定 GEMINI_API_KEY。請在專案根目錄建立 .env，"
            "並加入 GEMINI_API_KEY=你的_API_KEY。"
        )

    if not retrieved_products:
        return "目前沒有可供推薦的商品結果。建議改用較寬的查詢，例如「高領上衣」、「黑色短裙」或「度假洋裝」。"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        prompt = build_recommendation_prompt(
            user_query=user_query,
            retrieved_products=retrieved_products,
            diagnostics=diagnostics,
        )

        response = model.generate_content(prompt)

        text = getattr(response, "text", "")

        if not text:
            return "Gemini 沒有回傳有效文字。請稍後再試，或檢查 API 設定。"

        return text.strip()

    except Exception as e:
        return f"Gemini 生成失敗：{e}"
