# 多模態時尚商品搜尋與導購 Demo

## 中文版

## 專案簡介

本專案是一個多模態時尚商品搜尋與導購 Demo，使用 Python、OpenCLIP、ChromaDB、Streamlit 與 Gemini API 建置。

系統以快時尚電商商品資料為基礎，原始資料主要包含商品 ID、商品名稱與商品圖片路徑。由於資料屬於弱標註資料，缺乏完整商品描述與人工標籤，因此本專案採用 **Title-first Hybrid Retrieval** 架構：先根據商品名稱與分類篩選候選商品，再透過 OpenCLIP 圖片向量進行相似度 reranking，最後將檢索結果傳入 Gemini API，生成有依據的導購式推薦回答。

目前 v0.2 版本中，Gemini 不直接判讀商品圖片；圖片資訊已透過 OpenCLIP 圖片向量參與檢索與排序。Gemini 的角色是根據檢索出的商品名稱、系統標籤與檢索分數，整理成自然、可信的導購建議。

## 版本功能

### v0.1：多模態商品搜尋 Demo

- 商品資料前處理與欄位清理
- 使用 OpenCLIP 建立商品圖片向量
- 使用 ChromaDB 作為本地向量資料庫
- 採用 Title-first Search 進行候選商品篩選
- 使用圖片相似度進行 reranking
- 建立 Streamlit 商品搜尋展示介面
- 使用 Query Audit 檢查資料集對不同查詢的支援程度

### v0.2：Gemini 導購式推薦回答

- 將 Top-K 檢索商品傳入 Gemini API
- 生成 grounded shopping recommendation
- Gemini 只根據已檢索出的商品資料回答，不自行創造商品資訊
- 對資料不足或相近款商品提供較謹慎的導購說明
- 在回答中提醒使用者實際顏色與細節仍應以頁面商品圖為準

## 系統架構

```mermaid
flowchart TD
    A[Product Data<br/>id, product_name, image_path] --> B[Data Preprocessing]
    B --> C[Title and Attribute Extraction]
    B --> D[Image Embedding with OpenCLIP]
    D --> E[ChromaDB Image Vector Store]
    C --> F[Title-first Candidate Filtering]
    E --> G[Image Similarity Reranking]
    F --> G
    G --> H[Top-K Retrieved Products]
    H --> I[Gemini Grounded Recommendation]
    H --> J[Streamlit Product Cards]
    I --> K[Interactive Shopping Assistant Demo]
    J --> K
```

## 專案特色

- 多模態商品搜尋：結合商品名稱與圖片向量
- 本地向量資料庫：使用 ChromaDB 儲存圖片 embeddings
- 圖片 reranking：使用 OpenCLIP 評估圖片與查詢的相似度
- 弱標註資料處理：針對缺乏完整商品屬性的資料設計 Title-first Search
- Gemini grounded recommendation：根據檢索結果生成導購回答
- 可解釋排序：前端顯示總分、名稱分數與圖片分數
- Query Audit：分析資料集對不同查詢的支援程度

## 技術棧

- Python
- OpenCLIP
- ChromaDB
- Streamlit
- Gemini API
- Pandas
- NumPy
- Pillow
- Matplotlib
- scikit-learn

## 專案結構

```text
fashion_rag_project/
├── app.py
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_build_vector_db.py
│   ├── 04_prepare_product_attributes.py
│   ├── 06_generate_english_caption.py
│   ├── 07_build_text_vector_db.py
│   ├── 09_title_first_search_demo.py
│   ├── 10_dataset_query_audit.py
│   └── llm_recommender.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 執行方式

### 1. 建立並啟動虛擬環境

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 設定 Gemini API Key

在專案根目錄建立 `.env` 檔案：

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash
```

注意：`.env` 不應上傳到 GitHub，請確認 `.gitignore` 已包含：

```gitignore
.env
```

### 4. 啟動 Streamlit Demo

```bash
streamlit run app.py
```

啟動後，系統會開啟本機網頁介面，可輸入商品或穿搭需求進行搜尋。

## 建議展示查詢

目前資料較適合支援商品特徵明確、候選數充足的查詢，例如：

```text
白色高領上衣
黑色短裙
海邊度假風洋裝
甜美短裙
針織外套
灰色大衣
復古洋裝
休閒短褲
```

## 目前限制

目前版本使用的是弱標註商品資料，主要依賴商品名稱與商品圖片路徑。由於原始資料缺乏完整的人工商品屬性標籤，例如實際顏色、領口、袖長、正式程度與適用場合，因此部分較精細的導購查詢可能無法穩定取得理想結果。

例如：

```text
適合上班的簡約襯衫
氣質通勤翻領長袖襯衫
```

這類查詢需要更乾淨且更完整的商品屬性資料。若資料中缺乏足夠候選商品，即使使用多模態向量檢索，也可能受到商品名稱混雜或標籤不足影響。

此外，v0.2 中 Gemini 主要根據檢索出的商品名稱、系統標籤與檢索分數生成導購建議，尚未直接判讀商品圖片。圖片資訊目前是透過 OpenCLIP 圖片向量參與檢索與排序。

## 未來改進方向

- 使用 Vision-Language Model 自動補充商品圖片屬性
- 建立更可靠的 visual attributes，例如顏色、領口、袖長、版型、風格與場合
- 讓 Gemini 或其他多模態模型直接分析 Top-K 商品圖片
- 建立 Precision@K 等檢索評估指標
- 建立可公開的 sample dataset，讓 GitHub 專案可重現
- 將 Demo 部署到雲端平台，方便線上展示

---

# Multimodal Fashion Product Search and Shopping Assistant Demo

## English Version

## Project Overview

This project is a multimodal fashion product search and shopping assistant demo built with Python, OpenCLIP, ChromaDB, Streamlit, and Gemini API.

The system is designed for a weakly-labeled fast-fashion product dataset, where each product mainly contains a product ID, product title, and image path. Since the original dataset does not include complete product descriptions or manually verified attributes, this project adopts a **Title-first Hybrid Retrieval** architecture. It first filters candidate products based on product titles and categories, then applies OpenCLIP image similarity reranking, and finally passes the retrieved products to Gemini API to generate grounded shopping recommendations.

In the current v0.2 version, Gemini does not directly inspect product images. Image information is incorporated through OpenCLIP-based image retrieval and reranking. Gemini generates recommendations based only on retrieved product metadata, system-generated labels, and retrieval scores.

## Version Features

### v0.1: Multimodal Product Search Demo

- Product data preprocessing and cleaning
- Image embedding with OpenCLIP
- Local vector storage with ChromaDB
- Title-first candidate filtering
- Image similarity reranking
- Streamlit product search interface
- Query audit for analyzing dataset limitations

### v0.2: Gemini Grounded Shopping Recommendation

- Pass Top-K retrieved products to Gemini API
- Generate grounded shopping recommendations
- The LLM only responds based on retrieved product metadata
- The LLM avoids inventing unsupported product details
- The recommendation explains uncertainty when data is limited
- The response reminds users to verify actual color and details from product images

## System Architecture

```mermaid
flowchart TD
    A[Product Data<br/>id, product_name, image_path] --> B[Data Preprocessing]
    B --> C[Title and Attribute Extraction]
    B --> D[Image Embedding with OpenCLIP]
    D --> E[ChromaDB Image Vector Store]
    C --> F[Title-first Candidate Filtering]
    E --> G[Image Similarity Reranking]
    F --> G
    G --> H[Top-K Retrieved Products]
    H --> I[Gemini Grounded Recommendation]
    H --> J[Streamlit Product Cards]
    I --> K[Interactive Shopping Assistant Demo]
    J --> K
```

## Features

- Multimodal product search using product titles and image embeddings
- OpenCLIP-based image similarity reranking
- Local vector database with ChromaDB
- Title-first retrieval strategy for weak product metadata
- Gemini-grounded shopping recommendation
- Interpretable ranking scores: final score, title score, and image score
- Query audit for identifying dataset limitations

## Tech Stack

- Python
- OpenCLIP
- ChromaDB
- Streamlit
- Gemini API
- Pandas
- NumPy
- Pillow
- Matplotlib
- scikit-learn

## Project Structure

```text
fashion_rag_project/
├── app.py
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_build_vector_db.py
│   ├── 04_prepare_product_attributes.py
│   ├── 06_generate_english_caption.py
│   ├── 07_build_text_vector_db.py
│   ├── 09_title_first_search_demo.py
│   ├── 10_dataset_query_audit.py
│   └── llm_recommender.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash
```

Make sure `.env` is excluded from GitHub by adding it to `.gitignore`:

```gitignore
.env
```

### 4. Launch the Streamlit demo

```bash
streamlit run app.py
```

After launching the app, a local web interface will open. Users can enter fashion-related product queries and view recommended products.

## Recommended Demo Queries

The current dataset works better for queries with clear product features and sufficient candidate products, such as:

```text
white turtleneck top
black mini skirt
beach vacation dress
sweet mini skirt
knit cardigan
gray coat
vintage dress
casual shorts
```

Chinese demo queries:

```text
白色高領上衣
黑色短裙
海邊度假風洋裝
甜美短裙
針織外套
灰色大衣
復古洋裝
休閒短褲
```

## Current Limitations

The current version uses weak product metadata, mainly product titles and image paths. Since the dataset does not contain complete manually verified product attributes such as actual color, collar type, sleeve length, formality, and suitable occasion, some fine-grained shopping queries may not return ideal results.

For example:

```text
office-style minimal blouse
elegant commuter long-sleeve collared shirt
```

These queries require cleaner and more complete product attributes. Without sufficient high-quality candidates, even multimodal retrieval may produce unstable results.

In v0.2, Gemini generates recommendations based on retrieved product metadata, system-generated labels, and retrieval scores. It does not directly inspect product images. Image information is incorporated through OpenCLIP-based retrieval and reranking.

## Future Improvements

- Use a Vision-Language Model to generate visual product attributes
- Build more reliable visual labels, such as color, collar, sleeve length, fit, style, and occasion
- Enable Gemini or another multimodal model to directly analyze Top-K product images
- Evaluate retrieval quality with Precision@K
- Provide a public sample dataset for reproducibility
- Deploy the demo online for easier portfolio presentation
