# 多模態時尚商品搜尋 Demo

# Multimodal Fashion Product Search Demo

## 中文版

## 專案簡介

本專案是一個第一版的多模態時尚商品搜尋 Demo，使用 Python、OpenCLIP、ChromaDB 與 Streamlit 建置。

本系統以快時尚商品資料為基礎，資料欄位主要包含商品 ID、商品名稱與商品圖片路徑。由於原始資料缺乏完整商品描述與人工標註屬性，本專案採用「商品名稱優先搜尋（Title-first Search）」策略，先根據商品名稱與商品大類篩選候選商品，再使用 OpenCLIP 圖片向量進行相似度排序，最後透過 Streamlit 建立可互動的商品搜尋展示介面。

## 專案特色

- 商品資料前處理與欄位清理
- 使用 OpenCLIP 建立商品圖片向量
- 使用 ChromaDB 作為本地向量資料庫
- 採用 Title-first Search 策略進行商品候選篩選
- 使用圖片相似度進行 reranking
- 建立 Streamlit Demo 網頁介面
- 透過 Query Audit 分析資料集對不同查詢的支援程度
- 根據資料限制調整搜尋策略，而非單純依賴模型輸出

## 技術棧

- Python
- OpenCLIP
- ChromaDB
- Streamlit
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
│   └── 10_dataset_query_audit.py
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

### 3. 啟動 Streamlit Demo

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

目前版本使用的是弱標註商品資料，主要依賴商品名稱與商品圖片路徑。由於資料中缺乏完整的人工商品屬性標籤，例如實際顏色、領口、袖長、正式程度與適用場合，因此部分較精細的導購查詢可能無法穩定取得理想結果。

例如：

```text
適合上班的簡約襯衫
氣質通勤翻領長袖襯衫
```

這類查詢需要更乾淨且更完整的商品屬性資料，否則即使使用多模態向量檢索，也可能因候選商品不足或商品名稱混雜而影響結果。

## 未來改進方向

- 使用 Vision-Language Model 自動補充商品圖片屬性
- 建立更可靠的視覺標籤，例如顏色、領口、袖長、版型、風格與場合
- 加入 LLM 生成導購式推薦回覆
- 建立 Precision@K 等檢索評估指標
- 建立可公開的 sample dataset，讓 GitHub 專案可重現
- 將 Demo 部署到雲端平台，方便線上展示

---

## English Version

## Project Overview

This project is a first-version multimodal fashion product search demo built with Python, OpenCLIP, ChromaDB, and Streamlit.

The system is designed for a weakly-labeled fast-fashion product dataset, where each product mainly contains a product ID, product title, and image path. Since the original dataset does not include complete product descriptions or manually labeled attributes, this project adopts a title-first search strategy. It first filters candidate products based on product titles and categories, then applies OpenCLIP image similarity reranking, and finally presents the results through an interactive Streamlit demo interface.

## Features

- Product data preprocessing and cleaning
- Image embedding with OpenCLIP
- Local vector storage with ChromaDB
- Title-first product candidate filtering
- Image similarity reranking
- Interactive Streamlit demo interface
- Query audit for analyzing dataset limitations
- Search strategy adjustment based on real data constraints

## Tech Stack

- Python
- OpenCLIP
- ChromaDB
- Streamlit
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
│   └── 10_dataset_query_audit.py
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

### 3. Launch the Streamlit demo

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

## Future Improvements

- Use a Vision-Language Model to generate visual product attributes
- Build more reliable visual labels, such as color, collar, sleeve length, fit, style, and occasion
- Add LLM-generated shopping recommendation responses
- Evaluate retrieval quality with Precision@K
- Provide a public sample dataset for reproducibility
- Deploy the demo online for easier portfolio presentation
