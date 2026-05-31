# Multimodal Fashion Product Search Demo

# 多模態時尚商品搜尋 Demo

## Project Overview

## 專案簡介

This project is a first-version multimodal fashion product search demo built with Python, OpenCLIP, ChromaDB, and Streamlit.

本專案是一個第一版的多模態時尚商品搜尋 Demo，使用 Python、OpenCLIP、ChromaDB 與 Streamlit 建立。

The system uses product titles and image embeddings to retrieve visually and semantically relevant fashion products from a weakly-labeled fast-fashion product dataset.

本系統使用商品標題與商品圖片向量，從標註資訊較少的快時尚商品資料集中，檢索出在視覺與語意上較符合使用者需求的商品。

---

## Features

## 功能特色

- Product data preprocessing
  商品資料前處理

- Image embedding with OpenCLIP
  使用 OpenCLIP 建立商品圖片向量

- Local vector storage with ChromaDB
  使用 ChromaDB 進行本地向量資料庫儲存

- Title-first product filtering
  以商品名稱為優先的候選商品篩選策略

- Image similarity reranking
  使用圖片相似度進行重新排序

- Streamlit demo interface
  使用 Streamlit 建立互動式 Demo 介面

- Query audit for checking dataset limitations
  透過查詢診斷分析資料集限制

---

## Tech Stack

## 技術工具

- Python
- OpenCLIP
- ChromaDB
- Streamlit
- Pandas
- NumPy
- Pillow
- Matplotlib

---

## Project Structure

## 專案結構

```text
fashion_rag_project/
├── app.py
├── scripts/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run

## 執行方式

### 1. Clone the repository

### 1. 下載專案

```bash
git clone https://github.com/your-username/multimodal-fashion-search-demo.git
cd multimodal-fashion-search-demo
```

### 2. Install dependencies

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

### 3. 啟動 Streamlit Demo

```bash
streamlit run app.py
```

---

## Example Queries

## 範例查詢

The current demo works better for product queries with clearer visual or title-based cues, such as:

目前版本較適合處理具有明確視覺特徵或商品名稱線索的查詢，例如：

- 白色高領上衣
- 黑色短裙
- 海邊度假風洋裝
- 甜美短裙
- 針織外套
- 灰色大衣
- 復古洋裝
- 休閒短褲

---

## Current Limitations

## 目前限制

The current version uses weak product metadata, mainly product name and image path. Since the dataset does not include detailed human-labeled product attributes, some fine-grained shopping queries may not work well.

目前版本使用的商品資料標註較弱，主要欄位包含商品名稱與圖片路徑。由於資料集中缺乏人工標註的詳細商品屬性，因此部分較精細的導購查詢可能無法穩定取得理想結果。

For example, queries such as office-style blouse recommendation may not perform well if the original dataset does not contain enough clean and relevant candidates.

例如，若原始資料中沒有足夠乾淨且相關的商品候選，像是「適合上班的簡約襯衫」這類查詢效果可能較不穩定。

---

## Future Improvements

## 未來改進方向

Future improvements may include:

未來可進一步改進的方向包括：

- Vision-language model based product attribute extraction
  使用視覺語言模型自動抽取商品圖片屬性

- LLM-generated shopping recommendation responses
  加入大型語言模型生成自然語言導購推薦

- Evaluation with Precision@K
  使用 Precision@K 等指標評估檢索結果品質

- Public sample dataset for reproducible demo
  建立可公開的範例資料集，提升專案可重現性

- More robust handling of weak and noisy product metadata
  強化對弱標註與雜訊商品資料的處理能力

---

## Notes

## 備註

The full dataset and product images are not included in this repository due to data size and source limitations. A small sample dataset may be added in future versions for demonstration purposes.

基於資料大小與來源限制，本專案目前不直接提供完整商品資料與圖片。未來版本可加入小型範例資料集，方便展示與重現。
