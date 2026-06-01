# Decision Log

This document records the main design decisions made during the development of the multimodal fashion product search demo.

## 1. Initial Direction: Pure Multimodal Vector Retrieval

The project initially explored a pure multimodal retrieval approach. Product titles and images were embedded into a shared vector space using OpenCLIP, and ChromaDB was used as the local vector database.

This approach worked reasonably well for broad visual queries, such as:

- white turtleneck top
- black mini skirt
- beach vacation dress

However, it became unstable for fine-grained shopping queries such as:

- office-style minimal blouse
- elegant commuter long-sleeve collared shirt

The main issue was not the retrieval model itself, but the weak product metadata.

## 2. Dataset Limitation

The original dataset mainly contained:

- product ID
- product title
- image path

It did not contain reliable manually verified product attributes such as:

- actual color
- collar type
- sleeve length
- formality
- style
- suitable occasion

Some product titles also contained mixed or noisy terms, such as T-shirt, layered shirt, fake two-piece, or blouse-like items. This made pure vector retrieval unstable for fine-grained intent matching.

## 3. Shift to Title-first Hybrid Retrieval

After dataset auditing, the retrieval strategy was redesigned as Title-first Hybrid Retrieval.

The updated retrieval pipeline is:

1. Parse the user query into product-related conditions.
2. Filter candidate products using product title and category.
3. Use OpenCLIP image similarity to rerank candidates.
4. Display final results in Streamlit.

This approach is more stable and interpretable for weakly-labeled product data.

## 4. Gemini Grounded Recommendation

Gemini API was added in v0.2 to generate natural shopping recommendation responses.

The LLM is not responsible for retrieving products. Instead, retrieval is handled by the search pipeline, and Gemini only summarizes the retrieved products.

In v0.2:

- Gemini receives product titles, system-generated labels, and retrieval scores.
- Gemini does not directly inspect product images.
- Image information is incorporated through OpenCLIP image reranking.
- The prompt explicitly prevents the model from inventing unsupported product details.

## 5. Future Direction

The next major improvement is to use a Vision-Language Model to generate structured visual attributes for each product image.

Potential visual attributes include:

- visual color
- visual subcategory
- collar type
- sleeve length
- fit
- style
- suitable occasion
- whether the product is layered or fake two-piece

This would allow the system to better support fine-grained shopping queries.
