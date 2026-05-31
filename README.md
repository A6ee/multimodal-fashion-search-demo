\# Multimodal Fashion Product Search Demo



This project is a first-version multimodal fashion product search demo built with Python, OpenCLIP, ChromaDB, and Streamlit.



The system uses product titles and image embeddings to retrieve visually and semantically relevant fashion products from a weakly-labeled fast-fashion product dataset.



\## Features



\- Product data preprocessing

\- Image embedding with OpenCLIP

\- Local vector storage with ChromaDB

\- Title-first product filtering

\- Image similarity reranking

\- Streamlit demo interface

\- Query audit for checking dataset limitations



\## Tech Stack



\- Python

\- OpenCLIP

\- ChromaDB

\- Streamlit

\- Pandas

\- NumPy

\- Pillow



\## Project Structure



```text

fashion\_rag\_project/

├── app.py

├── scripts/

├── requirements.txt

├── README.md

└── .gitignore

```



\##  How to Run

pip install -r requirements.txt

streamlit run app.py





\##  Current Limitations



The current version uses weak product metadata, mainly product name and image path. Some fine-grained shopping queries, such as office-style blouse recommendation, may not work well if the original dataset does not contain enough clean candidates.



Future improvements may include:



\- Vision-language model based product attribute extraction

\- LLM-generated shopping recommendation responses

\- Evaluation with Precision@K

\- Public sample dataset for reproducible demo



