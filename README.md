# Hybrid Search for RAG LLM Applications  

## Why This Matters  
Effective search is crucial for Retrieval-Augmented Generation (RAG) in LLM applications. This project explores **BM25 Lexical Search, Semantic Search, Hybrid Search, and Reranking** to improve information retrieval for better LLM responses.  

## Features  
- **BM25 Lexical Search** – Fast, keyword-based retrieval with `BM25Retriever`.  
- **Semantic Search** – Context-aware search using sentence embeddings (`SemanticSearch`).  
- **Hybrid Search** – Blends lexical and semantic results dynamically.  
- **Reranking** – Uses a Cross Encoder to refine search results.  

## Dataset  
The project uses the **News Category Dataset v3** from Kaggle, containing news articles categorized into different topics. It provides structured news data with headlines, short descriptions, and categories, making it ideal for evaluating search and retrieval methods.  

Dataset source: [News Category Dataset v3](https://www.kaggle.com/code/vikashrajluhaniwal/recommending-news-articles-based-on-read-articles?select=News_Category_Dataset_v3.json)  

## Setup  
1. **Data:** Uses news data stored in Google Drive.  
2. **Dependencies:**  
   - `pandas`, `re`, `numpy`, `torch`, `faiss`, `annoy`  
   - Sentence Transformer: `all-MiniLM-L6-v2`  
   - Cross Encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`  

## How It Works  
### 🔹 Lexical Search  
- `BM25Retriever` indexes text and retrieves documents using BM25 scoring.  

### 🔹 Semantic Search  
- `SemanticSearch` indexes sentence embeddings with FAISS or Annoy.  
- Uses precomputed embeddings for efficient retrieval.  

### 🔹 Hybrid Search  
- Merges lexical and semantic results, weighted by an `alpha` parameter.  
- Ensures unique links and sorts by adjusted relevance scores.  

### 🔹 Reranking  
- Cross Encoder re-evaluates search results for better ranking.  
- Jointly encodes query and candidates for enhanced accuracy.  

## Conclusion  
This project demonstrates how combining different search techniques enhances retrieval for RAG-based LLM applications. The **hybrid search and reranking approach improves accuracy and relevance**, making it a strong foundation for real-world AI systems  
