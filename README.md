# Hybrid Search for RAG LLM Applications

## Overview
This repository implements a search engine that combines BM25 lexical search with semantic embedding search to retrieve news articles. It demonstrates a hybrid search approach for Retrieval-Augmented Generation (RAG) applications, using reranking with a cross-encoder for improved relevance.

## Features
- **BM25 Lexical Search** – Implements fast, keyword-based retrieval using the [`LexicalSearch`](engine.py) class.
- **Semantic Search** – Leverages sentence embeddings with FAISS via the [`SemanticSearch`](engine.py) class.
- **Hybrid Search** – Merges lexical and semantic results using reciprocal rank fusion ([`reciprocal_rank_fusion`](engine.py)).
- **Reranking** – Applies a Transformer-based cross-encoder to rerank the combined results, integrated in [app.py](app.py).
- **Interactive Interface** – Provides a Gradio web interface for live search demos.

## Repository Structure
- `app.py` – Main application that ties together search functionalities and the Gradio UI.
- `engine.py` – Contains the implementations of [`LexicalSearch`](engine.py), [`SemanticSearch`](engine.py), and the reciprocal rank fusion function.
- `embddings.py` – Handles document preprocessing and embedding generation via OpenAI.
- `data/` – Directory for datasets, indices, and storage of results.
- `.gitignore`, `pyproject.toml`, etc. – Configuration and dependency management files.

## Dataset  
The project uses the **News Category Dataset v3** from Kaggle, containing news articles categorized into different topics. It provides structured news data with headlines, short descriptions, and categories, making it ideal for evaluating search and retrieval methods.  

Dataset source: [News Category Dataset v3](https://www.kaggle.com/code/vikashrajluhaniwal/recommending-news-articles-based-on-read-articles?select=News_Category_Dataset_v3.json)  

## Conclusion  
This project demonstrates how combining different search techniques enhances retrieval for RAG-based LLM applications. The **hybrid search and reranking approach improves accuracy and relevance**, making it a strong foundation for real-world AI systems  
