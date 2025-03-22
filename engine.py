import os
import numpy as np
import pandas as pd
import faiss
import bm25s
from Stemmer import Stemmer
from operator import itemgetter
from typing import List, Dict, Optional, Tuple
import openai

# Constants
RETRIVERS_PATH = './data/retrivers/'
INDEX_PATH = './data/'

class LexicalSearch:
    def __init__(self, df: pd.DataFrame):
        self.data = df.copy()
        self.stemmer = Stemmer("english")
        self.retrievers = {}
        self.search_columns = ['headline', 'short_description', 'authors']

        required_cols = {'headline', 'short_description', 'authors', 'link', 'category', 'date'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"🚫 Missing required columns: {required_cols - set(df.columns)}")

    def index(self):
        print("📊 Building lexical indices...")
        self.corpus = {col: self.data[col].fillna('').tolist() for col in self.search_columns}
        for col, docs in self.corpus.items():
            tokens = bm25s.tokenize(docs, stopwords="en", stemmer=self.stemmer)
            retriever = bm25s.BM25()
            retriever.index(tokens)
            self.retrievers[col] = retriever
        print("✅ Lexical indices built successfully!")

    def save(self, path: str):
        print("💾 Saving lexical indices...")
        for col, retriever in self.retrievers.items():
            retriever.save(f"{path}_{col}", corpus=self.corpus[col])
        print("✅ Lexical indices saved successfully!")

    def load(self, path: str):
        print("📥 Loading lexical indices...")
        for col in self.search_columns:
            self.retrievers[col] = bm25s.BM25.load(f"{path}_{col}", load_corpus=True)
        print("✅ Lexical indices loaded successfully!")

    def search(self,
               query: str, 
               k: int = 5, 
               limit: int = 10, 
               category: Optional[List[str]] = None, 
               year_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)
        ranked_results = []

        for col, retriever in self.retrievers.items():
            results, scores = retriever.retrieve(query_tokens, k=k)
            for rank, (entry, score) in enumerate(zip(results[0], scores[0]), start=1):
                if score > 0:
                    entry_data = {
                        'id': entry['id'],
                        'score': score,
                        'matched_field': col
                    }
                    ranked_results.append(entry_data)

        matches = sorted(ranked_results, key=itemgetter('score'), reverse=True)

        # Map results to actual news articles
        filtered_results = []
        for match in matches:
            row = self.data.iloc[match['id']]
            result = {
                'link': row['link'],
                'headline': row['headline'],
                'category': row['category'],
                'short_description': row['short_description'],
                'authors': row['authors'],
                'date': row['date'],
                'score': match['score'],
                'matched_field': match['matched_field']
            }
            filtered_results.append(result)

        # Apply category filtering
        if category and len(category) > 0:
            filtered_results = [r for r in filtered_results if r['category'] in category]

        # Apply year range filtering
        if year_range and len(year_range) == 2:
            start_year, end_year = year_range
            filtered_results = [
                r for r in filtered_results 
                if start_year <= pd.to_datetime(r['date']).year <= end_year
            ]

        return filtered_results[:limit]

class SemanticSearch:
    def __init__(self, df: pd.DataFrame, model: str = "text-embedding-3-small", dimensions: int = 1536):
        self.data = df
        self.model = model
        self.dimensions = dimensions
        self.faiss_index = None
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def build(self):
        print("📊 Building semantic index...")
        embeddings = np.vstack(self.data["embedding"].values)
        index = faiss.IndexFlatL2(self.dimensions)
        index.add(embeddings)
        self.faiss_index = index
        print("✅ Semantic index built successfully!")

    def save(self, path: str):
        print("💾 Saving semantic index...")
        if not self.faiss_index:
            print("⚠️ No index to save! Build the index first.")
            return
        faiss.write_index(self.faiss_index, f"{path}_documents.index")
        print("✅ Semantic index saved successfully!")

    def load(self, path: str):
        print("📥 Loading semantic index...")
        self.faiss_index = faiss.read_index(f"{path}_documents.index")
        print("✅ Semantic index loaded successfully!")

    def _get_embedding(self, text: str):
        response = openai.embeddings.create(model=self.model, input=[text])
        return np.array(response.data[0].embedding).astype("float32")

    def search(self, 
               query: str, 
               k: int = 5, 
               limit: int = 10, 
               category: Optional[List[str]] = None, 
               year_range: Optional[Tuple[int, int]] = None):
        if not self.faiss_index:
            print("⚠️ No FAISS index found! Build or load the index first.")
            return []

        query_embedding = self._get_embedding(query).reshape(1, -1)
        matches = []

        distances, indices = self.faiss_index.search(query_embedding, k)
        for idx, distance in zip(indices[0], distances[0]):
            matches.append({'id': idx, 'score': 1 / (1 + distance), 'matched_field': 'semantic'})

        filtered_results = []
        for m in matches:
            row = self.data.iloc[m['id']]
            filtered_results.append({
                'link': row.get('link', 'N/A'),
                'headline': row.get('headline', 'N/A'),
                'category': row.get('category', 'N/A'),
                'short_description': row.get('short_description', 'N/A'),
                'authors': row.get('authors', 'N/A'),
                'date': row.get('date', None),
                'score': m['score'],
                'matched_field': m['matched_field']
            })

        # Apply category filtering
        if category and len(category) > 0:
            filtered_results = [c for c in filtered_results if c['category'] in category]
            
        # Apply year range filtering
        if year_range and len(year_range) == 2:
            start_year, end_year = year_range
            filtered_results = [
                c for c in filtered_results 
                if start_year <= pd.to_datetime(c['date']).year <= end_year
            ]

        return sorted(filtered_results, key=itemgetter('score'), reverse=True)[:limit]

# Reciprocal Rank Fusion for combining search results
def reciprocal_rank_fusion(search_results: List[List[Dict]], k: int = 60) -> List[Dict]:
    rrf_scores = {}
    
    for results in search_results:
        for rank, result in enumerate(results):
            link = result['link']
            
            # If we've seen this document before, update its score
            if link in rrf_scores:
                rrf_scores[link]['score'] += 1.0 / (rank + k)
                rrf_scores[link]['result'] = result
            else:
                rrf_scores[link] = {
                    'score': 1.0 / (rank + k),
                    'result': result
                }
    
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    final_results = []
    for link, data in sorted_results:
        result = data['result'].copy()
        result['score'] = data['score']
        final_results.append(result)
    
    return final_results