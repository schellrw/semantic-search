"""Hybrid search functionality combining semantic and keyword matching."""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import lancedb
import json


def calculate_keyword_score(query, product_text, keywords_dict):
    """Calculate keyword matching score using frequency-weighted word matching"""
    query_words = [word.lower().strip() for word in query.split()]
    product_text_lower = product_text.lower()
    
    if not query_words:
        return 0.0
    
    total_weight = 0
    matched_weight = 0
    
    for word in query_words:
        # Check if word is in keywords dict (should include coffee, maker, etc.)
        if word in keywords_dict:
            frequency = keywords_dict[word]
            weight = 1.0 / (1.0 + frequency)
            total_weight += weight
            
            # Check if word appears in product text
            if word in product_text_lower:
                matched_weight += weight
        else:
            # For words not in keywords dict, give them a small weight
            # This prevents total_weight from being 0 for queries with uncommon words
            weight = 0.1
            total_weight += weight
            
            if word in product_text_lower:
                matched_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return matched_weight / total_weight


class HybridSearcher:
    def __init__(self, model_name="all-MiniLM-L6-v2", keywords_dict=None, 
                 alpha=0.7, beta=0.3, keyword_threshold=0.1):
        self.model_name = model_name
        self.model = None
        self.keywords_dict = keywords_dict or {}
        self.alpha = alpha
        self.beta = beta
        self.keyword_threshold = keyword_threshold
        self.db = None
        self.table = None
        self.fitted = False
    
    def fit_with_vector_db(self, vector_db_path, table_name="products", keywords_path=None):
        print(f"Loading hybrid searcher with {self.model_name}")
        
        self.model = SentenceTransformer(self.model_name)
        self.db = lancedb.connect(vector_db_path)
        self.table = self.db.open_table(table_name)
        
        if keywords_path:
            with open(keywords_path, 'r') as f:
                self.keywords_dict = json.load(f)
            print(f"Loaded {len(self.keywords_dict)} keywords")
        
        print(f"Connected to vector database: {self.table.count_rows()} vectors")
        self.fitted = True
        return self
    
    def fit(self, df, text_column='combined_text_v1', keywords_dict=None):
        print(f"Initializing hybrid searcher with {self.model_name}")
        
        self.model = SentenceTransformer(self.model_name)
        
        texts = df[text_column].tolist()
        print(f"Creating embeddings for {len(texts)} products...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        print(f"Embeddings shape: {embeddings.shape}")
        
        self.product_df = df.copy()
        self.embeddings = embeddings
        self.text_column = text_column
        
        if keywords_dict:
            self.keywords_dict = keywords_dict
            print(f"Loaded {len(self.keywords_dict)} keywords")
        
        self.fitted = True
        return self
    
    def search(self, query, top_k=10):
        if not self.fitted:
            raise ValueError("HybridSearcher must be fitted before search")
        
        if self.table is not None:
            return self._search_vector_db(query, top_k)
        else:
            return self._search_embeddings(query, top_k)
    
    def _search_vector_db(self, query, top_k):
        query_embedding = self.model.encode([query])[0]
        semantic_results = self.table.search(query_embedding).distance_type("cosine").limit(top_k * 2).to_pandas()
        
        hybrid_scores = []
        for idx, row in semantic_results.iterrows():
            semantic_score = 1 - row['_distance']
            keyword_score = calculate_keyword_score(query, row['combined_text'], self.keywords_dict)
            
            if keyword_score >= self.keyword_threshold:
                hybrid_score = self.alpha * semantic_score + self.beta * keyword_score
            else:
                hybrid_score = semantic_score
            
            hybrid_scores.append({
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'hybrid_score': hybrid_score,
                'product_id': row['product_id'],
                'product_title': row['product_title'],
                'combined_text': row['combined_text']
            })
        
        hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
        top_results = hybrid_scores[:top_k]
        
        result_df = pd.DataFrame(top_results)
        result_df['score'] = result_df['hybrid_score']
        
        return result_df
    
    def _search_embeddings(self, query, top_k):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        hybrid_scores = []
        for idx, similarity in enumerate(similarities):
            product_text = self.product_df.iloc[idx][self.text_column]
            keyword_score = calculate_keyword_score(query, product_text, self.keywords_dict)
            
            if keyword_score >= self.keyword_threshold:
                hybrid_score = self.alpha * similarity + self.beta * keyword_score
            else:
                hybrid_score = similarity
            
            hybrid_scores.append({
                'index': idx,
                'semantic_score': similarity,
                'keyword_score': keyword_score,
                'hybrid_score': hybrid_score
            })
        
        hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
        top_results = hybrid_scores[:top_k]
        
        results = []
        for result in top_results:
            idx = result['index']
            results.append({
                'product_id': self.product_df.iloc[idx]['product_id'],
                'product_title': self.product_df.iloc[idx]['product_title'],
                'semantic_score': result['semantic_score'],
                'keyword_score': result['keyword_score'],
                'hybrid_score': result['hybrid_score'],
                'score': result['hybrid_score']
            })
        
        return pd.DataFrame(results)
    
    def update_weights(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        print(f"Updated weights: alpha={alpha}, beta={beta}")
    
    def update_keyword_threshold(self, threshold: float) -> None:
        """
        Update keyword threshold.
        
        Args:
            threshold: Minimum keyword score to apply boost
        """
        self.keyword_threshold = threshold
        print(f"Updated keyword threshold: {threshold}")
    
    def load_keywords(self, keywords_path: str) -> None:
        """
        Load keywords from JSON file.
        
        Args:
            keywords_path: Path to keywords JSON file
        """
        with open(keywords_path, 'r') as f:
            self.keywords_dict = json.load(f)
        print(f"Loaded {len(self.keywords_dict)} keywords from {keywords_path}")
    
    def explain_search(self, query, top_k=5):
        results = self.search(query, top_k)
        
        print(f"\nHybrid Search Explanation for query: '{query}'")
        print(f"Weights: α={self.alpha} (semantic), β={self.beta} (keyword)")
        print(f"Keyword threshold: {self.keyword_threshold}")
        print("-" * 80)
        
        for idx, row in results.iterrows():
            print(f"Rank {idx + 1}: {row['product_title'][:60]}...")
            print(f"  Semantic: {row.get('semantic_score', 0):.3f}")
            print(f"  Keyword:  {row.get('keyword_score', 0):.3f}")
            print(f"  Hybrid:   {row.get('hybrid_score', row['score']):.3f}")
            print()
        
        return results 