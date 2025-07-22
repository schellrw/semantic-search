import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle
import os


def create_tfidf_embeddings(df, text_column='combined_text', max_features=1000):
    """Create TF-IDF embeddings - baseline approach"""
    texts = df[text_column].tolist()
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        lowercase=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer


def create_transformer_embeddings(df, text_column='combined_text', model_name="all-MiniLM-L6-v2"):
    """Create transformer embeddings using sentence-transformers"""
    print(f"Creating transformer embeddings with {model_name}...")
    
    model = SentenceTransformer(model_name)
    texts = df[text_column].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print(f"Transformer embeddings shape: {embeddings.shape}")
    return embeddings, model


class TFIDFEmbedder:
    def __init__(self, max_features=1000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.embeddings = None
        self.fitted = False
    
    def fit(self, texts):
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=self.ngram_range,
            lowercase=True
        )
        
        self.embeddings = self.vectorizer.fit_transform(texts)
        self.fitted = True
        print(f"Fitted TF-IDF embedder: {self.embeddings.shape}")
        return self
    
    def transform(self, texts):
        if not self.fitted:
            raise ValueError("TFIDFEmbedder must be fitted before transform")
        return self.vectorizer.transform(texts)
    
    def search(self, query, product_df, top_k=10):
        if not self.fitted:
            raise ValueError("TFIDFEmbedder must be fitted before search")
        
        query_embedding = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'product_id': product_df.iloc[idx]['product_id'],
                'product_title': product_df.iloc[idx]['product_title'],
                'score': similarities[idx]
            })
        
        return pd.DataFrame(results)


class TransformerEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.fitted = False
    
    def fit(self, texts):
        print(f"Loading transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        print(f"Encoding {len(texts)} texts...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        self.fitted = True
        
        print(f"Fitted transformer embedder: {self.embeddings.shape}")
        return self
    
    def transform(self, texts):
        if not self.fitted:
            raise ValueError("TransformerEmbedder must be fitted before transform")
        return self.model.encode(texts, show_progress_bar=True)
    
    def search(self, query, product_df, top_k=10):
        if not self.fitted:
            raise ValueError("TransformerEmbedder must be fitted before search")
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'product_id': product_df.iloc[idx]['product_id'],
                'product_title': product_df.iloc[idx]['product_title'],
                'score': similarities[idx]
            })
        
        return pd.DataFrame(results)
    
    def encode_query(self, query):
        if not self.fitted:
            raise ValueError("TransformerEmbedder must be fitted before encoding")
        return self.model.encode([query])[0] 