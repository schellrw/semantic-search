"""Pure semantic search functionality."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import lancedb


class SemanticSearcher:
    """Pure semantic search using vector databases."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        vector_db_path: Optional[str] = None
    ):
        """
        Initialize semantic searcher.
        
        Args:
            model_name: Name of the transformer model
            vector_db_path: Path to existing vector database
        """
        self.model_name = model_name
        self.model = None
        self.vector_db_path = vector_db_path
        self.db = None
        self.table = None
        self.fitted = False
    
    def fit(self, df: pd.DataFrame, text_column: str = 'combined_text_v1') -> 'SemanticSearcher':
        """
        Fit semantic searcher on dataset.
        
        Args:
            df: DataFrame with product data
            text_column: Column containing text to embed
            
        Returns:
            Self for method chaining
        """
        print(f"Initializing semantic searcher with {self.model_name}")
        
        # Load transformer model
        self.model = SentenceTransformer(self.model_name)
        
        # Create embeddings
        texts = df[text_column].tolist()
        print(f"Creating embeddings for {len(texts)} products...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Store reference to data for search results
        self.product_df = df.copy()
        self.embeddings = embeddings
        self.fitted = True
        
        return self
    
    def fit_with_vector_db(
        self, 
        vector_db_path: str, 
        table_name: str = "products"
    ) -> 'SemanticSearcher':
        """
        Fit semantic searcher using existing vector database.
        
        Args:
            vector_db_path: Path to vector database
            table_name: Name of the table in vector database
            
        Returns:
            Self for method chaining
        """
        print(f"Loading vector database from {vector_db_path}")
        
        # Load transformer model
        self.model = SentenceTransformer(self.model_name)
        
        # Connect to vector database
        self.db = lancedb.connect(vector_db_path)
        self.table = self.db.open_table(table_name)
        
        print(f"Connected to vector database: {self.table.count_rows()} vectors")
        self.fitted = True
        
        return self
    
    def search(
        self, 
        query: str, 
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Perform semantic search.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            DataFrame with search results
        """
        if not self.fitted:
            raise ValueError("SemanticSearcher must be fitted before search")
        
        # If using vector database
        if self.table is not None:
            return self._search_vector_db(query, top_k)
        else:
            return self._search_embeddings(query, top_k)
    
    def _search_vector_db(self, query: str, top_k: int) -> pd.DataFrame:
        """Search using vector database."""
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Search vector database
        results = self.table.search(query_embedding).distance_type("cosine").limit(top_k).to_pandas()
        
        # Format results
        results['score'] = 1 - results['_distance']  # Convert distance to similarity
        results = results.drop(columns=['_distance'], errors='ignore')
        
        return results[['product_id', 'product_title', 'score']]
    
    def _search_embeddings(self, query: str, top_k: int) -> pd.DataFrame:
        """Search using in-memory embeddings."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create results DataFrame
        results = []
        for idx in top_indices:
            results.append({
                'product_id': self.product_df.iloc[idx]['product_id'],
                'product_title': self.product_df.iloc[idx]['product_title'],
                'score': similarities[idx]
            })
        
        return pd.DataFrame(results)
    
    def batch_search(
        self, 
        queries: list, 
        top_k: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of top results per query
            
        Returns:
            Dictionary mapping query to results DataFrame
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, top_k)
        return results 