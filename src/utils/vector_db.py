"""Vector database utilities for semantic search."""

import pandas as pd
import numpy as np
import lancedb
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Tuple, Optional, Dict, Any


def create_embeddings_and_vector_db(
    df: pd.DataFrame, 
    text_column: str, 
    model_name: str = "all-MiniLM-L6-v2", 
    db_path_suffix: str = ""
) -> Tuple[SentenceTransformer, np.ndarray, lancedb.db.LanceDBConnection, Any]:
    """
    Complete pipeline: raw text -> embeddings -> vector database.
    This demonstrates the full workflow a production system would use.
    
    Args:
        df: DataFrame with product data
        text_column: Column containing text to embed
        model_name: Name of the transformer model
        db_path_suffix: Suffix for database path
        
    Returns:
        Tuple of (model, embeddings, db, table)
    """
    print(f"Creating embeddings for {len(df)} products using {model_name}...")
    
    # Step 1: Load transformer model
    model = SentenceTransformer(model_name)
    
    # Step 2: Create embeddings from text
    texts = df[text_column].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Created embeddings shape: {embeddings.shape}")
    
    # Step 3: Create LanceDB vector database
    db_path = f"src/vector_databases/lancedb_hybrid{db_path_suffix}"
    os.makedirs(db_path, exist_ok=True)
    db = lancedb.connect(db_path)
    
    # Prepare data for LanceDB
    data = []
    for idx, (_, row) in enumerate(df.iterrows()):
        data.append({
            "vector": embeddings[idx].tolist(),
            "product_id": row['product_id'],
            "query": row['query'],
            "product_title": row['product_title'],
            "product_description": row.get('product_description', ''),
            "product_bullet_point": row.get('product_bullet_point', ''),
            "product_brand": row.get('product_brand', ''),
            "product_color": row.get('product_color', ''),
            "combined_text": row[text_column]
        })
    
    # Create table (drop if exists)
    table_name = "products"
    if table_name in db.table_names():
        db.drop_table(table_name)
    
    table = db.create_table(table_name, data)
    
    print(f"Created LanceDB table: {len(data)} vectors, {embeddings.shape[1]} dimensions")
    print(f"Database path: {db_path}")
    
    return model, embeddings, db, table


class VectorDatabaseManager:
    """Manager for vector database operations."""
    
    def __init__(self, base_path: str = "src/vector_databases"):
        """
        Initialize vector database manager.
        
        Args:
            base_path: Base path for vector databases
        """
        self.base_path = Path(base_path)
        self.active_connections = {}
    
    def create_database(
        self,
        df: pd.DataFrame,
        text_column: str,
        db_name: str,
        model_name: str = "all-MiniLM-L6-v2",
        table_name: str = "products"
    ) -> Tuple[SentenceTransformer, lancedb.db.LanceDBConnection, Any]:
        """
        Create a new vector database.
        
        Args:
            df: DataFrame with product data
            text_column: Column containing text to embed
            db_name: Name of the database
            model_name: Name of the transformer model
            table_name: Name of the table
            
        Returns:
            Tuple of (model, db, table)
        """
        print(f"Creating vector database: {db_name}")
        
        # Load transformer model
        model = SentenceTransformer(model_name)
        
        # Create embeddings
        texts = df[text_column].tolist()
        print(f"Creating embeddings for {len(texts)} texts...")
        embeddings = model.encode(texts, show_progress_bar=True)
        
        # Create database directory
        db_path = self.base_path / db_name
        os.makedirs(db_path, exist_ok=True)
        
        # Connect to database
        db = lancedb.connect(str(db_path))
        
        # Prepare data for LanceDB
        data = []
        for idx, (_, row) in enumerate(df.iterrows()):
            record = {
                "vector": embeddings[idx].tolist(),
                "product_id": row['product_id'],
                "product_title": row['product_title'],
                "combined_text": row[text_column]
            }
            
            # Add optional columns if they exist
            optional_cols = ['query', 'product_description', 'product_bullet_point', 
                           'product_brand', 'product_color']
            for col in optional_cols:
                if col in row:
                    record[col] = row.get(col, '')
            
            data.append(record)
        
        # Create table (drop if exists)
        if table_name in db.table_names():
            db.drop_table(table_name)
        
        table = db.create_table(table_name, data)
        
        # Store connection
        self.active_connections[db_name] = {
            'db': db,
            'table': table,
            'model': model,
            'model_name': model_name
        }
        
        print(f"Created database '{db_name}' with {len(data)} vectors")
        return model, db, table
    
    def connect_to_database(
        self,
        db_name: str,
        table_name: str = "products",
        model_name: str = "all-MiniLM-L6-v2"
    ) -> Tuple[SentenceTransformer, lancedb.db.LanceDBConnection, Any]:
        """
        Connect to existing vector database.
        
        Args:
            db_name: Name of the database
            table_name: Name of the table
            model_name: Name of the transformer model
            
        Returns:
            Tuple of (model, db, table)
        """
        db_path = self.base_path / db_name
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database '{db_name}' not found at {db_path}")
        
        print(f"Connecting to database: {db_name}")
        
        # Load model
        model = SentenceTransformer(model_name)
        
        # Connect to database
        db = lancedb.connect(str(db_path))
        table = db.open_table(table_name)
        
        # Store connection
        self.active_connections[db_name] = {
            'db': db,
            'table': table,
            'model': model,
            'model_name': model_name
        }
        
        print(f"Connected to database '{db_name}' with {table.count_rows()} vectors")
        return model, db, table
    
    def list_databases(self) -> list:
        """List available databases."""
        if not self.base_path.exists():
            return []
        
        databases = []
        for item in self.base_path.iterdir():
            if item.is_dir() and (item / "_versions").exists():
                databases.append(item.name)
        
        return databases
    
    def get_database_info(self, db_name: str) -> Dict[str, Any]:
        """Get information about a database."""
        db_path = self.base_path / db_name
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database '{db_name}' not found")
        
        try:
            db = lancedb.connect(str(db_path))
            tables = db.table_names()
            
            info = {
                'name': db_name,
                'path': str(db_path),
                'tables': tables
            }
            
            # Get table info if products table exists
            if 'products' in tables:
                table = db.open_table('products')
                info['num_vectors'] = table.count_rows()
                
                # Get sample record to determine schema
                sample = table.limit(1).to_pandas()
                if len(sample) > 0:
                    info['schema'] = list(sample.columns)
                    if 'vector' in sample.columns:
                        vector_dim = len(sample['vector'].iloc[0])
                        info['vector_dimension'] = vector_dim
            
            return info
            
        except Exception as e:
            return {'name': db_name, 'path': str(db_path), 'error': str(e)}
    
    def delete_database(self, db_name: str) -> bool:
        """
        Delete a database.
        
        Args:
            db_name: Name of the database to delete
            
        Returns:
            True if successful, False otherwise
        """
        db_path = self.base_path / db_name
        
        if not db_path.exists():
            print(f"Database '{db_name}' does not exist")
            return False
        
        try:
            import shutil
            shutil.rmtree(db_path)
            
            # Remove from active connections
            if db_name in self.active_connections:
                del self.active_connections[db_name]
            
            print(f"Deleted database '{db_name}'")
            return True
            
        except Exception as e:
            print(f"Error deleting database '{db_name}': {e}")
            return False
    
    def close_connection(self, db_name: str) -> None:
        """Close database connection."""
        if db_name in self.active_connections:
            del self.active_connections[db_name]
            print(f"Closed connection to '{db_name}'")
    
    def close_all_connections(self) -> None:
        """Close all database connections."""
        self.active_connections.clear()
        print("Closed all database connections")
    
    def get_active_connections(self) -> list:
        """Get list of active database connections."""
        return list(self.active_connections.keys()) 