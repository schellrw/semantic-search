import pandas as pd
import requests
import os
from pathlib import Path


def download_esci_data(data_dir="data/raw"):
    """Download Amazon ESCI dataset files from GitHub"""
    os.makedirs(data_dir, exist_ok=True)
    
    examples_url = "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
    products_url = "https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    
    print("Downloading ESCI dataset...")
    
    examples_response = requests.get(examples_url)
    examples_response.raise_for_status()
    
    examples_path = Path(data_dir) / "examples.parquet"
    with open(examples_path, "wb") as f:
        f.write(examples_response.content)
    
    del examples_response
    
    products_response = requests.get(products_url)
    products_response.raise_for_status()
    
    products_path = Path(data_dir) / "products.parquet"
    with open(products_path, "wb") as f:
        f.write(products_response.content)
    
    del products_response
    print(f"Downloaded ESCI dataset to {data_dir}")


def load_esci_data(data_dir="data/raw"):
    """Load and merge ESCI examples and products data"""
    examples_path = Path(data_dir) / "examples.parquet"
    products_path = Path(data_dir) / "products.parquet"
    
    if not examples_path.exists() or not products_path.exists():
        raise FileNotFoundError(
            f"ESCI data files not found in {data_dir}. "
            "Run download_esci_data() first."
        )
    
    print("Loading ESCI dataset...")
    
    df_examples = pd.read_parquet(examples_path)
    df_products = pd.read_parquet(products_path)
    
    print(f"Examples shape: {df_examples.shape}")
    print(f"Products shape: {df_products.shape}")
    
    df_merged = pd.merge(
        df_examples,
        df_products,
        how='left',
        left_on=['product_locale', 'product_id'],
        right_on=['product_locale', 'product_id']
    )
    
    print(f"Merged dataset shape: {df_merged.shape}")
    return df_merged


def filter_esci_data(df):
    """Apply filtering to match original notebook: small_version=1, then US locale and exact matches"""
    print(f"Original dataset shape: {df.shape}")
    
    # First filter for small_version (task 1 criteria)
    df_small = df[df['small_version'] == 1].copy()
    print(f"After small_version filter: {df_small.shape}")
    
    # Then filter for US locale and exact matches
    df_filtered = df_small[
        (df_small['product_locale'] == 'us') & 
        (df_small['esci_label'] == 'E')  # Exact matches only
    ].copy()
    
    print(f"After US/exact filter: {df_filtered.shape}")
    
    # Drop columns like original notebook
    drop_columns = ['product_locale', 'esci_label', 'small_version', 'large_version']
    df_filtered = df_filtered.drop(columns=drop_columns, errors='ignore')
    df_filtered = df_filtered.reset_index(drop=True)
    
    print(f"Final filtered shape: {df_filtered.shape}")
    print(f"Unique queries: {df_filtered['query'].nunique()}")
    
    return df_filtered 