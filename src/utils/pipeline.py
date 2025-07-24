import os
import json
from pathlib import Path
import pandas as pd

from ..data.loader import download_esci_data, load_esci_data, filter_esci_data
from ..data.sampling import create_sample_dataset
from ..text_processing.combining import create_combined_text_v1
from ..search.hybrid import HybridSearcher
from ..search.semantic import SemanticSearcher
from ..search.ranking import SecondaryRanker
from .vector_db import VectorDatabaseManager, infer_model_from_db_name


def setup_data_pipeline(config, force_download=False):
    """Set up the data pipeline: download, load, filter, and sample"""
    print("=" * 60)
    print("STEP 1: DATA SETUP")
    print("=" * 60)
    
    os.makedirs(config.raw_data_dir, exist_ok=True)
    os.makedirs(config.processed_data_dir, exist_ok=True)
    
    examples_path = Path(config.raw_data_dir) / "examples.parquet"
    products_path = Path(config.raw_data_dir) / "products.parquet"
    
    if force_download or not (examples_path.exists() and products_path.exists()):
        print("Downloading ESCI dataset...")
        download_esci_data(config.raw_data_dir)
    else:
        print("Using existing raw data files")
    
    print("\nLoading and filtering data...")
    df_raw = load_esci_data(config.raw_data_dir)
    df_filtered = filter_esci_data(df_raw)
    
    print(f"\nCreating sample dataset (target: {config.target_rows} rows, {config.target_queries} queries)...")
    final_df = create_sample_dataset(
        df_filtered,
        target_queries=config.target_queries,
        target_rows=config.target_rows,
        random_seed=config.random_seed,
        prioritize='rows'  # Back to rows to match original 519/53 approach
    )
    
    print("Adding combined text field...")
    # final_df['combined_text_v1'] = final_df.apply(create_combined_text_v1, axis=1)
    final_df['combined_text_v1'] = final_df.apply(lambda row: create_combined_text_v1(row, max_chars=2000), axis=1)
    
    # Show a sample of the queries to help debug
    print(f"\nSample queries in dataset:")
    sample_queries = final_df['query'].unique()[:10]
    for i, query in enumerate(sample_queries, 1):
        print(f"  {i}. '{query}'")
    
    processed_path = Path(config.processed_data_dir) / "sample_dataset.parquet"
    final_df.to_parquet(processed_path)
    print(f"Saved processed dataset to {processed_path}")
    
    return final_df


def setup_embeddings_and_vector_db(df, config):
    """Set up embeddings and vector database"""
    print("\n" + "=" * 60)
    print("STEP 2: EMBEDDINGS & VECTOR DATABASE")
    print("=" * 60)
    
    db_manager = VectorDatabaseManager(config.vector_db_dir)
    
    # Check if database exists to determine if we should infer model name
    db_exists = db_manager.database_exists_with_data(config.vector_db_name, config.table_name)
    
    if db_exists:
        # Database exists - infer the correct model name from database name
        try:
            inferred_model = infer_model_from_db_name(config.vector_db_name)
            if inferred_model != config.model_name:
                print(f"Auto-detected model mismatch:")
                print(f"  Config model: {config.model_name}")
                print(f"  Database requires: {inferred_model}")
                print(f"  Using: {inferred_model} (auto-inferred)")
                model_name_to_use = inferred_model
            else:
                model_name_to_use = config.model_name
        except ValueError as e:
            print(f"Warning: {e}")
            print(f"Using config model name: {config.model_name}")
            model_name_to_use = config.model_name
    else:
        # Database doesn't exist - use config model name
        model_name_to_use = config.model_name
    
    # Use the improved method that properly handles existing databases
    model, db, table = db_manager.get_or_create_database(
        df,
        'combined_text_v1',
        config.vector_db_name,
        model_name_to_use,
        config.table_name
    )
    
    return model, db, table, db_manager


def setup_search_system(model, db, table, config, semantic_only=False):
    """Set up the search system with secondary ranking"""
    print("\n" + "=" * 60)
    print("STEP 3: SEARCH SYSTEM SETUP") 
    print("=" * 60)
    
    # Use the actual model name from the loaded model, not config
    actual_model_name = model.get_sentence_embedding_dimension()
    # Map dimensions back to model names for consistency
    if actual_model_name == 384:
        model_name_for_searcher = "all-MiniLM-L6-v2"
    elif actual_model_name == 768:
        model_name_for_searcher = "all-mpnet-base-v2"
    else:
        # Fallback to config model name
        model_name_for_searcher = config.model_name
    
    if semantic_only:
        print("Using semantic-only search mode")
        
        searcher = SemanticSearcher(model_name=model_name_for_searcher)
        searcher.db = db
        searcher.table = table
        searcher.model = model
        searcher.fitted = True
        
    else:
        print("Using hybrid search mode")
        
        keywords_path = config.get_keywords_path()
        print(f"Loading keywords from {keywords_path}")
        
        with open(keywords_path, 'r') as f:
            keywords_dict = json.load(f)
        
        print(f"Loaded {len(keywords_dict)} keywords")
        
        searcher = HybridSearcher(
            model_name=model_name_for_searcher,
            keywords_dict=keywords_dict,
            alpha=config.hybrid_alpha,
            beta=config.hybrid_beta,
            keyword_threshold=config.keyword_threshold
        )
        
        searcher.db = db
        searcher.table = table
        searcher.model = model
        searcher.fitted = True
    
    secondary_ranker = SecondaryRanker(
        brand_boost=config.brand_boost,
        title_match_boost=config.title_match_boost,
        min_title_overlap=config.min_title_overlap
    )
    
    print("Search system ready!")
    return searcher, secondary_ranker


def run_evaluation(df, searcher, config):
    """Run systematic evaluation"""
    print("\n" + "=" * 60)
    print("STEP 5: SYSTEM EVALUATION")
    print("=" * 60)
    
    from ..evaluation.metrics import evaluate_search_system
    
    def eval_search(query, df, *args, top_k=10):
        return searcher.search(query, top_k)
    
    print("Running systematic evaluation on all queries...")
    print(f"Evaluating {df['query'].nunique()} unique queries...")
    
    metrics = evaluate_search_system(
        df, 
        eval_search, 
        k_values=config.eval_k_values
    )
    
    print(f"\n--- FINAL EVALUATION RESULTS ---")
    print(f"HITS@1:  {metrics['hits_at_1']:.3f}")
    print(f"HITS@5:  {metrics['hits_at_5']:.3f}")
    print(f"HITS@10: {metrics['hits_at_10']:.3f}")
    print(f"MRR:     {metrics['mrr']:.3f}")
    
    return metrics 