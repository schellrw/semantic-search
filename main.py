#!/usr/bin/env python3
"""
Semantic Search System - Production Demo

This script demonstrates the complete semantic search pipeline.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import json

sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import load_config
from src.utils.pipeline import (
    setup_data_pipeline, 
    setup_embeddings_and_vector_db, 
    setup_search_system, 
    run_evaluation
)


def demonstrate_search(hybrid_searcher, secondary_ranker):
    """Demonstrate search capabilities"""
    print("\n" + "=" * 60)
    print("STEP 4: SEARCH DEMONSTRATION")
    print("=" * 60)
    
    demo_queries = [
        "coffee maker",
        "single serve coffee maker", 
        "black dress women",
        "bluetooth headphones"
    ]
    
    print("Demonstrating search capabilities...")
    
    for query in demo_queries:
        print(f"\n--- Query: '{query}' ---")
        
        results = hybrid_searcher.search(query, top_k=3)
        print("Hybrid Search Results:")
        for idx, row in results.iterrows():
            semantic_score = row.get('semantic_score', 0)
            keyword_score = row.get('keyword_score', 0)
            hybrid_score = row.get('hybrid_score', row['score'])
            print(f"  {idx + 1}. {row['product_title'][:60]}...")
            print(f"     Semantic: {semantic_score:.3f}, Keyword: {keyword_score:.3f}, Hybrid: {hybrid_score:.3f}")
        
        if "coffee" in query.lower():
            print("\nWith Brand Boost (Cuisinart):")
            secondary_ranker.add_brand_preference(['Cuisinart'], 0.15)
            ranked_results = secondary_ranker.rank(results, query)
            for idx, row in ranked_results.iterrows():
                boost_indicator = " [BOOSTED]" if 'cuisinart' in row['product_title'].lower() else ""
                print(f"  {idx + 1}. {row['product_title'][:60]}...{boost_indicator}")
                print(f"     Score: {row['score']:.3f}")
            secondary_ranker.clear_preferences()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Semantic Search System Demo")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--force-download", action="store_true", help="Force download of data")
    parser.add_argument("--skip-demo", action="store_true", help="Skip search demonstration")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick mode: smaller dataset (200 rows, 20 queries)")
    parser.add_argument("--seed", type=int, help="Random seed for data sampling (overrides config)")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Override random seed if provided
    if args.seed is not None:
        config.random_seed = args.seed
        print(f"Using random seed: {config.random_seed}")
    
    if args.quick:
        config.target_queries = 20
        config.target_rows = 200
        print("Quick mode: Using smaller dataset for faster execution")
    
    if not config.validate():
        print("Configuration validation failed. Please fix errors and try again.")
        return 1
    
    print("SEMANTIC SEARCH SYSTEM - PRODUCTION DEMO")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Target dataset: {config.target_rows} rows, {config.target_queries} queries")
    print(f"Random seed: {config.random_seed}")
    print(f"Hybrid weights: α={config.hybrid_alpha}, β={config.hybrid_beta}")
    print(f"Keywords: {config.keywords_type}")
    print(f"Database: {config.vector_db_name} (demo-safe)")
    
    try:
        df_sample = setup_data_pipeline(config, args.force_download)
        model, db, table, db_manager = setup_embeddings_and_vector_db(df_sample, config)
        hybrid_searcher, secondary_ranker = setup_search_system(model, db, table, config)
        
        if not args.skip_demo:
            demonstrate_search(hybrid_searcher, secondary_ranker)
        
        if not args.skip_eval:
            metrics = run_evaluation(df_sample, hybrid_searcher, config)
            
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            results = {
                'config': config.to_dict(),
                'metrics': metrics,
                'dataset_info': {
                    'num_rows': len(df_sample),
                    'num_queries': df_sample['query'].nunique()
                }
            }
            
            results_path = results_dir / "semantic_search_demo_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {results_path}")
        
        print("\n" + "=" * 60)
        print("SEMANTIC SEARCH SYSTEM DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Data loading and preprocessing pipeline")
        print("✓ Text processing and embedding creation") 
        print("✓ Vector database with LanceDB")
        print("✓ Hybrid search (semantic + keyword)")
        print("✓ Secondary ranking with business logic")
        print("✓ Systematic evaluation with HITS@K and MRR")
        print("✓ Production-ready modular architecture")
        
        return 0
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if 'db_manager' in locals():
            db_manager.close_all_connections()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 