import numpy as np
# import pandas as pd

# Evaluation metrics for search systems
def calculate_hits_at_k(search_results, relevant_product_ids, k):
    """Calculate HITS@K: Did any relevant products appear in top K results?"""
    top_k_product_ids = search_results.head(k)['product_id'].tolist()
    relevant_found = len(set(top_k_product_ids).intersection(set(relevant_product_ids)))
    return 1 if relevant_found > 0 else 0

def calculate_mrr(search_results, relevant_product_ids):
    """Calculate Mean Reciprocal Rank"""
    for rank, product_id in enumerate(search_results['product_id'].tolist(), 1):
        if product_id in relevant_product_ids:
            return 1.0 / rank
    return 0.0

def evaluate_search_system(df, search_function, *search_args, k_values=[1, 5, 10]):
    """Evaluate search system using HITS@K and MRR"""
    unique_queries = df['query'].unique()
    
    results = {f'hits_at_{k}': [] for k in k_values}
    results['mrr'] = []
    
    print(f"Evaluating on {len(unique_queries)} unique queries...")
    
    for query in unique_queries:
        # Find relevant products for this query
        relevant_products = df[df['query'] == query]['product_id'].tolist()
        
        # Get search results
        search_results = search_function(query, df, *search_args, top_k=max(k_values))
        
        # Calculate metrics
        for k in k_values:
            hits = calculate_hits_at_k(search_results, relevant_products, k)
            results[f'hits_at_{k}'].append(hits)
        
        mrr = calculate_mrr(search_results, relevant_products)
        results['mrr'].append(mrr)
    
    # Calculate averages
    avg_results = {}
    for metric, values in results.items():
        avg_results[metric] = np.mean(values)
    
    return avg_results


# def calculate_precision_at_k(search_results: pd.DataFrame, relevant_product_ids: List[str], k: int) -> float:
#     """
#     Calculate Precision@K: What fraction of top K results are relevant?
    
#     Args:
#         search_results: DataFrame with search results
#         relevant_product_ids: List of relevant product IDs
#         k: Number of top results to consider
        
#     Returns:
#         Precision@K value between 0.0 and 1.0
#     """
#     if len(search_results) == 0 or k == 0:
#         return 0.0
        
#     top_k_product_ids = search_results.head(k)['product_id'].tolist()
#     relevant_found = len(set(top_k_product_ids).intersection(set(relevant_product_ids)))
#     return relevant_found / min(k, len(top_k_product_ids))


# def calculate_recall_at_k(search_results: pd.DataFrame, relevant_product_ids: List[str], k: int) -> float:
#     """
#     Calculate Recall@K: What fraction of relevant items are in top K results?
    
#     Args:
#         search_results: DataFrame with search results
#         relevant_product_ids: List of relevant product IDs
#         k: Number of top results to consider
        
#     Returns:
#         Recall@K value between 0.0 and 1.0
#     """
#     if len(relevant_product_ids) == 0:
#         return 0.0
        
#     if len(search_results) == 0:
#         return 0.0
        
#     top_k_product_ids = search_results.head(k)['product_id'].tolist()
#     relevant_found = len(set(top_k_product_ids).intersection(set(relevant_product_ids)))
#     return relevant_found / len(relevant_product_ids)


# def evaluate_search_system(
#     df: pd.DataFrame, 
#     search_function: Callable,
#     *search_args,
#     k_values: List[int] = [1, 5, 10],
#     verbose: bool = True
# ) -> Dict[str, float]:
#     """
#     Comprehensive evaluation of a search system using multiple metrics.
    
#     Args:
#         df: Dataset with 'query' and 'product_id' columns
#         search_function: Function that takes (query, df, *search_args, top_k) and returns results
#         *search_args: Additional arguments to pass to search_function
#         k_values: List of K values to evaluate (for HITS@K, Precision@K, etc.)
#         verbose: Whether to print progress information
        
#     Returns:
#         Dictionary with average metric values across all queries
#     """
#     unique_queries = df['query'].unique()
    
#     # Initialize results storage
#     results = {}
#     for k in k_values:
#         results[f'hits_at_{k}'] = []
#         results[f'precision_at_{k}'] = []
#         results[f'recall_at_{k}'] = []
#     results['mrr'] = []
    
#     if verbose:
#         print(f"Evaluating on {len(unique_queries)} unique queries...")
    
#     for i, query in enumerate(unique_queries):
#         if verbose and (i + 1) % 10 == 0:
#             print(f"  Processed {i + 1}/{len(unique_queries)} queries...")
            
#         # Find relevant products for this query
#         relevant_products = df[df['query'] == query]['product_id'].tolist()
        
#         # Get search results
#         search_results = search_function(query, df, *search_args, top_k=max(k_values))
        
#         # Calculate metrics for each K value
#         for k in k_values:
#             hits = calculate_hits_at_k(search_results, relevant_products, k)
#             precision = calculate_precision_at_k(search_results, relevant_products, k)
#             recall = calculate_recall_at_k(search_results, relevant_products, k)
            
#             results[f'hits_at_{k}'].append(hits)
#             results[f'precision_at_{k}'].append(precision)
#             results[f'recall_at_{k}'].append(recall)
        
#         # Calculate MRR
#         mrr = calculate_mrr(search_results, relevant_products)
#         results['mrr'].append(mrr)
    
#     # Calculate averages
#     avg_results = {}
#     for metric, values in results.items():
#         avg_results[metric] = np.mean(values)
    
#     return avg_results


# def print_evaluation_results(results: Dict[str, float], method_name: str) -> None:
#     """
#     Pretty print evaluation results in a standardized format.
    
#     Args:
#         results: Dictionary of metric results from evaluate_search_system
#         method_name: Name of the method being evaluated
#     """
#     print(f"\n--- {method_name} Performance ---")
    
#     # Print primary metrics (HITS@K and MRR)
#     for metric, value in results.items():
#         if metric.startswith('hits_at_'):
#             k = metric.split('_')[-1]
#             print(f"HITS@{k}:  {value:.3f}")
    
#     if 'mrr' in results:
#         print(f"MRR:     {results['mrr']:.3f}")
    
#     # Print additional metrics if available
#     precision_metrics = {k: v for k, v in results.items() if k.startswith('precision_at_')}
#     if precision_metrics:
#         print("\nPrecision metrics:")
#         for metric, value in precision_metrics.items():
#             k = metric.split('_')[-1]
#             print(f"  Precision@{k}: {value:.3f}")
    
#     recall_metrics = {k: v for k, v in results.items() if k.startswith('recall_at_')}
#     if recall_metrics:
#         print("Recall metrics:")
#         for metric, value in recall_metrics.items():
#             k = metric.split('_')[-1]
#             print(f"  Recall@{k}: {value:.3f}")


# def create_evaluation_summary(
#     method_results: Dict[str, Dict[str, float]], 
#     save_path: str = None
# ) -> pd.DataFrame:
#     """
#     Create a summary comparison table of multiple evaluation methods.
    
#     Args:
#         method_results: Dict where keys are method names and values are evaluation results
#         save_path: Optional path to save the comparison table as CSV
        
#     Returns:
#         DataFrame with comparison of all methods
#     """
#     summary_data = []
    
#     for method_name, results in method_results.items():
#         row = {'Method': method_name}
#         row.update(results)
#         summary_data.append(row)
    
#     summary_df = pd.DataFrame(summary_data)
    
#     # Reorder columns for better readability
#     primary_cols = ['Method']
#     hits_cols = [col for col in summary_df.columns if col.startswith('hits_at_')]
#     other_cols = [col for col in summary_df.columns if col not in primary_cols + hits_cols]
    
#     column_order = primary_cols + sorted(hits_cols) + sorted(other_cols)
#     summary_df = summary_df[column_order]
    
#     if save_path:
#         summary_df.to_csv(save_path, index=False)
#         print(f"Evaluation summary saved to: {save_path}")
    
#     return summary_df 