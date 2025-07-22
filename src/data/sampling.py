"""Data sampling utilities for creating assessment datasets."""

import pandas as pd
import numpy as np


def create_sample_dataset(
    df,
    target_queries=50,
    target_rows=500,
    random_seed=2025,
    prioritize='rows',
    max_queries=100
):
    """Create a sample dataset to meet both row and query targets.
    
    Replicates original notebook logic: keep trying queries until finding optimal balance.
    Only downsample if way over target (like 2x).
    """
    np.random.seed(random_seed)
    
    query_counts = df['query'].value_counts()
    unique_queries = query_counts[(query_counts >= 5) & (query_counts <= 20)].index.tolist()
    
    print(f"Total unique queries available: {len(unique_queries)}")
    print(f"Query product counts - Min: {query_counts.min()}, Max: {query_counts.max()}, Mean: {query_counts.mean():.1f}")
    
    if prioritize == 'rows':
        print(f"\nPrioritizing hitting {target_rows} rows target...")
        
        # Keep track of the best result that meets our target
        best_result = None
        current_queries = min(target_queries, len(unique_queries))
        
        # Try incrementally adding queries to find the sweet spot
        for attempt_queries in range(current_queries, min(max_queries + 1, len(unique_queries) + 1)):
            sample_queries = np.random.choice(unique_queries, size=attempt_queries, replace=False)
            df_candidate = df[df['query'].isin(sample_queries)].copy()
            
            print(f"  Trying {attempt_queries} queries: {df_candidate.shape[0]} total rows")
            
            # If we hit the target, save this as our best result
            if df_candidate.shape[0] >= target_rows:
                if best_result is None:
                    best_result = (attempt_queries, df_candidate.shape[0], df_candidate)
                    print(f"  * Found {attempt_queries} queries with {df_candidate.shape[0]} rows (>= {target_rows})")
                
                # Keep going for a few more tries to see if we get a better balance
                # (This mimics the original notebook behavior that got 519/53)
                continue_searching = attempt_queries < min(target_queries + 10, max_queries)
                if not continue_searching:
                    break
        
        if best_result is None:
            print(f"  Could not reach {target_rows} rows even with {max_queries} queries")
            # Use the last attempt
            final_df = df_candidate.reset_index(drop=True)
        else:
            _, best_rows, final_df = best_result
            print(f"  Using best result: {best_result[0]} queries with {best_rows} rows")
            
            # Only downsample if we have WAY too many rows (like 2x target)
            if best_rows > target_rows * 2:
                print(f"  Downsampling from {best_rows} to {target_rows} rows (way over target)")
                final_df = final_df.sample(n=target_rows, random_state=random_seed).reset_index(drop=True)
            else:
                final_df = final_df.reset_index(drop=True)
                
    elif prioritize == 'queries':
        print(f"\nPrioritizing hitting {target_queries} queries target...")
        
        n_queries = min(target_queries, len(unique_queries))
        sample_queries = np.random.choice(unique_queries, size=n_queries, replace=False)
        final_df = df[df['query'].isin(sample_queries)].copy()
        
        print(f"Sampled {n_queries} queries with {final_df.shape[0]} total rows")
        
        # Only downsample if WAY over target (like 2x)
        if final_df.shape[0] > target_rows * 2:
            print(f"Downsampling from {final_df.shape[0]} to {target_rows} rows (way over target)")
            rows_per_query = target_rows // n_queries
            remainder = target_rows % n_queries
            
            sampled_dfs = []
            queries_list = list(sample_queries)
            
            for i, query in enumerate(queries_list):
                query_df = final_df[final_df['query'] == query]
                n_rows_for_query = rows_per_query + (1 if i < remainder else 0)
                n_rows_for_query = min(n_rows_for_query, len(query_df))
                
                if n_rows_for_query > 0:
                    sampled_query_df = query_df.sample(n=n_rows_for_query, random_state=random_seed+i)
                    sampled_dfs.append(sampled_query_df)
            
            final_df = pd.concat(sampled_dfs, ignore_index=True)
        else:
            final_df = final_df.reset_index(drop=True)
    
    else:
        raise ValueError("prioritize must be either 'rows' or 'queries'")
    
    print(f"\nFinal sample dataset: {final_df.shape[0]} rows, {final_df['query'].nunique()} unique queries")
    
    product_cols = ['product_description', 'product_bullet_point', 'product_brand', 'product_color']
    print(f"\nMissing values in product columns:")
    for col in product_cols:
        if col in final_df.columns:
            missing_count = final_df[col].isna().sum()
            missing_pct = (missing_count / len(final_df)) * 100
            print(f"  {col}: {missing_count} ({missing_pct:.1f}%)")

    # Return just one dataframe (the final result)
    return final_df.reset_index(drop=True)


def add_query_overlap_metrics(df, text_columns):
    """Add query overlap metrics with product text fields"""
    def normalize(text):
        return str(text).lower().split()
    
    df_with_metrics = df.copy()
    
    for col in text_columns:
        col_suffix = col.replace("product_", "")
        
        contains_col = f'query_in_{col_suffix}'
        df_with_metrics[contains_col] = df_with_metrics.apply(
            lambda x: x['query'].lower() in str(x[col]).lower(), axis=1
        )

        count_col = f'overlap_count_{col_suffix}'
        ratio_q_col = f'overlap_ratio_query_{col_suffix}'
        jaccard_col = f'jaccard_sim_{col_suffix}'
        
        def compute_overlap(row):
            q_words = set(normalize(row['query']))
            p_words = set(normalize(row[col]))

            if not q_words or not p_words:
                return pd.Series([0, 0.0, 0.0])

            overlap = q_words.intersection(p_words)
            overlap_count = len(overlap)
            ratio_query = overlap_count / len(q_words)
            jaccard = len(overlap) / len(q_words.union(p_words))

            return pd.Series([overlap_count, ratio_query, jaccard])
        
        df_with_metrics[[count_col, ratio_q_col, jaccard_col]] = df_with_metrics.apply(
            compute_overlap, axis=1
        )
        
    return df_with_metrics


def simple_imputation_from_bullet_to_desc(df):
    """Use bullet_point text as description when description is missing"""
    df_imputed = df.copy()
    
    missing_desc = df_imputed['product_description'].isna()
    has_bullet = df_imputed['product_bullet_point'].notna()
    
    candidates = missing_desc & has_bullet
    print(f"Can impute description from bullet_point for {candidates.sum()} rows")
    
    df_imputed.loc[candidates, 'product_description'] = df_imputed.loc[candidates, 'product_bullet_point']
    
    return df_imputed


def simple_imputation_from_desc_to_bullet(df):
    """Use description text as bullet_point when bullet_point is missing"""
    df_imputed = df.copy()
    
    missing_bullet = df_imputed['product_bullet_point'].isna()
    has_desc = df_imputed['product_description'].notna()
    
    candidates = missing_bullet & has_desc
    print(f"Can impute bullet_point from description for {candidates.sum()} rows")
    
    df_imputed.loc[candidates, 'product_bullet_point'] = df_imputed.loc[candidates, 'product_description'].str[:500]
    
    return df_imputed 