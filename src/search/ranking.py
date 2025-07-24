"""Secondary ranking functionality for business logic re-ranking."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


def apply_secondary_ranking(search_results, boost_brands=None, brand_boost=0.1):
    """Apply business logic secondary ranking to search results"""
    results = search_results.copy()
    
    if boost_brands:
        for brand in boost_brands:
            brand_mask = results['product_title'].str.contains(brand, case=False, na=False)
            results.loc[brand_mask, 'score'] += brand_boost
    
    if len(results) > 0:
        query_words = set()
        if 'query' in results.columns and not results['query'].empty:
            query_words = set(str(results.iloc[0]['query']).lower().split())
        
        for idx, row in results.iterrows():
            title_words = set(str(row.get('product_title', '')).lower().split())
            overlap = len(query_words.intersection(title_words))
            if overlap >= 2:
                results.loc[idx, 'score'] += 0.05 * overlap
    
    results = results.sort_values('score', ascending=False).reset_index(drop=True)
    return results


class SecondaryRanker:
    def __init__(self, brand_boost=0.1, title_match_boost=0.05, min_title_overlap=2):
        self.brand_boost = brand_boost
        self.title_match_boost = title_match_boost
        self.min_title_overlap = min_title_overlap
        self.preferred_brands = []
        self.ranking_rules = []
    
    def add_brand_preference(self, brands, boost=None):
        boost = boost or self.brand_boost
        for brand in brands:
            self.preferred_brands.append({
                'brand': brand,
                'boost': boost
            })
        print(f"Added {len(brands)} preferred brands with boost {boost}")
    
    def add_custom_rule(self, rule_name, condition_func, boost_func):
        self.ranking_rules.append({
            'name': rule_name,
            'condition': condition_func,
            'boost': boost_func
        })
        print(f"Added custom ranking rule: {rule_name}")
    
    def rank(self, search_results, query=None):
        """Apply secondary ranking to search results"""
        results = search_results.copy()
        
        # Get the dtype of the score column to ensure compatibility
        score_dtype = results['score'].dtype
        
        for brand_pref in self.preferred_brands:
            brand = brand_pref['brand']
            boost = brand_pref['boost']
            brand_mask = results['product_title'].str.contains(brand, case=False, na=False)
            results.loc[brand_mask, 'score'] += score_dtype.type(boost)
        
        if query and len(results) > 0:
            query_words = set(query.lower().split())
            
            for idx, row in results.iterrows():
                title_words = set(str(row.get('product_title', '')).lower().split())
                overlap = len(query_words.intersection(title_words))
                if overlap >= self.min_title_overlap:
                    title_boost = score_dtype.type(self.title_match_boost * overlap)
                    results.loc[idx, 'score'] += title_boost
        
        for rule in self.ranking_rules:
            for idx, row in results.iterrows():
                if rule['condition'](row):
                    boost = rule['boost'](row)
                    results.loc[idx, 'score'] += score_dtype.type(boost)
        
        results = results.sort_values('score', ascending=False).reset_index(drop=True)
        return results
    
    def explain_ranking(self, search_results, query=None, top_k=5):
        """Apply ranking with detailed explanations"""
        original_results = search_results.copy()
        ranked_results = self.rank(search_results, query)
        
        print(f"\nSecondary Ranking Explanation")
        print(f"Query: '{query}'")
        print(f"Brand boost: {self.brand_boost}")
        print(f"Title match boost: {self.title_match_boost} per word")
        print("-" * 80)
        
        for idx in range(min(top_k, len(ranked_results))):
            row = ranked_results.iloc[idx]
            product_id = row['product_id']
            
            orig_pos = original_results[original_results['product_id'] == product_id].index[0]
            orig_score = original_results.iloc[orig_pos]['score']
            new_score = row['score']
            
            print(f"Rank {idx + 1}: {row['product_title'][:60]}...")
            print(f"  Original: Rank {orig_pos + 1}, Score: {orig_score:.3f}")
            print(f"  New:      Rank {idx + 1}, Score: {new_score:.3f}")
            print(f"  Boost:    {new_score - orig_score:.3f}")
            
            boosts = []
            
            for brand_pref in self.preferred_brands:
                brand = brand_pref['brand']
                if brand.lower() in str(row['product_title']).lower():
                    boosts.append(f"Brand '{brand}': +{brand_pref['boost']:.3f}")
            
            if query:
                query_words = set(query.lower().split())
                title_words = set(str(row.get('product_title', '')).lower().split())
                overlap = len(query_words.intersection(title_words))
                if overlap >= self.min_title_overlap:
                    title_boost = self.title_match_boost * overlap
                    boosts.append(f"Title overlap ({overlap} words): +{title_boost:.3f}")
            
            if boosts:
                print(f"  Applied:  {', '.join(boosts)}")
            print()
        
        return ranked_results
    
    def clear_preferences(self):
        self.preferred_brands = []
        print("Cleared all brand preferences")
    
    def clear_rules(self):
        self.ranking_rules = []
        print("Cleared all custom ranking rules")
    
    def get_preferences(self) -> List[Dict[str, Any]]:
        """Get current brand preferences."""
        return self.preferred_brands.copy()
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get current custom rules."""
        return [{'name': rule['name']} for rule in self.ranking_rules]


class HybridSearchWithRanking:
    """Combines hybrid search with secondary ranking."""
    
    def __init__(
        self,
        hybrid_searcher,
        secondary_ranker: Optional[SecondaryRanker] = None
    ):
        """
        Initialize hybrid search with ranking.
        
        Args:
            hybrid_searcher: Fitted HybridSearcher instance
            secondary_ranker: SecondaryRanker instance
        """
        self.hybrid_searcher = hybrid_searcher
        self.secondary_ranker = secondary_ranker or SecondaryRanker()
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        apply_ranking: bool = True
    ) -> pd.DataFrame:
        """
        Perform hybrid search with optional secondary ranking.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            apply_ranking: Whether to apply secondary ranking
            
        Returns:
            DataFrame with search results
        """
        # Get initial hybrid results with more candidates for reranking
        initial_k = top_k * 2 if apply_ranking else top_k
        results = self.hybrid_searcher.search(query, initial_k)
        
        if apply_ranking:
            # Apply secondary ranking
            results = self.secondary_ranker.rank(results, query)
            # Return top-k after ranking
            results = results.head(top_k)
        
        return results
    
    def add_brand_boost(self, brands: List[str], boost: float = 0.1) -> None:
        """Add brand boosting to secondary ranker."""
        self.secondary_ranker.add_brand_preference(brands, boost)
    
    def explain_search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Perform search with detailed explanations."""
        print("=== HYBRID SEARCH WITH SECONDARY RANKING ===")
        
        # Show hybrid search explanation
        hybrid_results = self.hybrid_searcher.explain_search(query, top_k * 2)
        
        # Show secondary ranking explanation
        final_results = self.secondary_ranker.explain_ranking(hybrid_results, query, top_k)
        
        return final_results.head(top_k) 