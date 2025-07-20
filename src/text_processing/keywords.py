from collections import Counter

# Create keyword extraction function
def extract_keywords(df, fields, cleaner_fn):
    """Extract keywords from product text fields."""
    keyword_counter = Counter()

    for _, row in df.iterrows():
        query_tokens = set(cleaner_fn(row['query']))
        
        for field in fields:
            field_tokens = set(cleaner_fn(row[field]))
            shared = query_tokens.intersection(field_tokens)
            keyword_counter.update(shared)
    
    return keyword_counter