from .cleaning import simple_clean_text

def combine_product_text(row, max_chars=2000, use_imputation=False):
    """Combine product text fields into single string for embedding."""
    # Get text fields
    title = simple_clean_text(row['product_title'])
    description = simple_clean_text(row['product_description'])
    bullet_point = simple_clean_text(row['product_bullet_point'])
    brand = simple_clean_text(row['product_brand'])
    color = simple_clean_text(row['product_color'])
    
    # Optional imputation (for later comparison)
    if use_imputation:
        # Use bullet_point for missing description
        if not description and bullet_point:
            description = bullet_point
        # Use description for missing bullet_point (truncated)
        if not bullet_point and description:
            bullet_point = description[:500]
    
    # Combine with clear separators
    components = []
    if title:
        components.append(f"Title: {title}")
    if description:
        components.append(f"Description: {description}")
    if bullet_point:
        components.append(f"Bullets: {bullet_point}")
    if brand:
        components.append(f"Brand: {brand}")
    if color:
        components.append(f"Color: {color}")
    
    combined = " | ".join(components)
    
    # Truncate if too long
    if len(combined) > max_chars:
        combined = combined[:max_chars].rsplit(' ', 1)[0] + "..."
    
    return combined
