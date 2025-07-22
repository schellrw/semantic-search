import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define functions to allow for different levels of cleaning
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Cleaning functions for keywords
def clean_raw(text):
    """Minimal text cleaning for keywords."""
    if pd.isna(text):
        return []
    return str(text).lower().split()

def clean_light(text):
    """Remove punctuation and symbols."""
    if pd.isna(text):
        return []
    text = re.sub(r'[^\w\s]', '', str(text).lower())  # remove punctuation and symbols
    # text = re.sub(r'[^a-z\s]', '', str(text).lower())  # commented out to allow for numbers
    tokens = text.split()
    return [w for w in tokens if w not in stop_words]

def clean_moderate(text):
    """Remove stopwords and lemmatize."""
    if pd.isna(text):
        return []
    tokens = clean_light(text)
    return [lemmatizer.lemmatize(w) for w in tokens]

# Text processing functions for notebooks (with consistent naming)
def clean_text_light(text):
    """Remove punctuation and symbols, return string."""
    if pd.isna(text):
        return ""
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(filtered_tokens)

def clean_text_moderate(text):
    """Remove stopwords and lemmatize, return string."""
    if pd.isna(text):
        return ""
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    return ' '.join(lemmatized_tokens)


# Original simple cleaning function for baseline - not used anymore
def simple_clean_text(text):
    """Minimal text cleaning for baseline."""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()
