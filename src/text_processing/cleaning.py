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
    return str(text).lower().split()

def clean_light(text):
    """Remove punctuation and symbols."""
    text = re.sub(r'[^\w\s]', '', str(text).lower())  # remove punctuation and symbols
    # text = re.sub(r'[^a-z\s]', '', str(text).lower())  # commented out to allow for numbers
    tokens = text.split()
    return [w for w in tokens if w not in stop_words]

def clean_moderate(text):
    """Remove stopwords and lemmatize."""
    tokens = clean_light(text)
    return [lemmatizer.lemmatize(w) for w in tokens]


# Cleaning functions for dataframes
def simple_clean_text(text):
    """Minimal text cleaning for baseline."""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()
