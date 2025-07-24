# Semantic Search System

A production-ready semantic search system built for e-commerce product discovery using the Amazon ESCI dataset.

## 🚀 Key Features

- **Hybrid Search**: Combines semantic similarity with keyword matching for optimal relevance
- **Vector Database**: High-performance search using LanceDB
- **Secondary Ranking**: Business logic re-ranking with brand preferences and title matching
- **Production Architecture**: Modular, testable, and configurable codebase
- **Comprehensive Evaluation**: HITS@K and MRR metrics with systematic testing

## 📊 Performance

### Transformer Model Comparison

Comprehensive evaluation of different transformer models and search strategies on 519 products, 53 unique queries from Amazon ESCI:

| Approach | Model | Dims | HITS@1 | HITS@5 | HITS@10 | MRR | Speed (s/query) |
|----------|-------|------|--------|--------|---------|-----|-----------------|
| miniLM Semantic | all-MiniLM-L6-v2 | 384 | 0.962 | 0.981 | 0.981 | 0.972 | ~0.019 |
| miniLM Hybrid | all-MiniLM-L6-v2 | 384 | 0.962 | 0.981 | 0.981 | 0.972 | ~0.018 |
| mpnet Semantic | all-mpnet-base-v2 | 768 | 0.943 | 0.981 | 1.000 | 0.962 | ~0.038 |
| mpnet Hybrid | all-mpnet-base-v2 | 768 | 1.000 | 1.000 | 1.000 | 1.000 | ~0.039 |

**Key Findings:**
- **mpnet Hybrid** achieves perfect scores across all metrics
- **miniLM Semantic** offers the best speed/accuracy/simplicity balance  
- **Hybrid search** provides significant boost for mpnet (+0.057 HITS@1)
- **miniLM approaches** show nearly identical accuracy and speed, making semantic preferable for speed

**Recommendations:**
- **For speed + accuracy + simplicity balance**: miniLM Semantic (simple, fast, effective)
- **For maximum accuracy**: mpnet Hybrid (perfect scores, ~2.7x slower)

*Results from `notebooks/06_transformer_comparison.ipynb` - run notebook to update with latest values*

## 🏗️ Architecture

```
src/
├── data/           # Data loading and preprocessing
├── models/         # Embedding creation (TF-IDF, transformers)
├── evaluation/     # Metrics and evaluation framework
├── search/         # Hybrid search and ranking systems
├── text_processing/# Text cleaning and combination strategies
└── utils/          # Vector database and configuration management

demo/               # Demo outputs (gitignored, safe to delete)
├── processed/      # Demo processed data
└── vector_dbs/     # Demo vector databases
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd semantic-search
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (required for text processing):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## 🎯 Quick Start

Run the complete semantic search pipeline:

```bash
python main.py
```

For quick testing with a smaller dataset:
```bash
python main.py --quick
```

**Safe Testing (Won't Overwrite Existing Work):**
```bash
# Full demo with 500 rows, 50 queries (recommended)
python main.py --config demo_config.yaml

# Quick demo with 200 rows, 20 queries  
python main.py --config demo_config.yaml --quick

# Test complete pipeline including data download
python main.py --config demo_config.yaml --quick --force-download

# Try different data subsets with different random seeds
python main.py --config demo_config.yaml --seed 42
python main.py --config demo_config.yaml --seed 123 --quick
```

**What happens:** Creates a `demo/` folder that can be safely deleted/gitignored. Shares raw data downloads with main project to avoid duplication.

**Command Line Options:**
```bash
# See all options
python main.py --help

# Common combinations
python main.py --config demo_config.yaml --seed 42 --quick
python main.py --config demo_config.yaml --skip-demo --skip-eval  # Setup only
python main.py --config demo_config.yaml --force-download --quick # Test complete pipeline
```

## ⚙️ Configuration

The system uses YAML configuration for easy customization:

```yaml
# config.yaml
model_name: "all-MiniLM-L6-v2"
hybrid_alpha: 0.7    # Semantic weight
hybrid_beta: 0.3     # Keyword weight
keywords_type: "light"
target_rows: 500
target_queries: 50
```

## 📝 Usage Examples

### Basic Search

```python
from src.search import HybridSearcher
from src.utils import load_config

# Load configuration
config = load_config('config.yaml')

# Initialize searcher
searcher = HybridSearcher()
searcher.fit_with_vector_db('src/vector_databases/semantic_search')

# Search
results = searcher.search("coffee maker", top_k=5)
print(results[['product_title', 'score']])
```

### With Secondary Ranking

```python
from src.search import SecondaryRanker

# Initialize ranker
ranker = SecondaryRanker()
ranker.add_brand_preference(['Cuisinart', 'Hamilton Beach'], boost=0.15)

# Apply ranking
ranked_results = ranker.rank(results, query="coffee maker")
```

### Custom Text Processing

```python
from src.text_processing.combining import create_combined_text_v1

# Create combined text with baseline strategy
df['combined_text_v1'] = df.apply(
    lambda row: create_combined_text_v1(row, max_chars=2000), axis=1
)
```

## 📈 Evaluation Framework

The system includes comprehensive evaluation metrics:

```python
from src.evaluation.metrics import evaluate_search_system

# Evaluate any search function
metrics = evaluate_search_system(
    df=test_data,
    search_function=my_search_function,
    k_values=[1, 5, 10]
)

print(f"HITS@1: {metrics['hits_at_1']:.3f}")
print(f"MRR: {metrics['mrr']:.3f}")
```

## 🔧 Command Line Options

```bash
python main.py [options]

Options:
  --config PATH        Path to config file (default: config.yaml)
  --quick             Use smaller dataset for testing (200 rows, 20 queries)
  --seed INT          Random seed for data sampling (overrides config)
  --skip-demo         Skip search demonstration
  --skip-eval         Skip evaluation
  --force-download    Force re-download of data
```

## 📁 Project Structure

### Core Modules

- **`src/data/`**: Data loading, filtering, and sampling utilities
- **`src/models/`**: TF-IDF and transformer embedding creation
- **`src/search/`**: Hybrid search, semantic search, and ranking algorithms
- **`src/evaluation/`**: Comprehensive evaluation metrics (HITS@K, MRR)
- **`src/text_processing/`**: Text cleaning, combination strategies, and keyword extraction
- **`src/utils/`**: Vector database management and configuration

### Key Files

- **`main.py`**: Production pipeline demonstration
- **`config.yaml`**: System configuration
- **`requirements.txt`**: Python dependencies
- **`notebooks/`**: Jupyter notebooks showing development process

## 🧪 Development Process

This project follows a systematic ML development approach:

1. **Data Exploration** (`01_data_exploration.ipynb`): Dataset analysis and sampling
2. **Text Processing** (`02_text_processing.ipynb`): Cleaning and keyword extraction
3. **Embedding Comparison** (`03_embedding_comparison.ipynb`): TF-IDF vs transformers
4. **Vector Database** (`04_vector_database.ipynb`): LanceDB implementation
5. **Hybrid Search** (`05_hybrid_search.ipynb`): Final system with ranking
6. **Transformer Comparison** (`06_transformer_comparison.ipynb`): miniLM vs mpnet evaluation

## 🎯 Key Technical Decisions

### Text Combination Strategy
- **Winner**: Original order (title + description + brand + bullets + color)
- **Improvement**: +1.9% HITS@1 over metadata-first ordering

### Hybrid Search Weights
- **Optimal**: α=0.7 (semantic), β=0.3 (keyword)
- **Result**: +1.9% HITS@1 improvement over pure semantic

### Keywords
- **Type**: Light cleaning (stopword removal, no lemmatization)
- **Performance**: Equal to moderate cleaning, faster processing

### Imputation
- **Decision**: No imputation needed
- **Reason**: +0.000 improvement over baseline


## 🤝 Contributing

This project demonstrates ML engineering best practices:

1. **Systematic Experimentation**: Jupyter notebooks document the full development process
2. **Production Architecture**: Clean separation of concerns with modular design
3. **Comprehensive Testing**: Evaluation framework with multiple metrics
4. **Configuration Management**: YAML-based configuration with validation
5. **Documentation**: Clear README, docstrings, and code comments

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 About

This semantic search system was built to demonstrate advanced ML engineering skills.  It showcases:

- **Production ML Systems**: End-to-end pipeline from data to deployment
- **Advanced Search**: Hybrid semantic-keyword search with business logic
- **ML Engineering**: Systematic experimentation, evaluation, and productionalization
- **Code Quality**: Clean, testable, maintainable Python architecture

View original repo here: https://github.com/amazon-science/esci-data
