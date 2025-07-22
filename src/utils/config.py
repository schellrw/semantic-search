import os
import json
import yaml
from dataclasses import dataclass, asdict


@dataclass
class SemanticSearchConfig:
    """Configuration class for semantic search system"""
    
    data_dir: str = "data"
    raw_data_dir: str = "data/raw" 
    processed_data_dir: str = "data/processed"
    target_queries: int = 50
    target_rows: int = 500
    random_seed: int = 2025
    
    model_name: str = "all-MiniLM-L6-v2"
    max_chars: int = 2000
    text_combination_strategy: str = "v1"
    
    tfidf_max_features: int = 1000
    tfidf_ngram_range: tuple = (1, 2)
    
    vector_db_dir: str = "src/vector_databases"
    vector_db_name: str = "semantic_search"
    table_name: str = "products"
    
    hybrid_alpha: float = 0.7
    hybrid_beta: float = 0.3
    keyword_threshold: float = 0.1
    keywords_type: str = "light"
    
    brand_boost: float = 0.1
    title_match_boost: float = 0.05
    min_title_overlap: int = 2
    
    eval_k_values: list = None
    
    def __post_init__(self):
        if self.eval_k_values is None:
            self.eval_k_values = [1, 5, 10]
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
    
    @classmethod  
    def from_json(cls, json_path):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self):
        return asdict(self)
    
    def to_json(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved config to {json_path}")
    
    def to_yaml(self, yaml_path):
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        print(f"Saved config to {yaml_path}")
    
    def get_keywords_path(self):
        return f"src/text_processing/artifacts/{self.keywords_type}_keywords.json"
    
    def get_vector_db_path(self):
        return os.path.join(self.vector_db_dir, self.vector_db_name)
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        return self
    
    def validate(self):
        """Validate configuration settings"""
        errors = []
        
        if not self.model_name:
            errors.append("Model name cannot be empty")
        
        if self.target_queries <= 0:
            errors.append("Target queries must be positive")
        
        if self.target_rows <= 0:
            errors.append("Target rows must be positive")
        
        if not (0 <= self.hybrid_alpha <= 1):
            errors.append("Hybrid alpha must be between 0 and 1")
        
        if not (0 <= self.hybrid_beta <= 1):
            errors.append("Hybrid beta must be between 0 and 1")
        
        if abs(self.hybrid_alpha + self.hybrid_beta - 1.0) > 0.001:
            errors.append("Hybrid alpha + beta must equal 1.0")
        
        if self.keywords_type not in ['light', 'moderate', 'raw']:
            errors.append(f"Invalid keywords type: {self.keywords_type}")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


def load_config(config_path=None):
    """Load configuration from file or create default"""
    if config_path is None:
        default_paths = ['config.yaml', 'config.json', 'src/config.yaml', 'src/config.json']
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return SemanticSearchConfig.from_yaml(config_path)
        elif config_path.endswith('.json'):
            return SemanticSearchConfig.from_json(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
    else:
        print("No config file found, using default configuration")
        return SemanticSearchConfig()


def create_default_config(output_path="config.yaml"):
    """Create a default configuration file"""
    config = SemanticSearchConfig()
    
    if output_path.endswith('.yaml') or output_path.endswith('.yml'):
        config.to_yaml(output_path)
    elif output_path.endswith('.json'):
        config.to_json(output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_path}")


def get_env_config():
    """Get configuration from environment variables"""
    env_config = {}
    
    env_mapping = {
        'MODEL_NAME': 'model_name',
        'VECTOR_DB_DIR': 'vector_db_dir',
        'DATA_DIR': 'data_dir',
        'HYBRID_ALPHA': 'hybrid_alpha',
        'HYBRID_BETA': 'hybrid_beta',
        'KEYWORDS_TYPE': 'keywords_type',
        'TEXT_STRATEGY': 'text_combination_strategy'
    }
    
    for env_var, config_key in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None:
            if config_key in ['hybrid_alpha', 'hybrid_beta', 'keyword_threshold', 'brand_boost']:
                env_config[config_key] = float(value)
            elif config_key in ['target_queries', 'target_rows', 'random_seed', 'max_chars']:
                env_config[config_key] = int(value)
            else:
                env_config[config_key] = value
    
    return env_config 