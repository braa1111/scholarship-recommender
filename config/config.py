import os
from dataclasses import dataclass, field
from typing import Dict, Any

# =======================
# Model Configuration
# =======================
@dataclass
class ModelConfig:
    """Configuration for machine learning models"""

    # Collaborative Filtering
    CF_ALGORITHM: str = 'svd'
    CF_N_FACTORS: int = 100
    CF_N_EPOCHS: int = 20
    CF_LR_ALL: float = 0.005
    CF_REG_ALL: float = 0.02

    # NLP Models
    SENTENCE_MODEL: str = 'all-MiniLM-L6-v2'
    SPACY_MODEL: str = 'en_core_web_md'

    # Hybrid Weights
    CF_WEIGHT: float = 0.6
    NLP_WEIGHT: float = 0.4

    # Cold Start Handling
    COLD_START_THRESHOLD: int = 3
    COLD_START_CF_WEIGHT: float = 0.3

    # Evaluation
    TEST_SIZE: float = 0.2
    TOP_N_RECOMMENDATIONS: int = 10


# =======================
# Data Configuration
# =======================
@dataclass
class DataConfig:
    """Configuration for data handling"""

    # File paths
    RAW_DATA_PATH: str = 'data/raw'
    PROCESSED_DATA_PATH: str = 'data/processed'
    MODEL_SAVE_PATH: str = 'models/trained_models'

    # Dataset sizes
    DEFAULT_NUM_STUDENTS: int = 1000
    DEFAULT_NUM_SCHOLARSHIPS: int = 200

    # Data validation
    MIN_RATING: int = 1
    MAX_RATING: int = 5
    MIN_GPA: float = 2.0
    MAX_GPA: float = 4.0


# =======================
# Web Configuration
# =======================
@dataclass
class WebConfig:
    """Configuration for web application"""

    # Flask settings
    DEBUG: bool = True
    HOST: str = '0.0.0.0'
    PORT: int = 5000
    SECRET_KEY: str = 'scholarship_recommender_secret_key_2025'

    # API settings
    API_PREFIX: str = '/api/v1'
    MAX_RECOMMENDATIONS: int = 50
    DEFAULT_RECOMMENDATIONS: int = 10


# =======================
# App Configuration
# =======================
@dataclass
class AppConfig:
    """Main application configuration"""

    # Application info
    APP_NAME: str = "Scholarship Recommender System"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "AI-powered scholarship recommendation using hybrid collaborative filtering and NLP"

    # Authors
    AUTHORS: list = None
    UNIVERSITY: str = "University of Khartoum"
    DEPARTMENT: str = "Department of Information Technology"

    # Components (FIXED: using default_factory)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    web: WebConfig = field(default_factory=WebConfig)

    def __post_init__(self):
        if self.AUTHORS is None:
            self.AUTHORS = [
                "Baraa Alshiekh Mohammed Ahmed",
                "Malaz Abbas Hammad Abbas",
                "Ethar Kamaleldein Elneel Flamein"
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'app': {
                'name': self.APP_NAME,
                'version': self.VERSION,
                'description': self.DESCRIPTION,
                'authors': self.AUTHORS,
                'university': self.UNIVERSITY,
                'department': self.DEPARTMENT
            },
            'model': {
                'cf_algorithm': self.model.CF_ALGORITHM,
                'cf_n_factors': self.model.CF_N_FACTORS,
                'hybrid_weights': {
                    'cf_weight': self.model.CF_WEIGHT,
                    'nlp_weight': self.model.NLP_WEIGHT
                }
            },
            'web': {
                'debug': self.web.DEBUG,
                'host': self.web.HOST,
                'port': self.web.PORT
            }
        }


# =======================
# Global Configuration
# =======================
config = AppConfig()


def get_config():
    """Get application configuration"""
    return config


def update_config(new_config: Dict[str, Any]):
    """Update configuration with new values"""
    global config

    if 'model' in new_config:
        for key, value in new_config['model'].items():
            attr = key.upper()
            if hasattr(config.model, attr):
                setattr(config.model, attr, value)

    if 'web' in new_config:
        for key, value in new_config['web'].items():
            attr = key.upper()
            if hasattr(config.web, attr):
                setattr(config.web, attr, value)

    print("Configuration updated successfully")
