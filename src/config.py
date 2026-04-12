import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Charger les variables du fichier .env
load_dotenv()

# On peut aussi définir ici des constantes globales
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

@dataclass
class Config:
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")  
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY")
    
    # Modèles
    MODEL_NAME: str = "gpt-4o-mini"  # Modèle rapide et économique pour les tâches de l'agent
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Chemins
    DATA_PATH: str = "data/"
    VECTORSTORE_PATH: str = "vectorstore/"
    
    # RAG Settings
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    
    # Agent Settings
    TEMPERATURE: float = 0.1
    MAX_ITERATIONS: int = 5
    
    @classmethod
    def validate(cls):
        """Vérifie que les clés essentielles sont présentes"""
        config = cls()
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY manquante")
        return config

CONFIG = Config.validate()