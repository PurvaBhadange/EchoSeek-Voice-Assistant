import os
from typing import List, Union, Any
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file from root directory or current directory
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Voice-Enabled RAG Model"
    API_V1_STR: str = "/api/v1"
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # CORS Origins
    CORS_ORIGINS: Any = ["*"]
    
    # API Keys
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Model Configuration
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))
    TOP_K: int = int(os.getenv("TOP_K", 3))

settings = Settings()
