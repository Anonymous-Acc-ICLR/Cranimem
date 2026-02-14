import os
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from typing import Literal, Optional
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

class Settings(BaseSettings):
    LLM_PROVIDER: Literal["groq", "gemini", "ollama", "huggingface", "local_hf"] = Field(default="groq")

    GROQ_API_KEY: Optional[SecretStr] = Field(default=None)
    GROQ_MODEL: str = Field(default="llama-3.1-8b-instant")

    GOOGLE_API_KEY: Optional[SecretStr] = Field(default=None)
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash")
    
    OLLAMA_MODEL: str = Field(default="Qwen2.5:7b")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")

    HUGGINGFACEHUB_API_TOKEN: Optional[SecretStr] = Field(default=None)
    HF_REPO_ID: str = Field(default="meta-llama/Llama-3.1-8B")

    HF_LOCAL_MODEL_PATH: Optional[str] = Field(default=None)
    HF_LOCAL_FILES_ONLY: bool = Field(default=False)
    HF_TRUST_REMOTE_CODE: bool = Field(default=False)
    HF_DEVICE_MAP: str = Field(default="auto")
    HF_TORCH_DTYPE: Literal["float16", "bfloat16", "float32"] = Field(default="float16")
    HF_MAX_NEW_TOKENS: int = Field(default=512)
    REASONING_MAX_CONTEXT_CHARS: Optional[int] = Field(default=None)


    NEO4J_URI: str = Field(default="neo4j+s://c1d2920c.databases.neo4j.io")
    NEO4J_USERNAME: str = Field(default="neo4j")
    NEO4J_PASSWORD: SecretStr = Field(...)


    GATING_MODE: Literal["focus", "explore", "default"] = Field(default="focus")
    CONSOLIDATION_INTERVAL: int = 5
    RELEVANCE_THRESHOLD: float = 0.25 
    REFLEX_PASS_THRESHOLD: float = 0.6

    MIN_CONSOLIDATION_SCORE: float = 0.2
    FREQUENCY_BONUS_STEP: float = 0.1
    MAX_FREQUENCY_BONUS: float = 1.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" 

settings = Settings()
