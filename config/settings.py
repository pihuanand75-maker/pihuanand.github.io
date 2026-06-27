"""
Central configuration using environment variables and .env file.
All config is accessed via: from config import settings
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        # OpenRouter
        self.openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_base_url: str = "https://openrouter.ai/api/v1"
        self.site_url: str = os.getenv("SITE_URL", "http://localhost:8501")
        self.site_name: str = os.getenv("SITE_NAME", "RAG SQL Agent")

        # Model defaults
        self.default_model: str = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")
        self.sql_generation_temperature: float = 0.05
        self.interpretation_temperature: float = 0.3

        # Agent behavior
        self.max_sql_retries: int = int(os.getenv("MAX_RETRIES", "3"))
        self.max_rows_in_context: int = int(os.getenv("MAX_ROWS_IN_CONTEXT", "50"))
        self.max_tokens_sql: int = 1000
        self.max_tokens_interpretation: int = 1500
        self.max_tokens_repair: int = 1000

        # RAG
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./vector_store_db")
        self.schema_top_k: int = 5

        # Data loading
        self.max_preview_rows: int = 100
        self.max_file_size_mb: int = 200

settings = Settings()

# Available OpenRouter models (curated list)
AVAILABLE_MODELS = {
    "GPT-4o Mini (Fast & Cheap)":         "openai/gpt-4o-mini",
    "GPT-4o (Best OpenAI)":               "openai/gpt-4o",
    "Claude 3.5 Sonnet (Best Anthropic)": "anthropic/claude-3.5-sonnet",
    "Claude 3 Haiku (Ultra Fast)":        "anthropic/claude-3-haiku",
    "Gemini Flash 1.5":                   "google/gemini-flash-1.5",
    "Gemini Pro 1.5":                     "google/gemini-pro-1.5",
    "Llama 3.1 70B Instruct":             "meta-llama/llama-3.1-70b-instruct",
    "Qwen 2.5 72B Instruct":              "qwen/qwen-2.5-72b-instruct",
}
