from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_DEPLOYMENT_NAME: str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_ENDPOINT: Optional[str] = None
    LINKEDIN_POSTS_CSV_PATH: str = "app/db/linkedin_multiple_posts.csv"
    HOOKS_CSV_PATH: str = "app/db/hooks.csv"
    FRAMEWORKS_CSV_PATH: str = "app/db/frameworks.csv"
    CTA_CSV_PATH: str = "app/db/cta.csv"
    FAISS_INDEX_PATH: str = "data/faiss_index"
    class Config:
        env_file = ".env"

settings = Settings()