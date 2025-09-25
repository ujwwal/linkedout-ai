from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from ..core.config import settings
from pydantic import SecretStr

class VectorStoreService:
    def __init__(self):
        self.index_dir = Path(settings.FAISS_INDEX_PATH)
        self.dataset_path = Path(settings.LINKEDIN_POSTS_CSV_PATH)
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            api_key=SecretStr(settings.AZURE_OPENAI_API_KEY),
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        self.vector_store = self._load_or_create_vector_store()
    
    def _load_or_create_vector_store(self):
        index_file = self.index_dir / "index.faiss"
        store_file = self.index_dir / "index.pkl"

        if index_file.exists() and store_file.exists():
            if not self.dataset_path.exists() or index_file.stat().st_mtime >= self.dataset_path.stat().st_mtime:
                return FAISS.load_local(str(self.index_dir), self.embeddings, allow_dangerous_deserialization=True)

        if self.dataset_path.exists():
            return self._build_vector_store_from_csv(self.dataset_path)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        return None

    def _build_vector_store_from_csv(self, csv_path: Path):
        if not csv_path.exists():
            return None

        df = pd.read_csv(csv_path)
        if df.empty:
            return None

        content_col_candidates = ["content", "post_content", "text", "body"]
        content_col = next((c for c in content_col_candidates if c in df.columns), None)
        if content_col is None:
            raise ValueError(
                f"CSV must contain one of the content columns: {content_col_candidates}. Found: {list(df.columns)}"
            )

        documents = []
        for _, row in df.iterrows():
            content = row.get(content_col)
            if pd.isna(content) or str(content).strip() == "":
                continue

            metadata = {
                key: ("" if pd.isna(value) else str(value))
                for key, value in row.items()
                if not pd.isna(value)
            }
            documents.append(
                Document(
                    page_content=str(content),
                    metadata=metadata
                )
            )

        if not documents:
            return None

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.vector_store.save_local(str(self.index_dir))
        return self.vector_store

    def load_posts_from_csv(self, csv_path):
        csv_path = Path(csv_path)
        return self._build_vector_store_from_csv(csv_path)

    def _ensure_vector_store(self):
        if self.vector_store is None:
            self.vector_store = self._load_or_create_vector_store()
        return self.vector_store

    def search_similar_posts(self, query, k=3):
        store = self._ensure_vector_store()
        if not store:
            return []
        return store.similarity_search(query, k=k)