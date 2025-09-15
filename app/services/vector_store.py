import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from ..core.config import settings
from pydantic import SecretStr

class VectorStoreService:
    def __init__(self):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            api_key=SecretStr(settings.AZURE_OPENAI_API_KEY),
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        self.vector_store = self._load_or_create_vector_store()
    
    def _load_or_create_vector_store(self):
        index_path = settings.FAISS_INDEX_PATH
        if os.path.exists(f"{index_path}/index.faiss"):
            return FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            os.makedirs(index_path, exist_ok=True)
            return None  # Will be built with your CSV loader below

    def load_posts_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        # Determine which column contains the post content
        content_col_candidates = ["content", "post_content", "text", "body"]
        content_col = next((c for c in content_col_candidates if c in df.columns), None)
        if content_col is None:
            raise ValueError(
                f"CSV must contain one of the content columns: {content_col_candidates}. Found: {list(df.columns)}"
            )
        documents = [
            Document(page_content=str(row[content_col]), metadata=row.to_dict())
            for _, row in df.iterrows()
        ]
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.vector_store.save_local(settings.FAISS_INDEX_PATH)

    def search_similar_posts(self, query, k=3):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)