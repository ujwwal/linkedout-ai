import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from ..core.config import settings

class VectorStoreService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            openai_api_key=settings.AZURE_OPENAI_API_KEY,
            openai_api_base=settings.AZURE_OPENAI_ENDPOINT,
            openai_api_type="azure",
            openai_api_version=settings.AZURE_OPENAI_API_VERSION
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
        documents = [Document(page_content=row['content'], metadata=row.to_dict()) 
                     for _, row in df.iterrows()]
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.vector_store.save_local(settings.FAISS_INDEX_PATH)

    def search_similar_posts(self, query, k=3):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)