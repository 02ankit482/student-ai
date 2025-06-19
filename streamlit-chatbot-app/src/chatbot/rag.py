from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os
from pathlib import Path

class RAGProcessor:
    def __init__(self, persist_directory=None):
        # Set default persist directory if none provided
        if persist_directory is None:
            project_root = Path(__file__).parent.parent.parent
            persist_directory = str(project_root / "data" / "vectorstore")
            
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(
            model="all-minilm",
            model_kwargs={"temperature": 0.0},
            base_url="http://localhost:11434"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        self._initialize_vector_store()
        
    def _initialize_vector_store(self):
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        except:
            self.vector_store = None

    def process_document(self, text: str):
        chunks = self.text_splitter.split_text(text)
        
        if not self.vector_store:
            self.vector_store = Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings
            )
        else:
            self.vector_store.add_texts(texts=chunks)
        
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)