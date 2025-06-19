import os
import hashlib
from pypdf import PdfReader
from chatbot.rag import RAGProcessor
from pathlib import Path

class PDFHandler:
    def __init__(self):
        self.rag_processor = RAGProcessor()
        self.processed_files = {}

    def process_pdf(self, file):
        try:
            # Extract text from PDF
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            
            # Process through RAG
            vector_store = self.rag_processor.process_document(text)
            return vector_store, True
            
        except Exception as e:
            raise Exception(f"PDF processing failed: {str(e)}")