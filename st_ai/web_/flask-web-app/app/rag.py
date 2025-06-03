import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pymupdf as fitz
from PIL import Image
import io
import pandas as pd
import numpy as np
import faiss
import nltk
from sentence_transformers import SentenceTransformer
import requests

from flask import Blueprint, request, jsonify, current_app

# Load environment variables
load_dotenv()
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
if not HF_API_TOKEN:
    raise ValueError("Please set HF_API_TOKEN in your .env file")

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "google/gemma-7b-it"
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')

nltk.download('punkt', quiet=True)

# Initialize embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL)

rag_bp = Blueprint('rag', __name__, url_prefix='/rag')

def split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into chunks using NLTK sentence tokenizer."""
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= chunk_size:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())
    # Add overlap
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks)):
            start = max(0, i - 1)
            overlapped_chunks.append(" ".join(chunks[start:i+1]))
        return overlapped_chunks
    return chunks

def process_pdf(pdf_file_path):
    """Extract text from PDF and split into chunks."""
    pdf_reader = PdfReader(pdf_file_path)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    texts = split_text(text)
    return texts

def extract_images_and_tables(pdf_file_path):
    """Extract images and tables from PDF."""
    doc = fitz.open(pdf_file_path)
    images = []
    tables = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append((f"Page {page_num + 1}, Image {img_index + 1}", image))
        # Table extraction placeholder (PyMuPDF doesn't extract tables natively)
        # You may use camelot or tabula for table extraction if needed
    return images, tables

def create_embeddings_and_vectorstore(texts):
    """Create embeddings and vector store from text chunks."""
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings, texts

def similarity_search(query, index, embeddings, texts, k=3):
    """Find top-k similar chunks for the query."""
    query_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k)
    results = []
    for idx, score in zip(I[0], D[0]):
        results.append((texts[idx], score))
    return results

def expand_query(query: str) -> str:
    """Expand the original query with related terms using a simple prompt."""
    # For simplicity, just return the query itself; you can use a small LLM or keyword expansion here if needed
    return query

def generate_answer(context, question):
    """Call Hugging Face Inference API for Gemma model."""
    prompt = f"""Context: {context}\n\nQuestion: {question}\n\nAnswer the question concisely based on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."""
    api_url = f"https://api-inference.huggingface.co/models/{GENERATION_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256, "temperature": 0.4}
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        elif isinstance(result, list) and "generated_text" in result[-1]:
            return result[-1]["generated_text"]
        else:
            return str(result)
    else:
        return f"Error: {response.text}"

def rag_pipeline(query, index, embeddings, texts, images, tables):
    """Run the RAG (Retrieval-Augmented Generation) pipeline."""
    expanded_query = expand_query(query)
    relevant_docs = similarity_search(expanded_query, index, embeddings, texts, k=3)
    context = ""
    log = "Query Expansion:\n"
    log += f"Original query: {query}\n"
    log += f"Expanded query: {expanded_query}\n\n"
    log += "Relevant chunks:\n"
    for i, (doc, score) in enumerate(relevant_docs, 1):
        context += doc + "\n\n"
        log += f"Chunk {i} (Score: {score:.4f}):\n"
        log += f"Sample: {doc[:200]}...\n\n"
    context += f"Number of images in the PDF: {len(images)}\n"
    context += f"Number of tables in the PDF: {len(tables)}\n"
    answer = generate_answer(context, query)
    return answer, log

def process_pdf_and_query(pdf_file_path, query):
    """Process PDF and handle the query."""
    texts = process_pdf(pdf_file_path)
    images, tables = extract_images_and_tables(pdf_file_path)
    index, embeddings, texts = create_embeddings_and_vectorstore(texts)
    result, chunks_log = rag_pipeline(query, index, embeddings, texts, images, tables)
    return result, len(texts), len(images), len(tables), chunks_log

# --- Flask Endpoints ---

@rag_bp.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400
    filename = file.filename
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    # Optionally, you can process and cache embeddings here for later queries
    return jsonify({'success': True, 'filename': filename})

@rag_bp.route('/query', methods=['POST'])
def query_rag():
    data = request.get_json()
    query = data.get('query')
    # For demo: use the latest uploaded PDF in uploads folder
    pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files:
        return jsonify({'error': 'No PDF uploaded yet.'}), 400
    latest_pdf = sorted(pdf_files, key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)), reverse=True)[0]
    pdf_path = os.path.join(UPLOAD_FOLDER, latest_pdf)
    result, num_chunks, num_images, num_tables, chunks_log = process_pdf_and_query(pdf_path, query)
    return jsonify({
        'result': result,
        'num_chunks': num_chunks,
        'num_images': num_images,
        'num_tables': num_tables,
        'log': chunks_log
    })