from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.base import ConversationalRetrievalChain  # Updated import
from langchain.memory import ConversationBufferWindowMemory
from pypdf import PdfReader
import os

def process_pdf(file_path):
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="phi:2.7b")
    
    # Read PDF
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    
    # Create vector store
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )
    return vector_store

def setup_qa_chain(vector_store):
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model="phi:2.7b",
        temperature=0.1,
    )
    
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=3,
        return_messages=True
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(  # Changed from ConversationalRetrievalQA
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    return qa_chain

def main():
    # Check if vector store exists
    if not os.path.exists("vectorstore"):
        pdf_path = input("Enter the path to your PDF file: ")
        vector_store = process_pdf(pdf_path)
        print("PDF processed and vectorized successfully!")
    else:
        # Load existing vector store
        vector_store = Chroma(
            persist_directory="vectorstore",
            embedding_function=OllamaEmbeddings(model="phi:2.7b")
        )
        print("Loaded existing vector store")
    
    qa_chain = setup_qa_chain(vector_store)
    
    print("\nChat with your documents (type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break
            
        try:
            response = qa_chain({"question": query})
            print("\nAnswer:", response['answer'])
            
            # Print sources
            if 'source_documents' in response:
                print("\nSources:")
                for i, doc in enumerate(response['source_documents'], 1):
                    print(f"\n{i}. {doc.page_content[:200]}...")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()