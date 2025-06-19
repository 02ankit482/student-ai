import streamlit as st
from chatbot.history import HistoryManager
from chatbot.user import UserManager
from documents.pdf_handler import PDFHandler

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from pathlib import Path

# --- User and History Management ---
history_manager = HistoryManager()
user_manager = UserManager()
pdf_handler = PDFHandler()

# Initialize all required session states
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        input_key="question"
    )

# Add custom CSS for layout
st.markdown("""
    <style>
    .main {
        padding-top: 60px;
    }
    </style>
    """, unsafe_allow_html=True)

# Create columns for layout
col1, col2 = st.columns([0.8, 0.2])

# Display image in the right column
with col2:
    image_path = os.path.join(os.path.dirname(__file__), "download.jpeg")
    st.image(image_path, width=100)

# Main content in the left column
with col1:
    st.title("Chatbot Application")

# Configure LLM
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="phi:2.7b",
    temperature=0.3,
    max_tokens=500,
    timeout=60,
    model_kwargs={
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1
    }
)

def get_conversation_chain(retriever):
    # Create the prompt template
    template = """You are a helpful AI assistant that provides accurate answers based on the documents provided.
    Answer the questions using ONLY the provided context. Be concise and clear.
    
    If the answer cannot be found in the context, respond with:
    "I cannot answer this based on the provided documents."
    
    Context: {context}
    Chat History: {chat_history}
    Question: {question}
    Answer:"""

    # Create the prompt
    prompt = ChatPromptTemplate.from_template(template)

    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": prompt
        },
        verbose=True
    )
    
    return chain

# --- Sidebar UI ---
with st.sidebar:
    st.title("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    
    if uploaded_file:
        if uploaded_file.name not in st.session_state.processed_files:
            with st.spinner("Processing document..."):
                try:
                    # Validate file size (max 10MB)
                    if uploaded_file.size > 10 * 1024 * 1024:
                        st.error("File size exceeds 10MB limit")
                    else:
                        vector_store, is_new = pdf_handler.process_pdf(uploaded_file)
                        st.session_state.vector_store = vector_store
                        st.session_state.conversation = get_conversation_chain(
                            vector_store.as_retriever(
                                search_kwargs={"k": 4}
                            )
                        )
                        if is_new:
                            st.session_state.processed_files.add(uploaded_file.name)
                            st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    
    # Add clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.success("Conversation cleared!")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

if prompt := st.chat_input("Ask about your documents"):
    if not st.session_state.conversation:
        st.warning("Please upload a document first.")
    else:
        try:
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": prompt})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
                
                if response.get('source_documents'):
                    with st.expander("View Sources"):
                        for i, doc in enumerate(response['source_documents'], 1):
                            st.markdown(f"**Source {i}:**\n```\n{doc.page_content[:200]}...\n```")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error("Please try again or rephrase your question.")