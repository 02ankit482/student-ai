# Streamlit Chatbot Application

This project is a Streamlit application that implements a chatbot with advanced features including chat history management, retrieval-augmented generation (RAG) context, user ID management, and document upload functionality primarily for PDFs. The application uses SQLite for data storage.

## Project Structure

```
streamlit-chatbot-app
├── src
│   ├── app.py                  # Entry point of the Streamlit application
│   ├── chatbot
│   │   ├── __init__.py         # Initializes the chatbot module
│   │   ├── history.py          # Manages chat history for users
│   │   ├── rag.py              # Handles retrieval-augmented generation context
│   │   └── user.py             # Manages user IDs and contexts
│   ├── db
│   │   ├── __init__.py         # Initializes the database module
│   │   └── sqlite_manager.py    # Handles SQLite database operations
│   ├── documents
│   │   ├── __init__.py         # Initializes the documents module
│   │   └── pdf_handler.py       # Manages PDF uploads and processing
│   └── utils
│       └── __init__.py         # Initializes the utils module
├── requirements.txt             # Lists project dependencies
└── README.md                    # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd streamlit-chatbot-app
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```
   streamlit run src/app.py
   ```

## Usage Guidelines

- Upon running the application, users can interact with the chatbot through the Streamlit interface.
- Users can upload PDF documents, which will be processed to extract text for RAG context.
- The chatbot maintains a history of interactions for each user, allowing for context-aware conversations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.