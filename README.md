# PDF Question Answering using RAG (Groq + LangChain)

This project implements a Retrieval-Augmented Generation (RAG) pipeline that enables users to ask natural language questions about a PDF document. The system retrieves relevant content from the document and generates concise, context-aware answers using a Groq-hosted large language model.

---

## Project Overview

The application processes a PDF file, converts its content into vector embeddings, stores them in a vector database, and uses retrieval-based prompting to answer user questions accurately. This approach ensures that responses are grounded in the document rather than relying on the model’s general knowledge.

---

## How It Works

1. **PDF Loading**  
   The PDF file is loaded using `PyPDFLoader`.

2. **Text Chunking**  
   The document is split into overlapping chunks using `RecursiveCharacterTextSplitter` to preserve semantic context.

3. **Embedding Generation**  
   HuggingFace sentence-transformer embeddings are used to generate vector representations of text chunks.  
   *(No OpenAI API key is required.)*

4. **Vector Storage**  
   The embeddings are stored in a Chroma vector database for efficient similarity-based retrieval.

5. **Context Retrieval**  
   Relevant document chunks are retrieved based on the user’s question.

6. **Answer Generation**  
   A Groq-powered LLaMA 3.1 model generates a concise answer using only the retrieved context.

---

## Technologies Used

- Python 3.11
- LangChain
- Groq (LLaMA 3.1 – 8B Instant)
- HuggingFace Sentence Transformers
- Chroma Vector Database
- PyPDFLoader

---
# Run the Project
-Create and Activate Virtual Environment (Poetry)
poetry install
poetry shell
python 001-qa-from-pdf.py



