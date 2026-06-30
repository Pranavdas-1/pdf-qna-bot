# 📄 PDF QnA Bot

A RAG-based PDF Question Answering chatbot built with LangChain, Google Gemini, and Streamlit.

🔗 **Live Demo:** [pdf-qna-bot.streamlit.app](https://pdf-qna-bot.streamlit.app)

---

## What it does

Upload any PDF and ask questions about it in natural language. The app retrieves the most relevant sections from the document and uses Google Gemini to generate accurate, context-grounded answers.

---

## How it works

```
PDF Upload
    ↓
Document Chunking (RecursiveCharacterTextSplitter)
    ↓
Embedding (Google Gemini Embeddings)
    ↓
Vector Store (InMemoryVectorStore)
    ↓
User Query → Similarity Search → Top 3 Relevant Chunks
    ↓
Gemini LLM generates answer from context
    ↓
Chat interface displays response
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| LangChain | RAG pipeline, document loading, text splitting |
| Google Gemini 2.5 Flash | LLM for answer generation |
| Gemini Embeddings | Converting text chunks to vectors |
| InMemoryVectorStore | Storing and searching embeddings |
| PyPDFLoader | Loading and parsing PDF files |
| Streamlit | Web UI and chat interface |
| Python Dotenv | Managing API keys |

---

## Features

- Upload any PDF via browser
- Automatic document chunking with overlap for context preservation
- Semantic search using vector embeddings
- Multi-turn chat interface with message history
- Answers strictly grounded in document context — no hallucination
- Clean Streamlit chat UI

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/pdf-qna-bot.git
cd pdf-qna-bot
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API key

Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your free API key at [Google AI Studio](https://aistudio.google.com)

### 5. Run the app
```bash
streamlit run app.py
```

---

## Project Structure

```
pdf-qna-bot/
├── app.py               # Main application
├── .env                 # API keys (not committed)
├── requirements.txt     # Dependencies
└── README.md
```

---

## Requirements

```
langchain
langchain-community
langchain-google-genai
langchain-text-splitters
streamlit
pypdf
python-dotenv
```

---

## RAG Pipeline

This project implements a basic **Retrieval-Augmented Generation (RAG)** pipeline:

1. **Load** — PDF is loaded using PyPDFLoader
2. **Split** — Document split into 1000-character chunks with 200-character overlap
3. **Embed** — Each chunk converted to a vector using Gemini Embeddings
4. **Store** — Vectors stored in InMemoryVectorStore
5. **Retrieve** — User query embedded and top 3 similar chunks fetched
6. **Generate** — Gemini LLM answers based only on retrieved context

---

## Author

**Pranav Das**
B.Tech CSE — College of Engineering Chengannur (KTU, 2026)

[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername) · [Behance](https://behance.net/yourprofile)
