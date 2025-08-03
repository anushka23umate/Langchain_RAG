# 💬 LangChain Learning & RAG PDF Chatbot Project

This repository documents my journey with [LangChain](https://python.langchain.com/) — from prompt engineering to building a Retrieval-Augmented Generation (RAG) chatbot using Streamlit, HuggingFace, and vector databases like FAISS and Chroma.

---

## 🔍 Features

### 📄 RAG PDF Chatbot (Streamlit)
- Upload a PDF, embed its content, and ask questions about it.
- Choose between persistent (ChromaDB) and in-memory (FAISS) vector stores.
- Embedding via HuggingFace (`all-MiniLM-L6-v2`), and responses generated using Mixtral via HuggingFace Inference API.
- PDF metadata is cached in SQLite for faster re-use and storage tracking.

### 🧠 Prompt Engineering
- Dynamic and static templates for single- and multi-turn prompts.
- Role-based messaging for conversation structuring.

### ✨ Embeddings & Similarity
- Generate embeddings using HuggingFace.
- Run cosine similarity comparisons across documents or queries.

### 📦 Structured Output Parsing
- Leverages `TypedDict` to parse structured model outputs for tasks like review analysis.

---

## 🗂️ Directory Overview

```
.
.
├── 1.1/
│   ├── chatmodel_localDownload.py         # Load local models
│   ├── embedding_huggingface.py           # Load HF embeddings
│   └── embeddings_cosine.py               # Cosine similarity-based search
│
├── 1prompts/
│   ├── singleMessage/                     # Prompt templates for single-turn chat
│   └── multiMessages/                     # Templates for multi-turn history
│
├── 2structured_outputs/
│   ├── BASic_pyDantic.py                  # Pydantic structure example
│   ├── pyDantic.py                        # Main pydantic model
│   ├── str.py, typedDict.py               # Other schema variants
│   └── output_parses/                     # Parsers for structured model outputs
│
├── 4rag/
│   ├── 1DocumentLoader/                   # Load PDF docs (PyMuPDF, etc.)
│   ├── 2TextSplitter/                     # Chunking logic
│   ├── 3VectorDB/                         # ChromaDB or FAISS
│   └── Retrievers/                        # RAG retriever logic
│
├── DBstreamlit.py                         # 🟢 Main Streamlit app
├── DB.py                                  # Local SQLite metadata logger
├── pdf_metadata.db                        # Stores PDF hash + meta
├── Rap.py, ok.py                          # Scratch / testing scripts
├── requirements.txt                       # 🔧 Python dependencies
├── .env                                   # Environment variables (HF token, etc.)
└── research_paper_summary_template.json   # Optional: for structured outputs

```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/anushka23umate/Langchain_RAG.git
cd langchain-learning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file and include your HuggingFace API key:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### 4. Launch the chatbot

```bash
streamlit run DBstreamlit.py
```

---

## 💡 Usage

- **RAG Chatbot:** Upload a PDF → Choose vector store → Ask questions.
- **Prompt Experiments:** Check `prompts/` for reusable templates and dynamic prompts.
- **Embedding & Similarity:** Use `embedding_huggingface.py` or `embeddings_cosine.py` to experiment with similarity scoring.
- **Structured Output:** Explore `structured_outputs/` for structured answer extraction.

---

## ⚠️ Notes

- Don’t expose API keys or private documents.
- You can switch between FAISS and ChromaDB using the UI options in the Streamlit app.
- Embeddings are cached locally for faster access when using ChromaDB.

---

## 📜 License

MIT

---
**Author:** Anushka Umate  
**LinkedIn:** [anushka-umate](https://www.linkedin.com/in/anushka-umate)

