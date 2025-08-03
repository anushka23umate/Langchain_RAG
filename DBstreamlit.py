import hashlib
import tempfile
from datetime import datetime
import streamlit as st
import sqlite3

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# === Streamlit UI ===
st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")
st.title("📄 RAG PDF Chatbot")
st.write("Upload a PDF, choose vector DB type, and ask questions!")

# === SQLite Metadata DB Setup ===
DB_PATH = "pdf_metadata.db"

def init_metadata_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_files (
            pdf_id TEXT PRIMARY KEY,
            file_name TEXT,
            uploaded_at TEXT,
            chroma_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_pdf_metadata(pdf_id, file_name, uploaded_at, chroma_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO pdf_files (pdf_id, file_name, uploaded_at, chroma_path)
        VALUES (?, ?, ?, ?)
    ''', (pdf_id, file_name, uploaded_at, chroma_path))
    conn.commit()
    conn.close()
    print("✅ Inserted metadata into DB:", pdf_id, file_name)
    st.info(f"✅ Inserted metadata into DB: {file_name}")


def check_pdf_exists(pdf_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM pdf_files WHERE pdf_id = ?', (pdf_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

# === Utility Functions ===
def get_pdf_id(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
def get_chroma_path(pdf_id):
    return os.path.join("chromadb", pdf_id)

def chroma_exists(pdf_id):
    return os.path.exists(os.path.join(get_chroma_path(pdf_id), "index"))

def get_or_create_chroma_vectorstore(chunks, embedding_model, pdf_id, file_name):
    persist_dir = get_chroma_path(pdf_id)
    os.makedirs(persist_dir, exist_ok=True)

    if chroma_exists(pdf_id):
        st.info("🔄 Using cached embeddings from ChromaDB.")
        vector_db = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    else:
        st.info("🧠 Creating new embeddings and storing in ChromaDB...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_dir,
            collection_metadata={
                "pdf_id": pdf_id,
                "file_name": file_name,
                "created_at": datetime.now().isoformat()
            }
        )
        vector_db.persist()
        st.success("✅ Embeddings stored to disk.")
        log_pdf_metadata(pdf_id, file_name, datetime.now().isoformat(), persist_dir)

    return vector_db

# === Init Metadata DB ===
init_metadata_db()

# === Upload PDF ===
pdf_file = st.file_uploader("📤 Upload your PDF", type=["pdf"])

# === Choose Vector Store ===
vector_store_choice = st.radio("📦 Choose Vector Store", ["ChromaDB (Persistent)", "FAISS (In-Memory)"])
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Main Logic ===
if pdf_file:
    file_bytes = pdf_file.read()
    pdf_id = get_pdf_id(file_bytes)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    with st.spinner("⏳ Processing document..."):
        loader = PyMuPDFLoader(tmp_path)
        pages = loader.lazy_load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        # === Vector Store ===
        if vector_store_choice == "FAISS (In-Memory)":
            st.info("⚡ Using FAISS (In-Memory)")
            vector_db = FAISS.from_documents(chunks, embedding_model)
        else:
            vector_db = get_or_create_chroma_vectorstore(chunks, embedding_model, pdf_id, pdf_file.name)

        # === Retriever ===
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "lambda_mult": 0.5},
        )

        # === LLM ===
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            task="text-generation"
        )
        model = ChatHuggingFace(llm=llm)

        # === QA Chain ===
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            return_source_documents=True
        )

    st.success("✅ Document loaded! Ask questions below.")

    # === Ask Question ===
    question = st.text_input("❓ Ask a question:")
    if question:
        with st.spinner("🤖 Generating answer..."):
            result = qa_chain.invoke(question)

        st.markdown("### ✅ Answer:")
        st.write(result['result'])

        with st.expander("📚 Source Chunks"):
            for i, doc in enumerate(result["source_documents"], 1):
                st.markdown(f"**Chunk {i}**:\n{doc.page_content.strip()}\n---")
