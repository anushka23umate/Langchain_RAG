import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import tempfile
import warnings
warnings.filterwarnings("ignore")


# Set page config
st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")

st.title("📄 RAG PDF Chatbot")
st.write("Upload a PDF, ask questions, and get answers powered by Mixtral LLM")

# Step 1: Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

if pdf_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    # Step 2: Load the PDF
    with st.spinner("Loading and splitting document..."):
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Step 3: Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        # Step 4: Create embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embedding_model)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        # Step 5: Setup LLM
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation"
        )
        model = ChatHuggingFace(llm=llm)

        # Step 6: Build QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            return_source_documents=True
        )

    st.success("Document loaded! You can now ask questions.")

    # Step 7: Ask Question
    user_question = st.text_input("Ask a question based on the PDF:")
    if user_question:
        with st.spinner("Generating answer..."):
            result = qa_chain.invoke(user_question)
            st.markdown("### 📌 Answer:")
            st.write(result['result'])

            with st.expander("📚 Source Chunks"):
                for doc in result["source_documents"]:
                    st.markdown(f"---\n{doc.page_content.strip()}")
