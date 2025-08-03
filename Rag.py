import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# === Load environment variables ===
load_dotenv()

# === STEP 1: Load and split PDF file ===
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

documents = load_pdf("WKBPCHDP Program.pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
split_docs = text_splitter.split_documents(documents)

# === STEP 2: Create or load FAISS vector store ===
embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore_path = "vectorstore"
faiss_index_path = os.path.join(vectorstore_path, "index.faiss")

if not os.path.exists(faiss_index_path):
    print("🔧 Creating new FAISS vector store...")
    vectordb = FAISS.from_documents(split_docs, embedding_model)
    vectordb.save_local(vectorstore_path)
else:
    print("📂 Loading existing vector store...")
    vectordb = FAISS.load_local(
        vectorstore_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

retriever = vectordb.as_retriever(search_type="similarity", k=4)

# === STEP 3: Load Hugging Face LLM ===
llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
llm = ChatHuggingFace(llm=llm_endpoint)

# === STEP 4: Build RetrievalQA chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# === STEP 5: Interactive chat ===
print("\n📘 PDF QA Chatbot Ready!\nType 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    result = qa_chain.invoke(query)  # ✅ Use invoke() instead of __call__()
    print("\n🤖 Answer:\n", result["result"])

    # 🔍 Optional: View source documents
    # print("\n🔍 Source Docs:")
    # for doc in result["source_documents"]:
    #     print("-", doc.metadata.get("source", "Unknown"))
