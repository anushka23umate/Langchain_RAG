from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Step 1: Load the PDF
loader = PyPDFLoader("D:\\LangChain\\rag\\1DocumentLoader\\content\\WKBPCHDP Program.pdf")
pages = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(pages)

# Step 3: Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store in FAISS (or Chroma)
vector_db = FAISS.from_documents(chunks, embedding_model)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Step 5: Setup LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Step 6: Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    return_source_documents=True
)

# Step 7: User question
query = "What is the document about?"
response = qa_chain.invoke(query)

print(response['result']) 
