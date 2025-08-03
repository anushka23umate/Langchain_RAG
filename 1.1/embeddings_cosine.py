from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

documents=[
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin .Tendulkar, also known as the.'God of Cricket', holds .many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his .unorthodox action and yorkers."
    ]

# Initialize the HuggingFaceEmbeddings with a specific model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

query="tell me about Virat Kohli"

documents_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

# Calculate cosine similarity between the query embedding and document embeddings
similarities = cosine_similarity([query_embedding], documents_embeddings)[0]

# print(list(enumerate(similarities))) # similarities of each document with the query
# print(sorted(enumerate(similarities), key=lambda x: x[1], reverse=True))  #sorting the similarities in descending order
index,score=sorted(enumerate(similarities), key=lambda x: x[1])[-1]  #sorting the similarities in ascending  order and printing the last one, which is the most similar document

print(query)
print(documents[index])
print("Cosine similarity score:", score)