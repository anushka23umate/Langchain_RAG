import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model= ChatHuggingFace(llm=llm)

result=model.invoke("What is the capital of India?")
print(result.content)

# This code uses the HuggingFaceEndpoint to access a model hosted on Hugging Face.
# Make sure you have the necessary API key set in your environment variables.