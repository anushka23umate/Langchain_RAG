##in this our llm dont have any context of past history, so it will not remember previous conversations.
# This code snippet is for a simple chatbot using Hugging Face's Llama-3.1-8B-Instruct model.
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

while True:
    user_input = input("You: ")
    if user_input == 'exit':
         break
    result = model.invoke(user_input)
    print("AI: ",result.content)

