# from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# model = ChatOpenAI()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model= ChatHuggingFace(llm=llm)

st.header('Reasearch To1')

user_input = st.text_input('Enter your prompt')

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)
