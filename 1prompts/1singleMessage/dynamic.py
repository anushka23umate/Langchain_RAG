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

st.header('🧪 Research Tool')

paper_input = st.selectbox(
    "📚 Select Research Paper", 
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "🎨 Select Explanation Style", 
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "📏 Select Explanation Length", 
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

template=load_prompt("D:\\LangChain\\research_paper_summary_template.json")

if st.button('Summarize'):
    prompt = template.format(
        paper=paper_input,
        style=style_input,
        length=length_input
    )
    result = model.invoke(prompt)
    st.subheader("📝 Summary")
    st.write(result.content)
