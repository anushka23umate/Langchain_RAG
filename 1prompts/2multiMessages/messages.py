from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(
        content="You are a helpful assistant! Your name is Bob."
    ),
    HumanMessage(
        content="What is your name?"
    )
]

result=model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)