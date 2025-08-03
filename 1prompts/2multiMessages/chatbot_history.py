from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(
        content="You are a helpful assistant! Your name is Bob."
    )
]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
         break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)
print("Chat History: ", chat_history)

#we inclue SystemMessage to prime the AI behavior
#we use HumanMessage to capture user input  
#we use AIMessage to capture AI response
#we use a while loop to continue the conversation until the user types 'exit'
#without that also it was working but it was not able to capture if its a user input or AI response
