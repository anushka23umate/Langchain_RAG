from typing import TypedDict 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# Step 1: Define the output structure
class ReviewAnalysis(TypedDict):
    summary: str
    sentiment: str 

# Step 2: Create the prompt
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a smart review analyst."),
    ("human", "Analyze the following review: {review_text}. "
              "Give a short summary and mention if the sentiment is positive or negative.")
])

# Step 3: Inject input
prompt = chat_template.invoke({
    "review_text": "I absolutely loved the new phone! The battery lasts forever and the camera is amazing."
})


structured_model = model.with_structured_output(ReviewAnalysis)

# Step 5: Get result
result = structured_model.invoke(prompt)

# Step 6: Print result
print("Summary:", result["summary"])
print("Sentiment:", result["sentiment"])
