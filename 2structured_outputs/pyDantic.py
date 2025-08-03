from typing import TypedDict 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from typing_extensions import Literal


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


class Review(BaseModel):
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )
    summary: str = Field(
        description="A brief summary of the review"
    )
    sentiment: Literal["pos", "neg"] = Field(
        description="Return sentiment of the review: either positive or negative"
    )
    pros: Optional[list[str]] = Field(
        default=None,
        description="Write down all the pros inside a list"
    )
    cons: Optional[list[str]] = Field(
        default=None,
        description="Write down all the cons inside a list"
    )
    name: Optional[str] = Field(
        default=None,
        description="Write the name of the reviewer"
    )

# Step 3: Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that analyzes product reviews and returns structured data."),
    ("human", "Analyze the following review: {review_text}")
])

# Step 4: Combine model with structured output
structured_model = model.with_structured_output(Review)

# Step 5: Inject sample input
user_input = {
    "review_text": (
        "I recently bought the Pixel 8 Pro. The camera is absolutely incredible, and the performance is smooth. "
        "However, the battery life could be better, and I find the phone a bit slippery without a case. "
        "Overall, I'm happy with it. - Reviewed by Anushka"
    )
}

# Step 6: Run model pipeline
final_prompt = prompt.invoke(user_input)
response = structured_model.invoke(final_prompt)

# Step 7: Display result
print("\n✅ Structured Output:")
print(response.model_dump())  # or response.dict() if using pydantic v1


# If you want to use with_structured_output method, then either you should use OpenAI, Gemini or Anthropic models
#and if u wanna use HuggingFace models, then you should use the model.with_structured_output(Review) method along with output parses