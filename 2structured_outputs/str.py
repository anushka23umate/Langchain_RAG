from langchain.output_parsers import RegexParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()


parser = RegexParser(
    regex=r"\*\*(.*)\*\*\n\n\*\*Genre:\*\* (.*)\n\n\*\*Rating:\*\* (.*)",
    output_keys=["title", "genre", "rating"]
)


prompt = PromptTemplate.from_template(
    "Describe the movie '{movie}' with title, genre and rating on new lines."
)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)
chain = prompt | model | parser

result = chain.invoke({"movie": "bahabuli 2"})
print(result)  # Structured dict extracted from raw text
