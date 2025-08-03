from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Step 0: Load environment variables
load_dotenv()

# Step 1: Set USER_AGENT if not already set
os.environ["USER_AGENT"] = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)

# Step 2: Load web content
loader = WebBaseLoader("https://python.langchain.com/docs/versions/migrating_chains/")
documents = loader.load()

# Debug print to verify document load
if not documents or not documents[0].page_content.strip():
    raise ValueError("Web page could not be loaded or content is empty!")

# Step 3: Setup LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# Step 4: Prompt
prompt = PromptTemplate(
    template="Answer the following question {question} based on the provided web content:\n\n{topic}",
    input_variables=["question","topic"]
)

# Step 5: Output parser
parser = StrOutputParser()

# Step 6: Build and run the chain
chain = prompt | model | parser
result = chain.invoke({"question":"what is the webpage all abt?","topic": documents[0].page_content})

print(result)
