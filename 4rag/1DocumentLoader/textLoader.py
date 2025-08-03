from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
    # repo_id="meta-llama/Llama-3.1-8B-Instruct",
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt= PromptTemplate(
    template='Summarize the following text in one line:\n\n {topic} ',
    input_variables=['topic']
)

parser= StrOutputParser()


# Load the document
loader = TextLoader(r"D:\LangChain\sample_1000_words.txt", encoding="utf-8")
document = loader.load()

chain= prompt | model | parser
result = chain.invoke({"topic": document[0].page_content})
print(result)