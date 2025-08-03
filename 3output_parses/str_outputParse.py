# #this block is not using output parser, this is just a simple example of using two prompts in sequence without using chaining concept

# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate 
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation",
# )

# model = ChatHuggingFace(llm=llm)

# # 1st prompt -> detailed report
# template1 = PromptTemplate(
#     template='Write a detailed report on {topic} ',
#     input_variables=['topic' ]
# )

# # 2nd prompt -> summary
# template2 = PromptTemplate(
#     template='5 line points on {topic} ',    
#     input_variables=['topic' ]
# )

# prompt1 = template1.invoke({"topic": "Artificial Intelligence"})
# result1 = model.invoke(prompt1)
# prompt2= template2.invoke(result1.content)
# result2 = model.invoke(prompt2)

# print(result2.content)





#this block is using output parser, this is just a simple example of using two prompts in sequence with chaining concept
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic} ',
    input_variables=['topic' ]
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='5 line points on {topic} ',    
    input_variables=['topic' ]
)

StrOutputParser = StrOutputParser()

chain= template1 | model | StrOutputParser | template2 | model | StrOutputParser

result = chain.invoke({"topic": "Artificial Intelligence"})
print(result)
