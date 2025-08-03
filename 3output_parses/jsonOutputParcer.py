from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

JsonOutputParser= JsonOutputParser()

template=PromptTemplate(
    template= "Give me name, age and city of fictional charcater in{formated_inst}",
    input_variables= [],
    partial_variables= {'formated_inst': JsonOutputParser.get_format_instructions()}
)

#without using chaining concept
# prompt=template.format()
# result = model.invoke(prompt)
# final_result = JsonOutputParser.parse(result.content)
# print(final_result)

#with using chaining concept
chain= template | model | JsonOutputParser
result = chain.invoke({})
print(result)