from typing import TypedDict 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')


parser = PydanticOutputParser(pydantic_object=Person)    

template=PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()})


#without chaining concept
# prompt = template. invoke({'place':'indian'})
# result = model. invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)


#with chaining concept
chain = template | model | parser
result = chain.invoke({'place':'indian'})
print(result)