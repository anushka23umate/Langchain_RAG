from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

structured_output_parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me 3 facts about {topic} in {formatted_instructions}",
    input_variables=['topic'],
    partial_variables={'formatted_instructions': structured_output_parser.get_format_instructions()}
)   

prompt= template.format(topic='Python programming')

# without using chaining concept
# result = model.invoke(prompt)
# final_result = structured_output_parser.parse(result.content)
# print(final_result)

# with using chaining concept
chain = template | model | structured_output_parser
result = chain.invoke({'topic': 'Python programming'})
print(result)