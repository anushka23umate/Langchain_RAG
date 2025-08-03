from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
    )
)

model = ChatHuggingFace(llm=llm)
result=model.invoke("What is the capital of France?")
print(result.content)


##this will download the model to the cache directory
# If you want to use a different model, change the model_id in the from_model_id