#Falcon LLM with LangChain
#LangChain has a pipeline called HuggingFacePipeline for models hosted in HuggingFace.

#install langchain package
#!pip install langchain

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from langchain.llms import HuggingFacePipeline

model_name = "tiiuae/falcon-7b-instruct"

#using the transformers library
#download tokeniser from the autotokenizer class
#download the model from the class AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#use case here- text gen
#parameters
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
#create a pipeline for the Falcon model
from langchain import HuggingFacePipeline

#call the huggingface pipeline() object & pass the pipeline and model parameters
#temperature of 0, makes the model not to hallucinate much (make its own answers)
llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

#Langchain contains PromptTemplate - allows to alter answers from the llm
#LLMChain - chains the prompttemplate and LLM together

from langchain import PromptTemplate, LLMChain

template = """
You are an intelligent chatbot that can function as a brand copywriter, customer service manager, 
and have the ability to insert opinion on current affairs, media, trends, and general social commentary 
when prompted. You will understand specific humor based off pop culture and media, sarcasm, 
and social references.
Question: {query}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["query"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

query = "How do i pay for a service at the market? Write me an approach for this"

print(llm_chain.run(query))






