#Run this script on colab or system with high GPU

#Falcon-40B-Instruct has already been finetuned on the conversational data
#we'll use the Transformers library to work with this model

#install transformers accelerate einops xformers

#A chatbot using Falcon 7b instruct model

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

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

#test on colab due to storage space on local pc
#test the llm-falcon7b by providing it with a query
sequences = pipeline(
    "Create a list of four important things a car driver should take note of"
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")

#by the time is the code is pushed to github, possibly there would have been advancements to enable
#you to run this locally.