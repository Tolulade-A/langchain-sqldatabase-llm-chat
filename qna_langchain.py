#Creating a question answering app using LangChain
#using openai api keys will require subscriptions
#first, let's Install langchain,openai and python-environ libraries using pip
#create an env file with your openai api


#then we create simple LLM call using LangChain
from langchain.llms import OpenAI

#accessing the openai key
import environ
env = environ.Env()
environ.Env.read_env()
API_KEY = env('OPENAI_API_KEY')

#let's make a simple call using Langchain

llm = OpenAI(model_name="text-davinci-003", openai_api_key=API_KEY)
question = "What is the best platform in Nigeria to learn data science ?"
print(question, llm(question))

#notes
#We have imported the OpenAI wrapper from langchain. The OpenAI wrapper requires an openai key.
# The OpenAI key is accessed from the environment variables using the environ library.
# Initialize it to a llm variable with text-davinci-003 model.
# Finally, define a question string and generate a response (llm(question)).

#run the llm call (script)
#python qna_langchain.py
