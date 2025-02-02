from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

import os

# ----------------------------------------------------------------------------------------
# How to build a simple LLM App with LangChain and use LangServe to quickly run an http server 
# with a simple UI to call rest APIs (using FastAPI)
#   * Very simple LLM App.
#   * Use FastAPI to create a rest API
#   * run the app - LangServe will create an http server with a UI and rest API
#   * The rest API is automatically configured to call the LangChain app
# 
# LangServe playground - run the app on your local
#   1. execute this python script in your terminal
#   2. go to: http://localhost:8000/chain/playground/ to see a basic UI for local testing of the LangChain
#       - swaggerUI: http://localhost:8000/docs/
#       - openAPI json schema: http://localhost:8000/openapi.json
#
# Or deploy to the cloud - you just need to be able to execute the python script on the server.
#  - very similar concept to:
#      - using node to launch an http server for a javascript app
#      - or using springboot to run/launch an http server for a java microservice
# ----------------------------------------------------------------------------------------

_ = load_dotenv(find_dotenv())

# Create a LangChain app in the usual way
openai_api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()
system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

chain = prompt_template | llm | parser


# create a FastAPI app
app = FastAPI(
  title="simpleTranslator",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)