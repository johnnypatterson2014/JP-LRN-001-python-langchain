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


#### LangServe vs. FastAPI
# Deploying a LangChain app with LangServe or FastAPI involves similar basic principles—both methods aim to serve your application over the web—but they differ in their level of specialization and the features they offer. Here's a simple explanation of the main differences between these two deployment options:
# 
## FastAPI
# 1. **General-Purpose Framework**: FastAPI is a modern, fast (high-performance) web framework for building APIs with Python. It's designed to be simple to use but powerful in capabilities, supporting the development of robust APIs and web applications.
# 2. **Flexibility**: FastAPI provides extensive flexibility in how you structure your application. It allows for detailed customization of request handling, response formatting, and middleware integration, making it suitable for a wide variety of web services beyond just language applications.
# 3. **Manual Setup**: When deploying a LangChain app with FastAPI, you need to manually set up the routing, request handling, and integration with LangChain. This involves writing more boilerplate code and handling more configuration details.
# 4. **Community and Ecosystem**: FastAPI has a large developer community and a rich ecosystem of plugins and tools, which can be advantageous for solving common web development problems and integrating with other technologies.
# 
## LangServe
# 1. **Specialized for LangChain**: LangServe is tailored specifically for deploying LangChain applications. This specialization means it comes with built-in configurations and setups optimized for language model applications, reducing the need to manually configure many aspects of deployment.
# 2. **Simplicity and Convenience**: LangServe aims to simplify the process of turning your LangChain model into a web service. It abstracts away many of the lower-level details of web service configuration, allowing you to focus more on developing the language model itself.
# 3. **Integrated Tools**: Since LangServe is designed to work seamlessly with LangChain, it often includes tools and features that specifically support language model operations, such as handling different types of language inputs and outputs more effectively.
# 4. **Limited Flexibility**: While offering simplicity, LangServe may not provide as much flexibility as FastAPI in terms of general web development capabilities. It's optimized for a specific type of application, which might limit its utility outside of deploying language models.