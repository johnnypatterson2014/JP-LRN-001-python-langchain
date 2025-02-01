# Example python project using LangChain and OpenAI 

## Setup
- pyenv local 3.11.4
- pyenv activate local
- poetry install
- poetry shell

## Run
 & {path-to-folder}/.venv/Scripts/python.exe {path-to-folder}/{filename}.py

## Request & response from OpenAI
View the request/response to OpenAI here: https://smith.langchain.com/

## Python script: 001-data-loaders.py
Demonstrate LangChain built-in data loaders:
- local text file
- local csv file
- local html file
- Wikipedia page content

Then ask the LLM a question about the loaded data.

## Python script: 002-simple-rag.py
Demonstrate a simple RAG app using LangChain (using LCEL and without). 

RAG process:
- Split document in small chunks of text
- Transform text chunks in numeric chunks (embeddings)
- Load embeddings to a vector database
- Given a question, retrieve the most similar embeddings in the vector db to send to the LLM 
- create a prompt with the question and embeddings
- send prompt to LLM
- receive response from LLM
- format the output of the LLM


