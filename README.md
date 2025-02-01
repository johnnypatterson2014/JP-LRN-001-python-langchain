# JP-LRN-001-python-langchain

## Setup
pyenv local 3.11.4
pyenv activate local
poetry install
poetry shell

## Run
 & {path-to-folder}/.venv/Scripts/python.exe {path-to-folder}/{filename}.py

## LLM QA trace
View the request/response to OpenAI here: https://smith.langchain.com/

## 001-data-loaders.py
Demonstrate LangChain built-in data loaders:
- local text file
- local csv file
- local html file
- Wikipedia page content

Then ask the LLM a question about the loaded data.


