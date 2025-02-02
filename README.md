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

## Python script: 003-chatbot-with-chat-history.py
Example chatbot that remembers the chat history. 
- We will use the ChatMessageHistory package.
- We will save the Chatbot memory in a python dictionary called chatbotMemory.
- We will define the get_session_history function to create a session_id for each conversation.
- We will use the built-in runnable RunnableWithMesageHistory.

Chatbot with memory
 - will remember previous chat history
 - chat history is saved to a file
 - note that the entire chat history is sent to the LLM context window.

Example Chatbot that saves chat history by session_id

Example Chatbot with history but limits the size of the history passed to the context window
 - The memory of a chatbot is included in the context window of the LLM so, if left unmanaged, can potentially overflow it.
 - We are now going to learn how to limit the size of the memory of a chatbot







