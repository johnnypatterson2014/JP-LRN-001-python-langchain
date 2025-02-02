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

## Python script: 004-key-data-extraction.py
We will create an app to extract structured information from unstructured text. 
Imagine, for example, that you want to extract the name, the lastname and the country of 
the users that submit comments in the website of your company.

## Python script: 005-sentiment-analysis.py
* Sentiment Analysis is a very popular functionality. For example, be able to determine if a product review is positive or negative.
* Our app will be able to do more than that. It will be a text classification app, also called a "tagging" app.
* In short, we will create an app to classify text into labels. And these labels can be:
    * Sentiment: Sentiment Analysis app.
    * Language: Language Analysis app.
    * Style (formal, informal, etc): Style Analysis app.
    * Topics or categories: Topic or category Analysis app.
    * Political tendency: Political Analysis app.
    * Etc.

## Python script: 006-qa-from-sql.py
SQL integration
  * We will create a Q&A app over tabular data in databases.
  * These app will allow us to **ask a question about the data in a database in natural language and get back an answer also in natural language**.
  * Building Q&A systems of SQL databases requires executing model-generated SQL queries. There are inherent risks in doing this. In production, make sure that your database connection permissions are always scoped as narrowly as possible for your chain's needs.



