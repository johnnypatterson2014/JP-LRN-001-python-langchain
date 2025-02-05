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
  * These app will allow us to ask a question about the data in a database in natural language and get back an answer also in natural language.
  * Building Q&A systems of SQL databases requires executing model-generated SQL queries. There are inherent risks in doing this. In production, make sure that your database connection permissions are always scoped as narrowly as possible for your chain's needs.

## Python script: 007-qa-from-pdf.py
* We will create a Q&A app that can answer questions about PDF files.
* We will use a Document Loader to load text in a format usable by an LLM, then build a retrieval-augmented generation (RAG) pipeline to answer questions, including citations from the source material.
* **We will use a basic approach for this project. You will see more advanced ways to solve the same problem in next projects**.

## Python script: 008-conversational-rag.py
What we need to solve
* Store the chat conversation.
* When the user enters a new input, put that input in context.
* Re-phrase the user input to have a contextualized input.
* Send the contextualized input to the retriever.
* Use the retriever to build a conversational rag chain.
* Add extra features like persising memory (save memory in a file) and session memories.

The process we will follow
1. Create a basic RAG without memory.
2. Create a ChatPrompTemplate able to contextualize inputs.
3. Create a retriever aware of memory.
4. Create a basic conversational RAG.
5. Create an advanced conversational RAG with persistence and session memories.

## Python script: 009-simple-agent.py
Example of a LangChain Agent implementation (called LangGraph).
The agent is given a set of tools to perform a decision or action.
In this example, we give the agent a tool to query a search engine (called Tavily).
We then use the agent to answer questions by using this tool (call the search engine for results and use this context to answer the question). 

## Python script: 010-langserve-http-server.py
LangServe is the python equivalent to springboot or node (for javascript):
 - you can quickly launch an http server to test your LangChain app
 - it creates a simple UI and rest API which will call your LangChain app
 - can be deployed to any server that has python installed

## Python script: 011-streamlit-evaluate-app.py
Streamlit is a python library that allows you to quickly create an html page and test it by starting up an http server.
We will use streamlit to create a web page that calls our LangChain app.

From your terminal:
streamlit run 011-streamlit-evaluate-app.py

You can now view the web app at:
http://localhost:8501/


LangChain evaluation Chain: the code in this script does the following:
 - upload a text file
 - text splitting, embedding and store in a vector db
 - create a retriever chain (RetrievalQA) which will ask the LLM questions about the uploaded text file
 - create an eval chain (QAEvalChain) which will take input:
     - a known answer to the question
     - the answer from the LLM retriever chain
   and will use an LLM to evaluate if the two answers are the same or not (semantically the same, doesn't have to be an exact text match)


