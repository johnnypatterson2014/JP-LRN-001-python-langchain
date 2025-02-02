import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", verbose=True)

# ----------------------------------------------------------------------------------------
# Conversational RAG
#  * In most RAG applications we want to allow the user to have a back-and-forth conversation, meaning the application needs some sort of "memory" of past questions and answers.
# ----------------------------------------------------------------------------------------

## What we need to solve
#   * Store the chat conversation.
#   * When the user enters a new input, put that input in context.
#   * Re-phrase the user input to have a contextualized input.
#   * Send the contextualized input to the retriever.
#   * Use the retriever to build a conversational rag chain.
#   * Add extra features like persising memory (save memory in a file) and session memories.

## The process we will follow
#   1. Create a basic RAG without memory.
#   2. Create a ChatPrompTemplate able to contextualize inputs.
#   3. Create a retriever aware of memory.
#   4. Create a basic conversational RAG.
#   5. Create an advanced conversational RAG with persistence and session memories.


# ----------------------------------------------------------------------------------------
## Step 1: Create a basic RAG without memory
#   * We will use the RAG process we already know.
#   * We will use create_stuff_documents_chain to build a qa chain: a chain able to asks questions to an LLM.
#   * We will use create_retrieval_chain and the qa chain to build the RAG chain: a chain able to asks questions to the retriever and then format the response with the LLM.

import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("./data/be-good.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "What is this article about?"})
print("\n----------\n")
print("What is this article about?")
print("\n----------\n")
print(response["answer"])
print("\n----------\n")

response = rag_chain.invoke({"input": "What was my previous question about?"})
print("\n----------\n")
print("What was my previous question about?")
print("\n----------\n")
print(response["answer"])
print("\n----------\n")

# As we can see in the following question, our app has no memory of the conversation.

# ----------------------------------------------------------------------------------------
## Step 2: Create a ChatPromptTemplate able to contextualize inputs
#   * Goal: put the input in context and re-phrase it so we have a contextualized input.
#   * We will define a new system prompt that instructs the LLM in how to contextualize the input.
#   * Our new ChatPromptTemplate will include:
#       * The new system prompt.
#       * MessagesPlaceholder, a placeholder used to pass the list of messages included in the chat_history.

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ----------------------------------------------------------------------------------------
## Step 3: Create a Retriever aware of the memory
#   * We will build our new retriever with create_history_aware_retriever that uses the contextualized input to get a contextualized response.

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# ----------------------------------------------------------------------------------------
## Step 4: Create a basic Conversational RAG
#   * We will use the retriever aware of memory, that uses the prompt with contextualized input.
#   * We will use create_stuff_documents_chain to build a qa chain: a chain able to asks questions to an LLM.
#   * We will use create_retrieval_chain and the qa chain to build the RAG chain: a chain able to asks questions to the retriever and then format the response with the LLM.

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

## Trying our basic conversational RAG
# Below we ask a question and a follow-up question that requires contextualization to return a sensible response. Because our chain includes a "chat_history" input, the caller needs to manage the chat history. We can achieve this by appending input and output messages to a list:

from langchain_core.messages import AIMessage, HumanMessage
chat_history = []
question = "What is this article about?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})

chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"]),
    ]
)

second_question = "What was my previous question about?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
print(ai_msg_2["answer"])

# ----------------------------------------------------------------------------------------
## Step 5: Advanced conversational RAG with persistence and session memories
#   * We will store the chat history in a python dictionary. In advanced apps, you will use advanced ways to store chat history.
#   * Associate chat history with user session with the function get_session_history().
#   * Inject chat history into inputs and update it after each interaction using BaseChatMessageHistory and RunnableWithMessageHistory.

# Let's now focus on how to handle and maintain chat history in the question and answer (Q&A) application to make conversations flow more naturally.
# 
# Here’s a simplified explanation:
# 
# 1. **Stateful Management of Chat History**: Instead of manually entering previous parts of the conversation every time a new input is made, the application can automatically manage and update chat history. This means that the application remembers past interactions and uses that information to understand and respond to new questions more accurately.
# 
# 2. **Tools for Managing Chat History**:
#    - **BaseChatMessageHistory**: This component is responsible for storing the history of the conversation.
#    - **RunnableWithMessageHistory**: This acts as a wrapper around the main processing chain (LCEL chain) and the chat history storage (BaseChatMessageHistory). It takes care of adding the historical chat data to new inputs and updating the stored history after each response.
# 
# 3. **How It Works**: When you use these components in an application:
#    - The application automatically retrieves and updates the chat history every time it processes a new input. This helps in maintaining a continuous and coherent conversation flow.
#    - When using `RunnableWithMessageHistory`, it manages chat history using a configuration that includes a unique session identifier (`session_id`). This identifier helps the system know which conversation history to retrieve and update whenever a user interacts with the system.
# 
# 4. **Alternative ways to store the chat history**: In our simple implementation, chat histories might be stored in a basic dictionary. More complex systems might use databases like Redis to ensure more reliable and long-term storage of conversation data.

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

response = conversational_rag_chain.invoke(
    {"input": "What is this article about?"},
    config={
        "configurable": {"session_id": "001"}
    },  # constructs a key "001" in `store`.
)

print("\n----------\n")
print("What is this article about?")
print("\n----------\n")
print(response["answer"])
print("\n----------\n")

response = conversational_rag_chain.invoke(
    {"input": "What was my previous question about?"},
    config={"configurable": {"session_id": "001"}},
)
print("\n----------\n")
print("What was my previous question about?")
print("\n----------\n")
print(response["answer"])
print("\n----------\n")

print("\n----------\n")
print("Conversation History:")
print("\n----------\n")

# The conversation history can be inspected in the store dict:
for message in store["001"].messages:
    if isinstance(message, AIMessage):
        prefix = "AI"
    else:
        prefix = "User"

    print(f"{prefix}: {message.content}\n")

print("\n----------\n")