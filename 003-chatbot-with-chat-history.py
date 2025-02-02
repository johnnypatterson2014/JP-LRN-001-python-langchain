import warnings
from langchain._api import LangChainDeprecationWarning
from langchain_openai import ChatOpenAI
from langchain import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o-mini")
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)


# ----------------------------------------------------------------------------------------
# Chatbot with memory
#  - will remember previous chat history
#  - chat history is saved to a file
#  - note that the entire chat history is sent to the LLM context window.
# ----------------------------------------------------------------------------------------

# save the chat history in a json file
chat_history_file = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True
)

prompt_template = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

# This is a legacy LangChain chain (not the LCEL version)
chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=chat_history_file
)

print("----------")
response = chain.invoke("hello!")
print(response)
print("----------")
response = chain.invoke("my name is Julio")
print(response)
print("----------")
response = chain.invoke("what is my name?")
print(response)
print("----------")

# print the chat message history
loader = TextLoader("./messages.json")
loaded_data = loader.load()
print("Contents of messages.json file:")
#print(loaded_data)

# ----------------------------------------------------------------------------------------
# Example Chatbot that saves chat history by session_id
# ----------------------------------------------------------------------------------------

# create an empty dictionary
chatbotMemory = {}

# create a method to get chat history by session_id
# input: session_id, output: chatbotMemory[session_id]
# note that: BaseChatMessageHistory is an abstract base class
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]

chatbot_with_message_history = RunnableWithMessageHistory(
    llm, 
    get_session_history
)

session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite color is red.")],
    config=session1,
)

print(responseFromChatbot.content)
print("----------")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my favorite color?")],
    config=session1,
)

print(responseFromChatbot.content)
print("----------")

session2 = {"configurable": {"session_id": "002"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my favorite color?")],
    config=session2,
)

print(responseFromChatbot.content)
print("----------")


# ----------------------------------------------------------------------------------------
# Example Chatbot with history but limits the size of the history passed to the context window
#  - The memory of a chatbot is included in the context window of the LLM so, if left unmanaged, can potentially overflow it.
#  - We are now going to learn how to limit the size of the memory of a chatbot
# ----------------------------------------------------------------------------------------

print(chatbotMemory)

# option 1: limit the number of chat history messages sent to the LLM

def limited_memory_of_messages(messages, number_of_messages_to_keep=2):
    return messages[-number_of_messages_to_keep:]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

limitedMemoryChain = (
    RunnablePassthrough.assign(messages=lambda x: limited_memory_of_messages(x["messages"]))
    | prompt 
    | llm
)

# the lambda function is used to create a small anonymous function in Python. The lambda function defined here takes one argument, `x`.
# The argument `x` is expected to be a dictionary that contains a key named `"messages"`. The value associated with this key is a list of messages.
# The body of the `lambda` function calls the `limited_memory_of_messages` function. It passes the list of messages found in `x["messages"]` to this function.
# In essence, the `lambda` function is a shorthand way to apply the `limited_memory_of_messages` function to the message list contained within a dictionary. It automatically trims the list to the last two messages.

chatbot_with_limited_message_history = RunnableWithMessageHistory(
    limitedMemoryChain,
    get_session_history,
    input_messages_key="messages",
)

# add a couple more chat messages
responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite vehicles are Vespa scooters.")],
    config=session1,
)

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite city is San Francisco.")],
    config=session1,
)

# The chatbot memory has now 4 messages. Let's check the Chatbot with limited memory.
# Remember, this chatbot only remembers the last 2 messages, so if we ask her about the first message she should not remember it.

responseFromChatbot = chatbot_with_limited_message_history.invoke(
    {
        "messages": [HumanMessage(content="what is my favorite color?")],
    },
    config=session1,
)

print(responseFromChatbot.content)
print("----------")