import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")


# ----------------------------------------------------------------------------------------
# How to build a simple Agent LLM App with LangGraph
#  * Very basic tool-using Agent LLM App with memory.
# ----------------------------------------------------------------------------------------

# Here’s how it works:
# 1. **Use the Language Model as a Brain**: The agent uses the language model to figure out which actions it should take, based on the information it has or what you ask it to do.
# 2. **Taking Action**: After deciding, the agent then goes ahead and does those actions.
# 3. **Learning and Adjusting**: Once the actions are done, the results can be given back to the language model. This helps the model check if everything is complete or if it needs to do something else.
# 
# So, essentially, LangChain helps turn a language model from just a tool for writing and answering into a system that can act and react, almost like a very simple robot brain.
# 
# Here we will build an agent that can interact with a search engine. You will be able to ask this agent questions, watch it calling the search tool, and have conversations with it.
#
# LangGraph is a library created by the LangChain team for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows.


from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(max_results=2)
response = search.invoke("Who are the top stars of the 2024 Eurocup?")

print("\n----------\n")
print("Who are the top stars of the 2024 Eurocup?")
print("\n----------\n")
print(response)
print("\n----------\n")

## Agents
#   * Agents use LLMs as reasoning engines to determine which actions to take.
#
## Tool-using Agent
#   * For this basic agent we will use just one tool. In next advanced projects, you will learn how to use agents with several tools.
#   * **Our tool of choice will be Tavily** - a search engine. 

tools = [search]
llm_with_tools = llm.bind_tools(tools)

## Create the agent
#   * We will be using LangGraph to construct the agent. 
#   * **Note that below we are passing in the origina chat model, not the llm_with_tools we built later**. That is because create_tool_calling_executor will call .bind_tools for us under the hood.

from langgraph.prebuilt import create_react_agent
agent_executor = create_react_agent(llm, tools)

# Run the Agent
#   Let's first try it with .invoke():
from langchain_core.messages import HumanMessage
response = agent_executor.invoke({"messages": [HumanMessage(content="Where is the soccer Eurocup 2024 played?")]})

print("\n----------\n")
print("Where is the soccer Eurocup 2024 played? (agent)")
print("\n----------\n")
print(response["messages"])
print("\n----------\n")

#   Now let's try it with .stream():
print("\n----------\n")
print("When and where will it be the 2024 Eurocup final match? (agent with streaming)")
print("\n----------\n")
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="When and where will it be the 2024 Eurocup final match?")]}
):
    print(chunk)
    print("----")

print("\n----------\n")


## Adding memory
#   * Adding memory in LangGraph is very similar to what we did with LangChain.

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

# Let's create our new agent with memory and set one thread_id so the agent can create a memory for each session as we did with our previous conversational RAG app:
agent_executor = create_react_agent(llm, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "001"}}

print("Who won the 2024 soccer Eurocup?")

# Let's now try this new agent with .stream():
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="Who won the 2024 soccer Eurocup?")]}, config
):
    print(chunk)
    print("----")

print("Who were the top stars of that winner team?")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="Who were the top stars of that winner team?")]}, config
):
    print(chunk)
    print("----")

print("(With new thread_id) About what soccer team we were talking?")

config = {"configurable": {"thread_id": "002"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="About what soccer team we were talking?")]}, config
):
    print(chunk)
    print("----")