import os
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# ----------------------------------------------------------------------------------------
# SQL integration
#  * We will create a Q&A app over tabular data in databases.
#  * These app will allow us to **ask a question about the data in a database in natural language and get back an answer also in natural language**.
#  * Building Q&A systems of SQL databases requires executing model-generated SQL queries. There are inherent risks in doing this. **In production, make sure that your database connection permissions** are always scoped as narrowly as possible for your chain's needs.
# ----------------------------------------------------------------------------------------

sqlite_db_path = "data/street_tree_db.sqlite"
db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")

# We can create a simple chain that takes a question and does the following:
#  * Convert the question into a SQL query;
#  * Execute the query;
#  * Use the result to answer the original question.
# The first step in a SQL chain is to take the user input and convert it to a SQL query. LangChain comes with a built-in chain for this, `create_sql_query_chain`:
chain = create_sql_query_chain(llm, db)

# --------------
# step 1: convert the user question into an SQL query
user_question = "List the species of trees that are present in San Francisco"
response = chain.invoke({"question": user_question})

print("\n")
print("SQL query for question: " + user_question)
print(response)
print("----------")

# if you wanted to manually run the db query, you can do this:
#db.run(response)

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query
response = chain.invoke({"question": user_question})

print("Response for question (with query execution included): " + user_question)
print(response)
print("----------")


# --------------
# step 3: Translate the SQL response into a natural language response
# Now that we've got a way to generate and execute queries, we need to combine the original question and SQL query result with the chat model to generate a final answer in natural language.
# We can do this by passing question and result to the LLM like this:

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke({"question": "List the species of trees that are present in San Francisco"})

print("Response for question (passing question and result to the LLM): " + user_question)
print(response)
print("\n")


#### Let's review what is happening in the above chain.
#  -  The user asks a question (identified by the variable name "question").
#  -  We use RunnablePassthrough to get that "question" variable, and we use .assign() twice to get the other two variables required by the prompt template: "query" and "result".
#  -  With the first .assign(), the write_query chain has que question as input and the SQL query (identified by the variable name "query") as output.
#  -  With the second .assign(), the execute_query chain has the SQL query as input and the SQL query execution (identified by the variable name "result") as output.
#  -  The prompt template has the question (identified by the variable name "question"), the SQL query (identified by the variable name "query") and the SQL query execution (identified by the variable name "result") as input, and the final prompt as the output.
#  -  The chat model has the prompt as he input and the AIMessage with the response in natural language as the output.
#  -  The StrOutputParser has the AIMessage with the response in natural language as the input and the response in natural language as a string of text as the output.

#### About the role of .assign() in this chain
# In this exercise we have learned more about the .assign() built-in method for Runnables. We have seen that the .assign() method **allows us to include additional keys and values** based on existing input.
#  -  With the first .assign(), the write_query chain has the question as input and the SQL query (identified by the variable name "query") as output.
#  -  With the second .assign(), the execute_query chain has the SQL query as input and the SQL query execution (identified by the variable name "result") as output.
#
# The .assign() method allows us to add new keys and values to the output of a Runnable **while keeping the original keys and values intact**. See the process again:
#  -  We use RunnablePassthrough to get that "question" variable, and we use .assign() twice to get the other two variables required by the prompt template: "query" and "result".
#  -  With the first .assign(), the write_query chain has que question as input and the SQL query (identified by the variable name "query") as output.
#  -  With the second .assign(), the execute_query chain has the SQL query as input and the SQL query execution (identified by the variable name "result") as output.


