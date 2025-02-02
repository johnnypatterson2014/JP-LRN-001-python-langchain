import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

def convert_vector_docs_to_string(docs):
    return "\n\n".join([d.page_content for d in docs])

llm = ChatOpenAI(model="gpt-4o-mini")

# Load a document that you want to ask the LLM about
loaded_document = TextLoader('./data/state_of_the_union.txt').load()

# split the text into chunks - a simple text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks_of_text = text_splitter.split_documents(loaded_document)

#text_splitter = CharacterTextSplitter(
#    separator="\n\n",
#    chunk_size=1000,
#    chunk_overlap=200,
#    length_function=len,
#    is_separator_regex=False,
#)

# split the text into chunks - recursive text splitter
#  - The "Recursive Character Splitter" is a method used to divide text into smaller, more manageable chunks, designed specifically to maintain the semantic integrity of the text.
#  - It operates by attempting to split the text using a list of characters in a specified order—beginning with the largest units like paragraphs, then moving to sentences, and finally to individual words if needed.
#  - The default sequence for splitting is ["\n\n", "\n", " ", ""], which means it first tries to split the text at double newline characters to separate paragraphs, then at single newlines for any remaining large blocks, followed by spaces to isolate sentences or phrases, and finally using an empty string if finer splitting is necessary.
#  - This method is particularly effective because it tries to keep text chunks as meaningful and complete as possible, ensuring that each chunk has a coherent piece of information.
#default_recursive_splitter = RecursiveCharacterTextSplitter(
#    chunk_size=26,
#    chunk_overlap=4
#)

# customized recursive splitter
#custom_recursive_splitter = RecursiveCharacterTextSplitter(
#    chunk_size=150,
#    chunk_overlap=0,
#    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
#)
#chunks_of_text = default_recursive_splitter.split_documents(loaded_document)

# Embeddings - transform the chunks of text into numbers (vectors) so it can be stored in a vector db
embeddings = OpenAIEmbeddings()

# Use a vector db. Chroma is one option for a vector db implementation
# vector_db = Chroma.from_documents(chunks_of_text, OpenAIEmbeddings())
# response = vector_db.similarity_search(question)

# Instead of working with the vector db directly, it is better to use a retriever (retriever uses a vector db under the hood)
vector_db = FAISS.from_documents(chunks_of_text, embeddings)
retriever = vector_db.as_retriever()
# alternatively, if you want to limit the number of docs retrieved from the vector db, use:
# retriever = vector_db.as_retriever(search_kwargs={"k": 1})
user_question = "What did the president say about the John Lewis Voting Rights Act?"
documents_from_vector_db = retriever.invoke(user_question)
#print("Relevant docs from vector db: ")
#print(documents_from_vector_db)

vector_db_context = convert_vector_docs_to_string(documents_from_vector_db)

prompt_template = """Answer the question based only on the following context:

{context}

Question: {question}
"""

# --------------------------------------------------
# Call LLM without using LCEL (legacy LangChain way)
# --------------------------------------------------
chat_template = ChatPromptTemplate.from_messages(
    [
        ("human", prompt_template),
    ]
)

messages = chat_template.format_messages(
    question=user_question,
    context=vector_db_context
)
response = llm.invoke(messages)

print("\n----------")
print("Given the data in the vector db, answer the following question (non-LCEL):")
print("Question: " + user_question)
print("Answer from LLM: ")
print(response)
print("----------\n")

# --------------------------------------------------
# Call LLM using LCEL 
# --------------------------------------------------

# Most used LCEL runnables:
# - RunnablePassthrough - does not modify the input
# - RunnableLambda - To use a custom function inside a LCEL chain we need to wrap it up with RunnableLambda
# - RunnableParallel - This is probably the most important and most useful Runnable from LangChain
#     - you define a list of operations and they will be executed in parallel
#
# eg. 
# chain = RunnableParallel(
#     {
#         "operation_a": RunnablePassthrough(),
#         "operation_b": RunnableLambda(some_custom_function_name),
#         "operation_c": RunnablePassthrough(),
#     }
# ) | prompt | model | output_parser
#
# eg. 
# chain = (
#     RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
#     | prompt
#     | model
#     | StrOutputParser()
# )

prompt = ChatPromptTemplate.from_template(prompt_template)

chain = (
    {"context": retriever | convert_vector_docs_to_string, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response_lcel = chain.invoke(user_question)

print("\n----------")
print("Given the data in the vector db, answer the following question (using LCEL):")
print("Question: " + user_question)
print("Answer from LLM: ")
print(response_lcel)
print("----------\n")