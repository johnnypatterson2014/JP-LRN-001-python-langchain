import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, CSVLoader, UnstructuredHTMLLoader, PyPDFLoader, WikipediaLoader
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

#openai_api_key = os.environ["OPENAI_API_KEY"]
chat_model = ChatOpenAI(model="gpt-4o-mini")

# Example to load data from a text file
loader = TextLoader("./data/be-good.txt")
loaded_data = loader.load()
#print(loaded_data)

# Example to load data from a csv file
loader = CSVLoader('./data/Street_Tree_List.csv')
loaded_data = loader.load()
#print(loaded_data)

# Example to load data from a local html file
loader = UnstructuredHTMLLoader('./data/100-startups.html')
loaded_data = loader.load()
#print(loaded_data)

# Example to load data from a pdf file
loader = PyPDFLoader('./data/5pages.pdf')
loaded_data = loader.load_and_split()
#print(loaded_data[0].page_content)

# Example to load data from a Wikipedia page
name = "JFK"
loader = WikipediaLoader(query=name, load_max_docs=1)
loaded_data = loader.load()[0].page_content

# Example to ask the LLM a question about the loaded data
# Note that the entire text in loaded_data is passed to the LLM in the context window.
chat_template = ChatPromptTemplate.from_messages(
    [
        ("human", "Answer this question: {question}. Here is some extra context: {context}"),
    ]
)

messages = chat_template.format_messages(
    question="What was the full name of JFK?",
    context=loaded_data
)
response = chat_model.invoke(messages)

print("\n----------")
print("Given the data from Wikipedia using the search 'JFK', answer the following question:")
print("Question: What was the full name of JFK?")
print("Answer from LLM: " + response.content)
print("----------\n")
