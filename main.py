import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS          # ✅ modern import
from langchain.chains import RetrievalQA


# Hardcoded sample docs
docs = [
    Document(page_content="The company offers 20 days of paid leave per year."),
    Document(page_content="Employees can work remotely up to 3 days a week."),
    Document(page_content="Health insurance is provided to all full-time staff.")
]

# Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

# Embeddings
embeddings = AzureOpenAIEmbeddings(
    model="gpt-4",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview"
)

# Vector store
db = FAISS.from_documents(chunks, embeddings)

# Retrieval + QA
retriever = db.as_retriever()
llm = AzureOpenAI(
    model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview",
    temperature=0
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# Query
print(qa.run("What is the company leave policy?"))
