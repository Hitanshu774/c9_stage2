import os
import glob

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

import gradio as gr

load_dotenv(override=True)

from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    file_path="example.md",
    encoding="utf-8"
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
chunks= text_splitter.split_documents(documents) 

#######################################################################################################
#######################################################################################################

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
# embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-large-en-v1.5")

db_name = "vector_db1"

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

######################################################################################################3
#######################################################################################################

retreiver = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 3})
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3-flash-preview",
#     temperature=0.3,
#     max_output_tokens=512,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )
llm = ChatOpenAI(
    model="gemini-3-pro-preview",  # example
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3,
    max_tokens=512,
)