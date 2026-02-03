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

# from langchain_text_splitters import MarkdownHeaderTextSplitter

loader = TextLoader(
    file_path="dataset0.md",
    encoding="utf-8"
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 650, chunk_overlap = 100)
chunks= text_splitter.split_documents(documents) 

#######################################################################################################
#######################################################################################################

# embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
# # embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-large-en-v1.5")

# db_name = "vector_db1"

# if os.path.exists(db_name):
#     Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    
# vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="./vector_db1",
    embedding_function=embedding
)
######################################################################################################3
#######################################################################################################

retreiver = vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 3})
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3-pro-preview",
#     temperature=0.1,
#     max_output_tokens=512,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )
llm = ChatOpenAI(
    model="stepfun/step-3.5-flash:free",  # example
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
)

#######################################################################################################

SYSTEM_PROMPT_TEMPLATE = """
You are an Information Retrieval agent specialized in tactical analysis of the game Valorant.

You are given retrieved reference material delimited below.

--- RETRIEVED CONTEXT START ---
{context}
--- RETRIEVED CONTEXT END ---

Your task is to analyze the retrieved context and identify COMMON TEAM-WIDE STRATEGIES.

Definitions:
- A "team-wide strategy" is a coordinated, repeatable pattern involving multiple players, roles, or utility usage.
- Strategies may relate to attack, defense, mid-round adaptations, defaults, executions, rotations, or economy-based decisions.
- Ignore individual mechanical plays unless they are part of a broader team pattern.

Your responsibilities:
1. Extract and identify recurring strategic patterns across rounds or matches.
2. Group similar behaviors under a single strategy label when applicable.
3. Focus on intent and structure (e.g., default → probe → late exec), not raw outcomes.
4. Prefer strategies that are explicitly stated OR strongly implied through repetition.

Output format:
- Return a concise list of strategies.
- For each strategy, include:
  - Strategy Name
  - Short Description (1–2 sentences)
  - Evidence Snippet(s) from the retrieved text
  - Applicable Context (Map, Side, Agent Composition, or Economy state if mentioned)

Constraints:
- Do NOT invent strategies not supported by the retrieved context.
- Do NOT provide coaching advice or counter-strategies.
- Do NOT summarize entire documents.
- If no clear team-wide strategy is present, explicitly state:
  "No consistent team-wide strategy identified."

Tone:
- Analytical, neutral, precise.

"""

# print(retreiver.invoke("Who is 100 Thieves?"))
print(llm.invoke("Who is 100 Thieves?"))

# def answer_question():
#     docs = retreiver.invoke("team coordination strategy patterns")
#     context = "\n\n".join(doc.page_content for doc in docs)
#     system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
#     response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content="Hello, who are you")])
#     return response.content

# print(answer_question())
# response = llm.invoke("Who are you?")
# print(response.content)

