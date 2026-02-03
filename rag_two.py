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
    file_path="dataset1.md",
    encoding="utf-8"
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 650, chunk_overlap = 100)
chunks= text_splitter.split_documents(documents) 

#######################################################################################################
#######################################################################################################

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
# embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-large-en-v1.5")

db_name = "vector_db2"

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

# embedding = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2"
# )

# vectordb = Chroma(
#     persist_directory="./vector_db2",
#     embedding_function=embedding
# )
######################################################################################################3
#######################################################################################################

retreiver = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 3})
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3-pro-preview",
#     temperature=0.3,
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

Your task is to analyze the retrieved context and identify KEY PLAYER TENDENCIES.

Definitions:
- A "player tendency" is a repeatable, individual behavior pattern exhibited by a specific player.
- Tendencies may involve positioning, utility usage, aggression timing, role fulfillment, rotation speed, anchoring habits, or decision-making under similar conditions.
- Tendencies must be consistent across multiple rounds or situations.
- Ignore one-off plays or isolated highlights unless clearly repeated.

Your responsibilities:
1. Identify recurring behavioral patterns tied to specific players.
2. Attribute each tendency to the correct player whenever possible.
3. Group similar actions under a single tendency label.
4. Focus on habits and preferences, not success or failure of the play.

Output format:
- Return a concise list of player tendencies.
- For each tendency, include:
  - Player Name (or Identifier if unnamed)
  - Tendency Name
  - Short Description (1–2 sentences)
  - Evidence Snippet(s) from the retrieved text
  - Applicable Context (Map, Side, Agent, Role, or Economy state if mentioned)

Constraints:
- Do NOT invent tendencies not supported by the retrieved context.
- Do NOT provide advice, counterplay, or evaluation.
- Do NOT generalize team behavior as a player tendency.
- Do NOT summarize entire documents.
- If no clear player tendencies are present, explicitly state:
  "No consistent player tendencies identified."

Tone:
- Analytical, neutral, precise.

"""

def answer_question(question: str):
    docs = retreiver.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    return response.content

print(answer_question("Highlight Key Player Tendencies for 100 Thieves"))

# response = llm.invoke("Who are you?")
# print(response.content)
