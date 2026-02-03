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
    file_path="dataset2.md",
    encoding="utf-8"
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 650, chunk_overlap = 100)
chunks= text_splitter.split_documents(documents) 

#######################################################################################################
#######################################################################################################

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
# embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-large-en-v1.5")

db_name = "vector_db3"

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

# embedding = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2"
# )

# vectordb = Chroma(
#     persist_directory="./vector_db1",
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

Your task is to analyze the retrieved context and summarize TEAM COMPOSITIONS AND SETUPS.

Definitions:
- A "composition" refers to the combination of agents and their roles used by a team.
- A "setup" refers to the initial positioning, role assignments, or utility deployment patterns at the start of a round.
- Setups may apply to attack, defense, pistol rounds, bonus rounds, or specific economy states.
- Ignore mid-round adaptations unless they are explicitly described as part of the initial setup.

Your responsibilities:
1. Identify agent compositions used by the team(s).
2. Identify recurring setups associated with those compositions.
3. Group identical or near-identical compositions under a single entry.
4. Associate setups with the correct side, map, or round type when mentioned.

Output format:
- Return a concise summary of compositions and setups.
- For each entry, include:
  - Agent Composition
  - Evidence Snippet(s) from the retrieved text
  - Applicable Context (Map, Side, Round Type, or Economy state if mentioned)

Constraints:
- Do NOT invent compositions or setups not supported by the retrieved context.
- Do NOT infer positioning or roles unless explicitly stated or clearly repeated.
- Do NOT provide analysis, evaluation, or strategic advice.
- Do NOT summarize entire documents.
- If no clear compositions or setups are present, explicitly state:
  "No consistent compositions or setups identified."

Tone:
- Analytical, neutral, precise.

"""

def answer_question(question: str):
    docs = retreiver.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    return response.content

print(answer_question("Summarize Compositions & Setups for 100 Thieves"))

# response = llm.invoke("Who are you?")
# print(response.content)
