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

# team_name = input("Enter team name: ").strip()


# from langchain_text_splitters import MarkdownHeaderTextSplitter

# loader = TextLoader(
#     file_path="dataset0.md",
#     encoding="utf-8"
# )

# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 650, chunk_overlap = 100)
# chunks= text_splitter.split_documents(documents) 

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
# print(llm.invoke("Who is 100 Thieves?"))

def answer_question(question: str):
    docs = retreiver.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    return response.content

print(answer_question(f"Identify Common Team-wide Strategies for {team_name}"))
# response = llm.invoke("Who are you?")
# print(response.content)

#########################################################################################################################3

##########################################################################################################################

##########################################################################################################################



# loader2 = TextLoader(
#     file_path="dataset1.md",
#     encoding="utf-8"
# )

# documents2 = loader2.load()

# text_splitter2 = RecursiveCharacterTextSplitter(chunk_size = 650, chunk_overlap = 100)
# chunks2= text_splitter2.split_documents(documents) 

#######################################################################################################
#######################################################################################################

# embeddings2 = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
# # embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-large-en-v1.5")

# db_name2 = "vector_db2"

# if os.path.exists(db_name2):
#     Chroma(persist_directory=db_name2, embedding_function=embeddings2).delete_collection()
    
# vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

embedding2 = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectordb2 = Chroma(
    persist_directory="./vector_db2",
    embedding_function=embedding2
)
######################################################################################################3
#######################################################################################################

retreiver2 = vectordb2.as_retriever(search_type="similarity",search_kwargs={"k": 3})
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3-pro-preview",
#     temperature=0.3,
#     max_output_tokens=512,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )
llm2 = ChatOpenAI(
    model="stepfun/step-3.5-flash:free",  # example
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
)

#######################################################################################################

SYSTEM_PROMPT_TEMPLATE2 = """
You are an Information Retrieval agent specialized in tactical analysis of the game Valorant.

You are given retrieved reference material delimited below.

--- RETRIEVED CONTEXT START ---
{context2}
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

def answer_question2(question: str):
    docs2 = retreiver2.invoke(question)
    context2 = "\n\n".join(doc.page_content for doc in docs2)
    system_prompt2 = SYSTEM_PROMPT_TEMPLATE2.format(context2=context2)
    response2 = llm2.invoke([SystemMessage(content=system_prompt2), HumanMessage(content=question)])
    return response2.content

print(answer_question2(f"Highlight Key Player Tendencies for {team_name}"))

# response = llm.invoke("Who are you?")
# print(response.content)




#########################################################################################################################3

##########################################################################################################################

##########################################################################################################################

# import os
# import glob

# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_openai import ChatOpenAI

# import gradio as gr

# load_dotenv(override=True)

# from langchain_community.document_loaders import TextLoader

# loader = TextLoader(
#     file_path="dataset2.md",
#     encoding="utf-8"
# )

# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 650, chunk_overlap = 100)
# chunks= text_splitter.split_documents(documents) 

#######################################################################################################
#######################################################################################################

# embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
# # embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-large-en-v1.5")

# db_name = "vector_db3"

# if os.path.exists(db_name):
#     Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    
# vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

embedding3 = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectordb3 = Chroma(
    persist_directory="./vector_db3",
    embedding_function=embedding3
)
######################################################################################################3
#######################################################################################################

retreiver3 = vectordb3.as_retriever(search_type="similarity",search_kwargs={"k": 3})
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3-pro-preview",
#     temperature=0.3,
#     max_output_tokens=512,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )
llm3 = ChatOpenAI(
    model="stepfun/step-3.5-flash:free",  # example
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
)

#######################################################################################################

SYSTEM_PROMPT_TEMPLATE3 = """
You are an Information Retrieval agent specialized in tactical analysis of the game Valorant.

You are given retrieved reference material delimited below.

--- RETRIEVED CONTEXT START ---
{context3}
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

def answer_question3(question: str):
    docs3 = retreiver3.invoke(question)
    context3 = "\n\n".join(doc.page_content for doc in docs3)
    system_prompt3 = SYSTEM_PROMPT_TEMPLATE3.format(context3=context3)
    response3 = llm3.invoke([SystemMessage(content=system_prompt3), HumanMessage(content=question)])
    return response3.content

# print(answer_question3(f"Summarize Compositions & Setups for {team_name}"))

# response = llm.invoke("Who are you?")
# print(response.content)
###################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################

llm4 = ChatOpenAI(
    model="nvidia/nemotron-nano-9b-v2:free",  # example
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
)
SCOUTING_REPORT_PROMPT ="""
You are a report-generation agent specialized in Valorant competitive scouting reports.

You are given three pre-extracted inputs:
1. Team-Wide Strategies
2. Key Player Tendencies
3. Compositions & Setups

These inputs were produced by strict information retrieval agents and must be treated as factual.

Your task is to generate a clean, well-structured SCOUTING REPORT using ONLY the provided inputs.

Rules:
- Do NOT invent new information.
- Do NOT infer intent beyond what is written.
- Do NOT add advice, counter-strategies, or evaluation.
- Do NOT merge or reinterpret sections.
- Preserve the meaning and constraints of each input.

Formatting requirements:
- Use clear section headers.
- Each section must correspond exactly to one input.
- Keep language concise, professional, and analytical.
- If any input explicitly states that no information was identified, avoid including it in the report and dont even include that information missing from the context.

Report structure:

SCOUTING REPORT — {team_name}

SECTION 1: Team-Wide Strategies
<Formatted content based only on Input 1>

SECTION 2: Key Player Tendencies
<Formatted content based only on Input 2>

SECTION 3: Compositions & Setups
<Formatted content based only on Input 3>

Tone:
- Neutral
- Analytical
- Professional

"""

def generate_scouting_report(team_name, strategies, tendencies, comps):
    system_prompt = SCOUTING_REPORT_PROMPT.format(team_name=team_name)

    human_message = f"""
INPUT 1 — TEAM-WIDE STRATEGIES:
{strategies}

INPUT 2 — KEY PLAYER TENDENCIES:
{tendencies}

INPUT 3 — COMPOSITIONS & SETUPS:
{comps}
"""

    response = llm4.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ])

    return response.content

def run_scouting_report(team_name: str):
    strategies = answer_question(f"Identify common team-wide strategies for {team_name}")
    tendencies = answer_question2(f"Highlight key player tendencies for {team_name}")
    comps = answer_question3(f"Summarize compositions and setups for {team_name}")

    report = generate_scouting_report(
        team_name,   # ✅ team name from user is passed here
        strategies,
        tendencies,
        comps
    )

    return report


gr.Interface(
    fn=run_scouting_report,   # ✅ pass function reference
    inputs=gr.Textbox(
        label="Team Name",
        placeholder="e.g. 100 Thieves"
    ),
    outputs=gr.Markdown(label="Scouting Report"),
    submit_btn="Generate Report"
).launch(inbrowser=True)


# strategies = answer_question(f"Identify common team-wide strategies for {team_name}")
# tendencies = answer_question2(f"Highlight key player tendencies for {team_name}")
# comps = answer_question3(f"Summarize compositions and setups for {team_name}")

# report = generate_scouting_report(
#     team_name,
#     strategies,
#     tendencies,
#     comps
# )

# print(report)