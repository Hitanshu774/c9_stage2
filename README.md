# 🎮 Valorant Competitive Scouting Report Generator

A **multi-agent RAG system** that auto-generates professional scouting reports for Valorant esports teams — analyzing team-wide strategies, key player tendencies, and agent compositions from match performance data using specialized AI agents and semantic retrieval.

Covers **12 professional teams** including Cloud9, 100 Thieves, LOUD, Sentinels, G2, FURIA, NRG, Evil Geniuses, and more.

---

## ✨ Features

- **Multi-Agent Architecture** — 3 specialized Information Retrieval agents, each with a dedicated vector store and constrained system prompt, feeding into a final synthesis agent
- **Role-Specialized Agents** — Strategy Analyst, Player Tendency Analyst, and Composition Analyst operate independently before being aggregated
- **Constrained IR Prompting** — each agent is strictly forbidden from inventing information, ensuring all output is grounded in retrieved context
- **Final Synthesis Agent** — a dedicated report-generation LLM assembles all three agent outputs into a clean, structured scouting report
- **Gradio Web UI** — enter any team name and receive a formatted markdown scouting report in seconds
- **Dockerized** — fully containerized for reproducible deployment

---

## 🏗️ Multi-Agent Architecture

```
                        [User: Team Name]
                               │
               ┌───────────────┼───────────────┐
               │               │               │
               ▼               ▼               ▼
      ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
      │  Agent 1    │  │  Agent 2    │  │  Agent 3    │
      │  Strategy   │  │  Tendency   │  │  Composition│
      │  Analyst    │  │  Analyst    │  │  Analyst    │
      └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
             │                │                │
      ChromaDB (DB1)   ChromaDB (DB2)   ChromaDB (DB3)
      dataset0.md      dataset1.md      dataset2.md
             │                │                │
             ▼                ▼                ▼
      Team Strategies   Player Tendencies  Compositions
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Synthesis Agent    │
                   │  (Report Generator) │
                   │  Nemotron-nano-9b   │
                   └─────────────────────┘
                              │
                              ▼
                   Structured Scouting Report
                       (Gradio UI Output)
```

---

## 🤖 Agent Breakdown

### Agent 1 — Strategy Analyst
- **Vector Store:** `vector_db1` (dataset0.md — team performance metrics)
- **Task:** Extracts recurring **team-wide strategies** — attack patterns, defense setups, economy decisions, executions, rotations, and mid-round adaptations
- **Output format:** Strategy Name → Description → Evidence Snippet → Context (map/side/economy)

### Agent 2 — Player Tendency Analyst
- **Vector Store:** `vector_db2` (dataset1.md — player behavior data)
- **Task:** Identifies repeatable **individual player patterns** — positioning habits, utility timing, aggression tendencies, rotation speed, anchoring behavior
- **Output format:** Player Name → Tendency Name → Description → Evidence → Context

### Agent 3 — Composition Analyst
- **Vector Store:** `vector_db3` (dataset2.md — agent composition data)
- **Task:** Summarizes **agent compositions and round setups** — side-specific compositions, pistol round defaults, economy-based lineups
- **Output format:** Agent Composition → Evidence → Context (map/side/round type)

### Agent 4 — Synthesis / Report Generator
- **Model:** `nvidia/nemotron-nano-9b-v2` via OpenRouter
- **Task:** Takes structured outputs from Agents 1–3 and generates a final **SCOUTING REPORT** with clean section headers, neutral analytical tone, and no invented content

---

## 📊 Teams Covered

| Team | Region |
|---|---|
| Cloud9 | NA |
| 100 Thieves | NA |
| NRG | NA |
| Sentinels | NA |
| Evil Geniuses | NA |
| LOUD | BR |
| FURIA | BR |
| MIBR | BR |
| G2 | EU |
| KRÜ Esports | LATAM |
| Leviatán Esports | LATAM |
| 2GAME eSports | LATAM |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Document Loading | LangChain `TextLoader` |
| Text Splitting | `RecursiveCharacterTextSplitter` (chunk=650, overlap=100) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Stores | 3× ChromaDB (persisted, per agent) |
| LLM — IR Agents | `stepfun/step-3.5-flash` via OpenRouter |
| LLM — Synthesis | `nvidia/nemotron-nano-9b-v2` via OpenRouter |
| UI | Gradio `Interface` |
| Containerization | Docker |
| Language | Python 3.13 |

---

## 🚀 Getting Started

### Option A — Run with Docker

```bash
git clone https://github.com/Hitanshu774/c9_stage2.git
cd c9_stage2

# Build the image
docker build -t valorant-scout .

# Run with your API key
docker run -e API_KEY=your_openrouter_key -p 7860:7860 valorant-scout
```

Then open `http://localhost:7860` in your browser.

### Option B — Run Locally

```bash
git clone https://github.com/Hitanshu774/c9_stage2.git
cd c9_stage2

python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

Create a `.env` file:

```env
API_KEY=your_openrouter_api_key_here
```

Get a free key at [openrouter.ai](https://openrouter.ai) — the models used are free tier.

```bash
python app.py
```

The Gradio UI will launch at `http://localhost:7860`.

---

## 💡 Usage

1. Open the Gradio interface
2. Enter a team name in the text box (e.g. `Cloud9`, `LOUD`, `100 Thieves`)
3. Click **Generate Report**
4. Receive a structured scouting report with three sections:
   - Team-Wide Strategies
   - Key Player Tendencies
   - Compositions & Setups

### Example Output Structure

```
SCOUTING REPORT — CLOUD9

SECTION 1: Team-Wide Strategies
  - Late-contact Default: Cloud9 consistently delays first contact...
  - Elite Site Execution: Site Hit Frequency of 0.736 indicates...

SECTION 2: Key Player Tendencies
  - [Player]: Aggressive anchor tendency on defense...

SECTION 3: Compositions & Setups
  - Agent Composition: [agents used]...
```

---

## 🗂️ Project Structure

```
c9_stage2/
├── app.py                  # Main multi-agent pipeline
├── dataset0.md             # Team strategy performance data
├── dataset1.md             # Player tendency data
├── dataset2.md             # Agent composition data
├── vector_db1/             # Persisted ChromaDB — Strategy Agent
├── vector_db2/             # Persisted ChromaDB — Tendency Agent
├── vector_db3/             # Persisted ChromaDB — Composition Agent
├── requirements.txt
├── Dockerfile
└── .gitignore
```

---

## 🔮 Future Enhancements

- [ ] LangGraph-based orchestration for dynamic agent routing
- [ ] Real-time data ingestion from VCT match APIs
- [ ] Head-to-head matchup comparison between two teams
- [ ] LangSmith tracing for agent evaluation and prompt versioning
- [ ] Export reports as PDF

---

## 👤 Author

**Hitanshu** — M.Tech Artificial Intelligence, IIIT Lucknow

Specializing in multi-agent AI systems, RAG pipelines, LangGraph, LangSmith, and LLM deployment.

[![GitHub](https://img.shields.io/badge/GitHub-Hitanshu774-black?logo=github)](https://github.com/Hitanshu774)
