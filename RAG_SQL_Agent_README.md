# RAG SQL Agent

A production-grade, end-to-end RAG SQL Agent that accepts data files,
generates SQL from natural language, self-corrects errors, and interprets results.
Powered by OpenRouter (any LLM) + DuckDB + ChromaDB + Streamlit.

## Quick Start

### 1. Clone and Install
```bash
cd rag_sql_agent
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
# Get a key at: https://openrouter.ai/keys
```

### 3. Run
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Supported File Formats
- CSV (auto-detects separator and encoding)
- TSV (tab-separated)
- Excel (.xlsx and .xls – all sheets loaded as separate tables)
- JSON (array or nested object format)
- JSON Lines (.jsonl)
- Parquet

## How It Works
1. Upload your data file in the sidebar
2. The agent extracts schema intelligence and embeds it in ChromaDB (vector store)
3. Ask a question in plain English
4. The agent retrieves relevant schema context via semantic search (RAG)
5. Generates DuckDB SQL using your chosen LLM via OpenRouter
6. Executes SQL against DuckDB (in-memory, no server)
7. Self-corrects if SQL fails (up to 3 attempts)
8. Interprets results in natural language

## Architecture
- **OpenRouter**: LLM provider (supports GPT-4o, Claude, Gemini, Llama, etc.)
- **DuckDB**: In-memory SQL engine — no database setup needed
- **ChromaDB**: Vector store for schema RAG
- **sentence-transformers**: Local embeddings (all-MiniLM-L6-v2, runs on CPU)
- **Streamlit**: Web UI

## Project Structure
```
rag_sql_agent/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── .env.example
├── README.md
├── agent/
│   ├── sql_agent.py          # Core orchestrator
│   ├── sql_generator.py      # LLM-based SQL generation
│   ├── sql_executor.py       # DuckDB execution + self-correction
│   ├── result_interpreter.py # LLM result analysis
│   └── intent_classifier.py  # Route queries by intent
├── rag/
│   ├── schema_extractor.py   # Extract & describe schema from DataFrames
│   ├── vector_store.py       # ChromaDB wrapper
│   └── context_builder.py    # Build LLM-ready context from retrieved schema
├── data/
│   └── loader.py             # Multi-format file loader → pandas DataFrame
├── config/
│   └── settings.py           # Pydantic settings: API key, model, limits
└── utils/
    ├── sql_validator.py      # sqlglot-based syntax validation
    ├── formatters.py         # DataFrame → markdown tables, result summaries
    └── session_state.py      # Streamlit session state helpers
```

## Environment Variables
See `.env.example` for all configuration options.

| Variable | Default | Description |
|---|---|---|
| OPENROUTER_API_KEY | (required) | Your OpenRouter API key |
| DEFAULT_MODEL | openai/gpt-4o-mini | Default LLM model |
| MAX_RETRIES | 3 | SQL self-correction attempts |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | Local embedding model |
| CHROMA_PERSIST_DIR | ./vector_store_db | ChromaDB storage path |

## Available Models (via OpenRouter)
| Model | ID | Best For |
|---|---|---|
| GPT-4o Mini | openai/gpt-4o-mini | Fast SQL, high volume |
| GPT-4o | openai/gpt-4o | Complex multi-table SQL |
| Claude 3.5 Sonnet | anthropic/claude-3.5-sonnet | Best reasoning |
| Claude 3 Haiku | anthropic/claude-3-haiku | Ultra-fast responses |
| Gemini Flash 1.5 | google/gemini-flash-1.5 | Fast, large context |
| Llama 3.1 70B | meta-llama/llama-3.1-70b-instruct | Open weights |
