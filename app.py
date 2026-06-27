"""
RAG SQL Agent – Streamlit Application Entry Point
Run with: streamlit run app.py

Architecture: Streamlit UI → SQLAgent → [RAG + SQLGenerator + SQLExecutor + ResultInterpreter]
All AI calls route through OpenRouter API (OpenAI-compatible).
"""
import streamlit as st
import pandas as pd
import logging
import os
from pathlib import Path

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load env vars
from dotenv import load_dotenv
load_dotenv()

# Internal imports
from agent.sql_agent import SQLAgent, AgentResponse
from data.loader import DataLoader, DataLoadError
from config.settings import settings, AVAILABLE_MODELS
from utils.session_state import SessionState
from utils.formatters import ResultFormatter


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG SQL Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

    .chat-user {
        background: #1a1a2e; border-radius: 12px; padding: 14px 18px;
        margin: 8px 0; border-left: 3px solid #4a9eff;
    }
    .chat-assistant {
        background: #0f2027; border-radius: 12px; padding: 14px 18px;
        margin: 8px 0; border-left: 3px solid #00c853;
    }
    .sql-block {
        background: #111827; border: 1px solid #1f2937; border-radius: 8px;
        padding: 12px 16px; font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.85rem; color: #a5f3fc; margin: 8px 0;
        white-space: pre-wrap; word-break: break-all;
    }
    .badge-success { background: #14532d; color: #86efac; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; }
    .badge-error   { background: #7f1d1d; color: #fca5a5; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; }
    .badge-repaired { background: #1e3a5f; color: #93c5fd; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; }

    section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #1f1f1f; }
    section[data-testid="stSidebar"] .block-container { padding: 1rem; }

    .stTextInput > div > div > input { background: #1a1a1a; color: #f0f0f0; border: 1px solid #333; border-radius: 8px; }
    .stTextArea > div > div > textarea { background: #1a1a1a; color: #f0f0f0; border: 1px solid #333; border-radius: 8px; }
    .stSelectbox > div > div { background: #1a1a1a; }
    [data-testid="stMetric"] { background: #111827; border-radius: 8px; padding: 10px; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
    hr { border-color: #1f2937; margin: 1rem 0; }
    .streamlit-expanderHeader { background: #111827 !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZE SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
SessionState.init()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## RAG SQL Agent")
    st.markdown("*Powered by OpenRouter*")
    st.divider()

    # API Key input
    st.markdown("### API Configuration")
    api_key_input = st.text_input(
        "OpenRouter API Key",
        value=SessionState.get(SessionState.API_KEY) or os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
        placeholder="sk-or-v1-...",
        help="Get your API key at openrouter.ai/keys",
    )
    if api_key_input:
        SessionState.set(SessionState.API_KEY, api_key_input)

    # Model selection
    model_display_names = list(AVAILABLE_MODELS.keys())
    current_model_id = SessionState.get(SessionState.MODEL)
    current_display = next(
        (k for k, v in AVAILABLE_MODELS.items() if v == current_model_id),
        model_display_names[0]
    )
    selected_display = st.selectbox(
        "AI Model",
        options=model_display_names,
        index=model_display_names.index(current_display) if current_display in model_display_names else 0,
        help="Different models have different strengths and costs.",
    )
    selected_model_id = AVAILABLE_MODELS[selected_display]
    SessionState.set(SessionState.MODEL, selected_model_id)

    st.divider()

    # File upload
    st.markdown("### Upload Data")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet", "jsonl"],
        accept_multiple_files=True,
        help="CSV, TSV, Excel, JSON, Parquet, JSON Lines supported",
        label_visibility="collapsed",
    )

    if uploaded_files:
        load_button = st.button(
            f"Load {len(uploaded_files)} File(s)",
            type="primary",
            use_container_width=True,
        )
        if load_button:
            api_key = SessionState.get(SessionState.API_KEY)
            if not api_key:
                st.error("Please enter your OpenRouter API key first.")
            else:
                agent = SessionState.get(SessionState.AGENT)
                if agent is not None:
                    agent.reset()

                agent = SQLAgent(
                    api_key=api_key,
                    model=selected_model_id,
                    vector_store_dir=settings.chroma_persist_dir,
                )

                loaded_count = 0
                errors = []

                with st.spinner("Loading and indexing data..."):
                    for uploaded_file in uploaded_files:
                        try:
                            tables = DataLoader.load_file(uploaded_file, uploaded_file.name)
                            for table_name, df in tables.items():
                                agent.load_data(table_name, df)
                                loaded_count += 1
                        except DataLoadError as e:
                            errors.append(f"{uploaded_file.name}: {str(e)}")
                        except Exception as e:
                            errors.append(f"{uploaded_file.name}: Unexpected error – {str(e)}")

                if errors:
                    for err in errors:
                        st.error(err)

                if loaded_count > 0:
                    SessionState.set(SessionState.AGENT, agent)
                    SessionState.set(SessionState.LOADED_TABLES, agent.get_loaded_tables())
                    SessionState.set(SessionState.FILES_LOADED, True)
                    SessionState.set(SessionState.LAST_RESULT_DF, None)
                    SessionState.add_message(
                        "assistant",
                        f"Loaded and indexed **{loaded_count} table(s)**: "
                        + ", ".join(f"`{t}`" for t in agent.get_loaded_tables())
                        + ". You can now ask questions about your data.",
                    )
                    st.success(f"Loaded {loaded_count} table(s) successfully!")

    # Loaded tables info
    loaded_tables = SessionState.get(SessionState.LOADED_TABLES)
    if loaded_tables:
        st.divider()
        st.markdown("### Loaded Tables")
        agent = SessionState.get(SessionState.AGENT)
        for table_name in loaded_tables:
            if agent:
                schema = agent.get_schema_dict(table_name)
                if schema:
                    st.markdown(
                        f"**`{table_name}`**  \n"
                        f"{schema['row_count']:,} rows · {schema['col_count']} cols"
                    )

    # Controls
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            SessionState.clear_chat()
            st.rerun()
    with col2:
        if st.button("Reset All", use_container_width=True):
            agent = SessionState.get(SessionState.AGENT)
            if agent:
                agent.reset()
            SessionState.set(SessionState.AGENT, None)
            SessionState.set(SessionState.LOADED_TABLES, [])
            SessionState.set(SessionState.FILES_LOADED, False)
            SessionState.clear_chat()
            st.rerun()

    st.divider()
    st.markdown(
        "<small style='color:#444'>RAG SQL Agent v1.0 · DuckDB · ChromaDB<br>"
        "Embeddings: sentence-transformers</small>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# RAG SQL Agent")
st.markdown(
    "Ask questions about your data in plain English. "
    "The agent writes SQL, executes it, and explains the results."
)

# Status bar
api_key = SessionState.get(SessionState.API_KEY)

col1, col2, col3 = st.columns(3)
with col1:
    if api_key:
        st.success("API Key: Connected")
    else:
        st.warning("API Key: Not set")
with col2:
    loaded = SessionState.get(SessionState.LOADED_TABLES)
    if loaded:
        st.success(f"Data: {len(loaded)} table(s) loaded")
    else:
        st.info("Data: No files loaded")
with col3:
    st.info(f"Model: {SessionState.get(SessionState.MODEL, 'Not selected')}")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    chat_history = SessionState.get(SessionState.CHAT_HISTORY)

    if not chat_history:
        st.markdown(
            """
            <div style='text-align:center; padding: 60px 20px; color: #444;'>
                <h3>Welcome to RAG SQL Agent</h3>
                <p>Upload a data file in the sidebar, then ask a question below.</p>
                <p style='font-size:0.85rem;'>Example: "Show me the top 10 rows by revenue" or "What is the average age by department?"</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        for i, msg in enumerate(chat_history):
            role = msg['role']
            content = msg['content']
            metadata = msg.get('metadata', {})

            if role == 'user':
                st.markdown(
                    f'<div class="chat-user"><strong>You</strong><br>{content}</div>',
                    unsafe_allow_html=True
                )
            else:
                with st.container():
                    # Status badges
                    if metadata:
                        success = metadata.get('success', True)
                        was_repaired = metadata.get('was_repaired', False)
                        rows = metadata.get('rows_returned', 0)

                        badge_parts = []
                        if success:
                            badge_parts.append('<span class="badge-success">Success</span>')
                        else:
                            badge_parts.append('<span class="badge-error">Error</span>')
                        if was_repaired:
                            badge_parts.append('<span class="badge-repaired">Auto-repaired</span>')
                        if success and rows > 0:
                            badge_parts.append(f'<span class="badge-success">{rows:,} rows</span>')

                        if badge_parts:
                            st.markdown(' '.join(badge_parts), unsafe_allow_html=True)

                    st.markdown(
                        f'<div class="chat-assistant"><strong>Agent</strong><br>{content}</div>',
                        unsafe_allow_html=True
                    )

                    # Show SQL if present
                    if metadata.get('sql_query'):
                        with st.expander("View SQL Query", expanded=False):
                            st.code(metadata['sql_query'], language='sql')

                    # Show result DataFrame for last message only
                    if metadata.get('has_result_df') and i == len(chat_history) - 1:
                        result_df = SessionState.get(SessionState.LAST_RESULT_DF)
                        if result_df is not None and len(result_df) > 0:
                            with st.expander(
                                f"View Results ({len(result_df):,} rows × {len(result_df.columns)} cols)",
                                expanded=True
                            ):
                                st.dataframe(
                                    result_df.head(settings.max_preview_rows),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                if len(result_df) > settings.max_preview_rows:
                                    st.caption(
                                        f"Showing first {settings.max_preview_rows} of {len(result_df):,} rows"
                                    )

                            # Auto-chart
                            if ResultFormatter.should_auto_chart(result_df):
                                fig = ResultFormatter.auto_chart(result_df)
                                if fig:
                                    with st.expander("Auto-generated Chart", expanded=True):
                                        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# QUERY INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.divider()

with st.form(key="query_form", clear_on_submit=True):
    col_input, col_btn = st.columns([8, 1])
    with col_input:
        user_input = st.text_input(
            "Ask a question about your data",
            placeholder="e.g. What are the top 10 products by sales?",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Send", type="primary", use_container_width=True)

if submitted and user_input.strip():
    api_key = SessionState.get(SessionState.API_KEY)

    if not api_key:
        st.error("Please enter your OpenRouter API key in the sidebar.")
        st.stop()

    agent: SQLAgent = SessionState.get(SessionState.AGENT)

    # Create agent if not exists (for greeting/help without file upload)
    if agent is None:
        agent = SQLAgent(
            api_key=api_key,
            model=SessionState.get(SessionState.MODEL),
            vector_store_dir=settings.chroma_persist_dir,
        )
        SessionState.set(SessionState.AGENT, agent)

    # Add user message
    SessionState.add_message("user", user_input)

    # Process query
    with st.spinner("Thinking..."):
        response: AgentResponse = agent.process_query(user_input)

    # Store last result DataFrame for display
    if response.result_df is not None:
        SessionState.set(SessionState.LAST_RESULT_DF, response.result_df)
    else:
        SessionState.set(SessionState.LAST_RESULT_DF, None)

    # Add assistant message with metadata
    SessionState.add_message(
        "assistant",
        response.answer,
        metadata={
            "success": response.success,
            "sql_query": response.sql_query,
            "was_repaired": response.was_repaired,
            "attempts": response.attempts,
            "rows_returned": len(response.result_df) if response.result_df is not None else 0,
            "has_result_df": response.result_df is not None and len(response.result_df) > 0,
            "intent": response.intent,
        }
    )

    st.rerun()
