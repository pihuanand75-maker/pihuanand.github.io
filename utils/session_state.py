"""
Streamlit session state manager.
Centralizes all st.session_state keys to prevent typos and provide defaults.
"""
import streamlit as st
from typing import Any


class SessionState:
    # Keys
    AGENT = "sql_agent"
    CHAT_HISTORY = "chat_history"
    LOADED_TABLES = "loaded_tables"
    API_KEY = "api_key"
    MODEL = "selected_model"
    FILES_LOADED = "files_loaded"
    LAST_RESULT_DF = "last_result_df"

    @classmethod
    def init(cls) -> None:
        """Initialize all session state keys with defaults."""
        defaults = {
            cls.AGENT: None,
            cls.CHAT_HISTORY: [],
            cls.LOADED_TABLES: [],
            cls.API_KEY: "",
            cls.MODEL: "openai/gpt-4o-mini",
            cls.FILES_LOADED: False,
            cls.LAST_RESULT_DF: None,
        }
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        st.session_state[key] = value

    @classmethod
    def add_message(cls, role: str, content: str, metadata: dict = None) -> None:
        """Add a message to chat history."""
        msg = {"role": role, "content": content}
        if metadata:
            msg["metadata"] = metadata
        history = st.session_state.get(cls.CHAT_HISTORY, [])
        history.append(msg)
        # Cap at 50 messages to prevent memory issues
        if len(history) > 50:
            history = history[-50:]
        st.session_state[cls.CHAT_HISTORY] = history

    @classmethod
    def clear_chat(cls) -> None:
        st.session_state[cls.CHAT_HISTORY] = []
