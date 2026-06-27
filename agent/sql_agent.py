"""
Core SQLAgent orchestrator.
This is the main entry point called by the Streamlit UI.
Orchestrates: intent classification → RAG retrieval → SQL generation → execution → interpretation.
"""
import logging
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from agent.sql_generator import SQLGenerator
from agent.sql_executor import SQLExecutor, ExecutionResult
from agent.result_interpreter import ResultInterpreter
from agent.intent_classifier import IntentClassifier
from rag.schema_extractor import SchemaExtractor
from rag.vector_store import SchemaVectorStore
from rag.context_builder import ContextBuilder
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Complete response from the SQL Agent for one user query."""
    success: bool
    intent: str
    answer: str                          # Natural language response (always present)
    sql_query: Optional[str]             # Generated SQL (None for non-data queries)
    result_df: Optional[pd.DataFrame]    # Query results DataFrame (None on error)
    error_message: Optional[str]         # Error details (None on success)
    attempts: int                        # How many SQL execution attempts were made
    was_repaired: bool                   # Whether SQL was auto-corrected


class SQLAgent:
    """
    The main RAG SQL Agent class.
    Instantiate once per session with API key and model.
    Call process_query() for each user question.
    """

    def __init__(self, api_key: str, model: str, vector_store_dir: str):
        self._api_key = api_key
        self._model = model
        self._generator = SQLGenerator(api_key=api_key, model=model)
        self._executor = SQLExecutor()
        self._interpreter = ResultInterpreter(api_key=api_key, model=model)
        self._vector_store = SchemaVectorStore(
            persist_dir=vector_store_dir,
            embedding_model_name=settings.embedding_model
        )
        self._table_schemas: dict[str, dict] = {}
        self._tables: dict[str, pd.DataFrame] = {}

    def load_data(self, table_name: str, df: pd.DataFrame) -> dict:
        """
        Load a DataFrame into the agent:
        1. Extract schema intelligence
        2. Store schema in vector store
        3. Register table in DuckDB
        Returns: schema dict for display in UI
        """
        schema_dict = SchemaExtractor.extract_full_schema(table_name, df)
        self._table_schemas[table_name] = schema_dict
        self._tables[table_name] = df

        self._vector_store.add_schema(table_name, schema_dict['schema_text'])
        self._executor.register_table(table_name, df)

        logger.info(f"Loaded table: {table_name} ({schema_dict['row_count']} rows)")
        return schema_dict

    def reset(self) -> None:
        """Clear all loaded data. Called before loading new files."""
        self._executor.reset()
        self._vector_store.clear_all()
        self._table_schemas.clear()
        self._tables.clear()
        logger.info("Agent reset: all data cleared")

    def get_loaded_tables(self) -> list[str]:
        """Return names of currently loaded tables."""
        return list(self._table_schemas.keys())

    def get_schema_dict(self, table_name: str) -> Optional[dict]:
        """Return schema dict for a specific table."""
        return self._table_schemas.get(table_name)

    def process_query(self, user_query: str) -> AgentResponse:
        """
        Main entry point: process a natural language query.

        Pipeline:
          1. Classify intent
          2. If DATA_QUERY: RAG retrieve → generate SQL → execute → interpret
          3. Otherwise: return direct response
        """
        if not user_query.strip():
            return AgentResponse(
                success=False, intent='EMPTY', answer="Please enter a question.",
                sql_query=None, result_df=None, error_message="Empty query",
                attempts=0, was_repaired=False
            )

        intent = IntentClassifier.classify(user_query)

        if intent == 'GREETING':
            return AgentResponse(
                success=True, intent=intent,
                answer=IntentClassifier.greeting_response(),
                sql_query=None, result_df=None, error_message=None,
                attempts=0, was_repaired=False
            )

        if intent == 'HELP':
            return AgentResponse(
                success=True, intent=intent,
                answer=IntentClassifier.help_response(),
                sql_query=None, result_df=None, error_message=None,
                attempts=0, was_repaired=False
            )

        if intent == 'META_QUESTION':
            return self._handle_meta_question(user_query)

        if not self._table_schemas:
            return AgentResponse(
                success=False, intent=intent,
                answer="No data loaded yet. Please upload a file using the sidebar first.",
                sql_query=None, result_df=None, error_message="No data loaded",
                attempts=0, was_repaired=False
            )

        return self._process_data_query(user_query)

    def _process_data_query(self, user_query: str) -> AgentResponse:
        """Execute the full RAG → SQL → Execute → Interpret pipeline."""
        try:
            retrieved_chunks = self._vector_store.query(
                user_query, top_k=settings.schema_top_k
            )

            schema_context = ContextBuilder.build_sql_context(
                retrieved_chunks=retrieved_chunks,
                all_table_schemas=self._table_schemas,
                user_query=user_query,
            )

            initial_sql = self._generator.generate_sql(
                user_query=user_query,
                schema_context=schema_context,
            )
            logger.info(f"Generated SQL:\n{initial_sql}")

            exec_result: ExecutionResult = self._executor.execute_with_retry(
                initial_sql=initial_sql,
                sql_generator=self._generator,
                schema_context=schema_context,
                original_query=user_query,
                max_retries=settings.max_sql_retries,
            )

            if not exec_result.success:
                error_answer = self._interpreter.generate_error_response(
                    user_query=user_query,
                    error_msg=exec_result.error_message,
                    sql_attempted=exec_result.sql_used,
                )
                return AgentResponse(
                    success=False,
                    intent='DATA_QUERY',
                    answer=error_answer,
                    sql_query=exec_result.sql_used,
                    result_df=None,
                    error_message=exec_result.error_message,
                    attempts=exec_result.attempts,
                    was_repaired=exec_result.was_repaired,
                )

            interpretation = self._interpreter.interpret(
                user_query=user_query,
                sql_used=exec_result.sql_used,
                result_df=exec_result.data,
            )

            return AgentResponse(
                success=True,
                intent='DATA_QUERY',
                answer=interpretation,
                sql_query=exec_result.sql_used,
                result_df=exec_result.data,
                error_message=None,
                attempts=exec_result.attempts,
                was_repaired=exec_result.was_repaired,
            )

        except Exception as e:
            logger.exception(f"Unexpected error in data query pipeline: {e}")
            return AgentResponse(
                success=False,
                intent='DATA_QUERY',
                answer=f"An unexpected error occurred: {str(e)}. Please try rephrasing your question.",
                sql_query=None,
                result_df=None,
                error_message=str(e),
                attempts=0,
                was_repaired=False,
            )

    def _handle_meta_question(self, user_query: str) -> AgentResponse:
        """Handle questions about schema/structure without SQL execution."""
        if not self._table_schemas:
            answer = "No data is loaded. Please upload a file first."
        else:
            lines = ["**Loaded Tables:**\n"]
            for table_name, schema_dict in self._table_schemas.items():
                lines.append(f"**{table_name}** – {schema_dict['row_count']:,} rows, {schema_dict['col_count']} columns")
                for col in schema_dict['column_info']:
                    col_detail = f"  - `{col['name']}` ({col['dtype']})"
                    if col.get('categories'):
                        col_detail += f" – values: {', '.join(str(c) for c in col['categories'][:6])}"
                    lines.append(col_detail)
                lines.append("")
            answer = '\n'.join(lines)

        return AgentResponse(
            success=True, intent='META_QUESTION', answer=answer,
            sql_query=None, result_df=None, error_message=None,
            attempts=0, was_repaired=False
        )
