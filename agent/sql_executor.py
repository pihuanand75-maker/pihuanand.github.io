"""
DuckDB SQL execution engine with self-correcting retry loop.
Manages: database connection, table registration, execution, error recovery.
"""
import duckdb
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    data: Optional[pd.DataFrame]
    sql_used: str
    error_message: Optional[str]
    attempts: int
    rows_returned: int = 0
    was_repaired: bool = False


class SQLExecutor:

    def __init__(self):
        # In-memory DuckDB connection – fastest, no file I/O
        self._conn = duckdb.connect(database=':memory:', read_only=False)
        self._registered_tables: set[str] = set()

    def register_table(self, table_name: str, df: pd.DataFrame) -> None:
        """
        Register a pandas DataFrame as a DuckDB table.
        Safe to call multiple times – replaces existing table.
        """
        if table_name in self._registered_tables:
            try:
                self._conn.execute(f'DROP VIEW IF EXISTS "{table_name}"')
            except Exception:
                pass

        self._conn.register(table_name, df)
        self._registered_tables.add(table_name)
        logger.info(f"Registered table: {table_name} ({len(df)} rows, {len(df.columns)} cols)")

    def get_registered_tables(self) -> list[str]:
        """Return list of currently registered table names."""
        return list(self._registered_tables)

    def execute_with_retry(
        self,
        initial_sql: str,
        sql_generator,
        schema_context: str,
        original_query: str,
        max_retries: int = 3
    ) -> ExecutionResult:
        """
        Execute SQL with automatic self-correction on failure.

        Flow:
          1. Validate SQL syntax (sqlglot) before execution
          2. Execute against DuckDB
          3. On error: ask LLM to repair → retry
          4. After max_retries failures: return error result
        """
        from utils.sql_validator import SQLValidator

        sql = initial_sql
        last_error = None
        was_repaired = False

        for attempt in range(1, max_retries + 1):
            # Step 1: Pre-validate SQL syntax
            validation_error = SQLValidator.validate(sql)
            if validation_error and attempt == 1:
                logger.warning(f"SQL syntax warning: {validation_error}")

            # Step 2: Safety check
            if not SQLValidator.is_safe(sql):
                return ExecutionResult(
                    success=False,
                    data=None,
                    sql_used=sql,
                    error_message="Query blocked: contains potentially destructive SQL operations.",
                    attempts=attempt,
                    rows_returned=0,
                    was_repaired=was_repaired,
                )

            # Step 3: Execute
            try:
                result_df = self._conn.execute(sql).df()
                return ExecutionResult(
                    success=True,
                    data=result_df,
                    sql_used=sql,
                    error_message=None,
                    attempts=attempt,
                    rows_returned=len(result_df),
                    was_repaired=was_repaired,
                )

            except duckdb.Error as e:
                last_error = str(e)
                logger.warning(f"SQL execution error (attempt {attempt}/{max_retries}): {last_error}")

                if attempt < max_retries:
                    try:
                        sql = sql_generator.repair_sql(
                            broken_sql=sql,
                            error_message=last_error,
                            schema_context=schema_context,
                            original_query=original_query,
                        )
                        was_repaired = True
                        logger.info(f"Repaired SQL (attempt {attempt+1}):\n{sql}")
                    except Exception as repair_err:
                        logger.error(f"SQL repair LLM call failed: {repair_err}")
                        break

            except Exception as e:
                last_error = str(e)
                logger.error(f"Non-SQL error during execution: {last_error}")
                break

        return ExecutionResult(
            success=False,
            data=None,
            sql_used=sql,
            error_message=last_error,
            attempts=max_retries,
            rows_returned=0,
            was_repaired=was_repaired,
        )

    def describe_tables(self) -> str:
        """Run DuckDB SHOW TABLES and return result as string."""
        try:
            result = self._conn.execute("SHOW TABLES").df()
            return result.to_string(index=False)
        except Exception:
            return str(list(self._registered_tables))

    def reset(self) -> None:
        """Drop all registered tables and reset state."""
        for table in list(self._registered_tables):
            try:
                self._conn.execute(f'DROP VIEW IF EXISTS "{table}"')
            except Exception:
                pass
        self._registered_tables.clear()
        logger.info("DuckDB executor reset: all tables dropped")
