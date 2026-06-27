"""
SQL syntax validator using sqlglot.
Pre-validates SQL before execution to catch obvious errors early.
"""
import sqlglot
import logging

logger = logging.getLogger(__name__)


class SQLValidator:

    @staticmethod
    def validate(sql: str) -> str | None:
        """
        Validate SQL syntax using sqlglot with DuckDB dialect.
        Returns: error message string if invalid, None if valid.
        """
        if not sql or not sql.strip():
            return "SQL query is empty"
        try:
            parsed = sqlglot.parse(sql, dialect="duckdb")
            if not parsed or parsed[0] is None:
                return "SQL could not be parsed"
            return None
        except sqlglot.errors.ParseError as e:
            return f"SQL syntax error: {str(e)}"
        except Exception as e:
            logger.debug(f"sqlglot validation exception (non-critical): {e}")
            return None

    @staticmethod
    def is_safe(sql: str) -> bool:
        """
        Check if SQL contains dangerous operations (DDL, DROP, DELETE without WHERE, etc.).
        Returns True if safe to execute, False if potentially destructive.
        """
        sql_upper = sql.upper().strip()
        dangerous_keywords = [
            'DROP TABLE', 'DROP DATABASE', 'TRUNCATE',
            'DELETE FROM', 'DROP VIEW', 'DROP SCHEMA',
            'CREATE TABLE', 'ALTER TABLE', 'INSERT INTO',
            'UPDATE ', 'EXEC ', 'EXECUTE ',
        ]
        for kw in dangerous_keywords:
            if kw in sql_upper:
                logger.warning(f"Blocked dangerous SQL keyword: {kw}")
                return False
        return True
