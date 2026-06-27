"""
SQL generation via OpenRouter LLM.
Produces: syntactically valid DuckDB SQL from natural language + schema context.
Uses low temperature (0.05) for deterministic SQL output.
"""
import re
import logging
from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_SQL = """You are an expert SQL engineer specializing in DuckDB SQL dialect.
Your ONLY job is to write correct, efficient SQL queries based on the user's question and schema.

STRICT RULES:
1. Output ONLY the SQL query – no explanation, no markdown, no backticks, no preamble
2. Use DuckDB SQL syntax (not MySQL, not PostgreSQL – DuckDB)
3. Table names and column names are exactly as given in the schema – case-sensitive if quoted
4. Always use table_name.column_name notation when multiple tables exist
5. For aggregations: use GROUP BY for all non-aggregate columns in SELECT
6. For date operations: use DuckDB's strftime(), date_trunc(), date_diff()
7. Limit results to 500 rows unless the user asks for more: append LIMIT 500 if no LIMIT exists
8. If a DISTINCT count is needed, use COUNT(DISTINCT column_name)
9. For string matching, use ILIKE for case-insensitive matching
10. NEVER use table aliases that shadow the table name – use descriptive aliases

DuckDB-specific functions to prefer:
- String: lower(), upper(), trim(), regexp_matches(), string_split()
- Date: strftime('%Y-%m-%d', date_col), date_trunc('month', date_col), year(), month(), day()
- Numeric: round(), ceil(), floor(), abs(), power()
- Aggregation: percentile_cont(0.5) WITHIN GROUP (ORDER BY col), mode()
- Window: ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...), LAG(), LEAD(), NTILE()

Output format: Plain SQL only. No explanations. No markdown code blocks. No semicolons at end."""


SYSTEM_PROMPT_REPAIR = """You are an expert SQL debugger for DuckDB.
You are given a broken SQL query and the exact error message from DuckDB.
Your job: fix the SQL to eliminate the error.

RULES:
1. Output ONLY the corrected SQL query – nothing else
2. Make the minimum change needed to fix the error
3. Preserve the original query's intent exactly
4. Do NOT add explanations, comments, or markdown
5. If the error indicates a column doesn't exist, check the schema carefully and use the correct name
6. If the error is a syntax error, fix the syntax per DuckDB dialect"""


class SQLGenerator:

    def __init__(self, api_key: str, model: str):
        self._client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=api_key,
        )
        self._model = model
        self._extra_headers = {
            "HTTP-Referer": settings.site_url,
            "X-Title": settings.site_name,
        }

    def generate_sql(self, user_query: str, schema_context: str) -> str:
        """
        Generate SQL from natural language.
        Returns: clean SQL string (no markdown, no extra text).
        """
        user_message = (
            f"Schema:\n{schema_context}\n\n"
            f"Question: {user_query}\n\n"
            "SQL:"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SQL},
                {"role": "user", "content": user_message}
            ],
            temperature=settings.sql_generation_temperature,
            max_tokens=settings.max_tokens_sql,
            extra_headers=self._extra_headers,
        )

        raw_sql = response.choices[0].message.content.strip()
        return self._clean_sql(raw_sql)

    def repair_sql(self, broken_sql: str, error_message: str, schema_context: str, original_query: str) -> str:
        """
        Repair a broken SQL query given the error message.
        Returns: corrected SQL string.
        """
        user_message = (
            f"Schema:\n{schema_context}\n\n"
            f"Original question: {original_query}\n\n"
            f"Broken SQL:\n{broken_sql}\n\n"
            f"DuckDB Error:\n{error_message}\n\n"
            "Fixed SQL:"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_REPAIR},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=settings.max_tokens_repair,
            extra_headers=self._extra_headers,
        )

        raw_sql = response.choices[0].message.content.strip()
        return self._clean_sql(raw_sql)

    @staticmethod
    def _clean_sql(raw: str) -> str:
        """Remove markdown formatting and extra whitespace from LLM SQL output."""
        raw = re.sub(r'```(?:sql|SQL)?\s*', '', raw)
        raw = re.sub(r'```', '', raw)
        raw = re.sub(r'^(?:SQL|Query|Answer):\s*', '', raw, flags=re.IGNORECASE)
        raw = raw.rstrip(';').strip()
        return raw
