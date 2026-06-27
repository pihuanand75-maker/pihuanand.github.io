"""
LLM-based result interpreter.
Takes a SQL query result (pandas DataFrame) and generates:
  - A natural language summary of the findings
  - Key insights and patterns
  - Suggested follow-up questions
"""
import pandas as pd
import logging
from openai import OpenAI
from config.settings import settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_INTERPRET = """You are a senior data analyst communicating findings to a business stakeholder.
You receive a SQL query, its result data, and the original user question.

Your response must follow this EXACT structure (use these exact headers):

## Summary
[1-3 sentences: the direct answer to the user's question based on the data]

## Key Findings
[3-5 bullet points: specific, quantified insights from the data]

## Notable Patterns
[1-3 observations about trends, outliers, or relationships in the data – skip if not applicable]

## Suggested Follow-ups
[2-3 natural language questions the user could ask next, formatted as a numbered list]

RULES:
- Be specific: reference actual numbers, column names, and values from the results
- Do not guess or infer beyond what the data shows
- If the result is empty (0 rows), explain clearly why and what it might mean
- If the result has >20 rows, summarize the distribution rather than listing all values
- Use plain language – no SQL jargon in the natural language response
- Format numbers with comma separators (1,234,567 not 1234567)"""


class ResultInterpreter:

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

    def interpret(
        self,
        user_query: str,
        sql_used: str,
        result_df: pd.DataFrame,
    ) -> str:
        """
        Generate natural language interpretation of SQL results.
        Returns: Formatted markdown string with analysis.
        """
        data_summary = self._build_data_summary(result_df)

        user_message = (
            f"User's question: {user_query}\n\n"
            f"SQL query executed:\n{sql_used}\n\n"
            f"Query results:\n{data_summary}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_INTERPRET},
                    {"role": "user", "content": user_message}
                ],
                temperature=settings.interpretation_temperature,
                max_tokens=settings.max_tokens_interpretation,
                extra_headers=self._extra_headers,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Result interpretation failed: {e}")
            return f"Results returned {len(result_df):,} rows. Analysis unavailable: {str(e)}"

    def generate_error_response(self, user_query: str, error_msg: str, sql_attempted: str) -> str:
        """Generate a helpful response when SQL execution fails after all retries."""
        user_message = (
            f"User asked: {user_query}\n\n"
            f"SQL attempted:\n{sql_attempted}\n\n"
            f"Error after 3 retry attempts: {error_msg}\n\n"
            "Explain in plain language why this query couldn't be executed and suggest how the user can rephrase their question."
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.3,
                max_tokens=400,
                extra_headers=self._extra_headers,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return (
                f"Could not execute this query after 3 attempts.\n\n"
                f"**Error:** {error_msg}\n\n"
                "Try rephrasing your question or checking that the column names are correct."
            )

    @staticmethod
    def _build_data_summary(df: pd.DataFrame) -> str:
        """Convert DataFrame to LLM-digestible text summary."""
        if df is None or len(df) == 0:
            return "RESULT: Empty (0 rows returned)"

        rows = len(df)
        cols = len(df.columns)

        if rows <= 20:
            return f"ROWS: {rows} | COLUMNS: {cols}\n\n{df.to_string(index=False)}"
        else:
            lines = [f"ROWS: {rows:,} | COLUMNS: {cols}"]
            lines.append(f"\nFIRST 20 ROWS:\n{df.head(20).to_string(index=False)}")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                lines.append(f"\nNUMERIC COLUMN STATS:\n{df[numeric_cols].describe().round(2).to_string()}")
            return '\n'.join(lines)
