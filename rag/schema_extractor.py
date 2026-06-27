"""
Schema intelligence extractor.
Given a pandas DataFrame, produces:
  1. A machine-readable schema dict (for DuckDB registration)
  2. A rich natural language schema description (for vector embedding)
  3. Statistical profile (for LLM context)
"""
import pandas as pd
import numpy as np
from typing import Any


class SchemaExtractor:

    @classmethod
    def extract_full_schema(cls, table_name: str, df: pd.DataFrame) -> dict:
        """
        Returns a dict with:
          - 'schema_text': natural language description for embedding
          - 'llm_context': condensed schema for SQL generation prompts
          - 'column_info': list of column detail dicts
          - 'stats': DataFrame describe() as dict
          - 'sample_rows': first 5 rows as list of dicts
          - 'row_count': total rows
          - 'col_count': total columns
        """
        column_info = cls._extract_columns(df)
        stats = cls._compute_statistics(df)
        sample_rows = df.head(5).fillna('NULL').to_dict('records')
        row_count = len(df)

        schema_text = cls._build_schema_text(table_name, df, column_info, stats, sample_rows, row_count)
        llm_context = cls._build_llm_context(table_name, column_info, row_count)

        return {
            'table_name': table_name,
            'schema_text': schema_text,
            'llm_context': llm_context,
            'column_info': column_info,
            'stats': stats,
            'sample_rows': sample_rows,
            'row_count': row_count,
            'col_count': len(df.columns),
        }

    @classmethod
    def _extract_columns(cls, df: pd.DataFrame) -> list[dict]:
        """Detailed per-column intelligence."""
        columns = []
        for col in df.columns:
            series = df[col]
            dtype_str = cls._map_dtype(series.dtype)
            null_count = int(series.isna().sum())
            null_pct = round(null_count / len(df) * 100, 1) if len(df) > 0 else 0
            unique_count = int(series.nunique())
            unique_pct = round(unique_count / len(df) * 100, 1) if len(df) > 0 else 0

            info = {
                'name': col,
                'dtype': dtype_str,
                'pandas_dtype': str(series.dtype),
                'null_count': null_count,
                'null_pct': null_pct,
                'unique_count': unique_count,
                'unique_pct': unique_pct,
            }

            # Sample values (non-null)
            sample_vals = series.dropna().unique()[:5].tolist()
            info['sample_values'] = [str(v) for v in sample_vals]

            # Numeric stats
            if pd.api.types.is_numeric_dtype(series):
                info['min'] = cls._safe_val(series.min())
                info['max'] = cls._safe_val(series.max())
                info['mean'] = cls._safe_val(series.mean())

            # Date detection for object columns
            elif series.dtype == object and null_pct < 95:
                info['is_likely_date'] = cls._looks_like_date(series)

            # Categorical detection
            if unique_count <= 20 and unique_count > 0:
                info['categories'] = [str(v) for v in series.dropna().unique().tolist()]

            columns.append(info)
        return columns

    @classmethod
    def _compute_statistics(cls, df: pd.DataFrame) -> dict:
        """Numeric statistics for all numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {}
        try:
            return numeric_df.describe().round(4).to_dict()
        except Exception:
            return {}

    @classmethod
    def _build_schema_text(cls, table_name: str, df: pd.DataFrame,
                            column_info: list[dict], stats: dict,
                            sample_rows: list[dict], row_count: int) -> str:
        """Build the rich natural language schema description used for embedding."""
        lines = [
            f"TABLE: {table_name}",
            f"ROWS: {row_count:,} | COLUMNS: {len(column_info)}",
            "",
            "COLUMNS:",
        ]
        for col in column_info:
            col_line = f"  - {col['name']} ({col['dtype']})"
            if col.get('categories'):
                col_line += f" | Values: {', '.join(col['categories'][:10])}"
            elif col.get('sample_values'):
                col_line += f" | Examples: {', '.join(col['sample_values'][:5])}"
            if col.get('min') is not None:
                col_line += f" | Range: {col['min']} to {col['max']}"
            if col['null_pct'] > 5:
                col_line += f" | {col['null_pct']}% null"
            lines.append(col_line)

        lines.append("")
        lines.append("SAMPLE ROWS:")
        for i, row in enumerate(sample_rows[:3]):
            lines.append(f"  Row {i+1}: {dict(list(row.items())[:8])}")

        return '\n'.join(lines)

    @classmethod
    def _build_llm_context(cls, table_name: str, column_info: list[dict], row_count: int) -> str:
        """Compact schema for SQL generation system prompts."""
        lines = [f"Table: {table_name} ({row_count:,} rows)"]
        lines.append("Columns:")
        for col in column_info:
            line = f"  {col['name']} {col['dtype']}"
            if col.get('categories'):
                cats = col['categories'][:6]
                line += f" -- allowed values: {', '.join(repr(c) for c in cats)}"
            elif col.get('is_likely_date'):
                line += " -- date/time column"
            if col.get('min') is not None:
                line += f" -- min:{col['min']} max:{col['max']}"
            lines.append(line)
        return '\n'.join(lines)

    @classmethod
    def _map_dtype(cls, dtype) -> str:
        """Map pandas dtype to SQL-friendly type name."""
        dtype_str = str(dtype)
        if 'int' in dtype_str: return 'INTEGER'
        if 'float' in dtype_str: return 'FLOAT'
        if 'bool' in dtype_str: return 'BOOLEAN'
        if 'datetime' in dtype_str: return 'DATETIME'
        if 'date' in dtype_str: return 'DATE'
        return 'TEXT'

    @classmethod
    def _looks_like_date(cls, series: pd.Series) -> bool:
        """Heuristic check if a string column contains dates."""
        sample = series.dropna().head(10).astype(str)
        import re
        date_pattern = re.compile(
            r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}|'
            r'\d{1,2}/\d{1,2}/\d{2,4}'
        )
        matches = sum(1 for v in sample if date_pattern.search(v))
        return matches >= len(sample) * 0.7

    @classmethod
    def _safe_val(cls, val) -> Any:
        """Convert numpy types to Python native for JSON serialization."""
        if pd.isna(val): return None
        if hasattr(val, 'item'): return val.item()
        return val
