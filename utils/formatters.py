"""
Output formatting utilities for the Streamlit UI.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ResultFormatter:

    @staticmethod
    def should_auto_chart(df: pd.DataFrame) -> bool:
        """Determine if results are suitable for automatic chart generation."""
        if df is None or len(df) == 0:
            return False
        if len(df.columns) == 2:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            other_cols = [c for c in df.columns if c not in numeric_cols]
            return len(numeric_cols) == 1 and len(other_cols) == 1
        if len(df) <= 50 and df.select_dtypes(include=['number']).shape[1] >= 1:
            return True
        return False

    @staticmethod
    def auto_chart(df: pd.DataFrame) -> go.Figure | None:
        """Generate the most appropriate Plotly chart for the data."""
        if df is None or len(df) == 0:
            return None

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        other_cols = [c for c in df.columns if c not in numeric_cols]

        if not numeric_cols:
            return None

        try:
            if len(df.columns) == 2 and len(other_cols) == 1 and len(numeric_cols) == 1:
                x_col = other_cols[0]
                y_col = numeric_cols[0]
                if len(df) <= 20:
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                else:
                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                fig.update_layout(template='plotly_dark', height=350)
                return fig

            elif len(numeric_cols) >= 2 and len(df) <= 500:
                fig = px.scatter(
                    df, x=numeric_cols[0], y=numeric_cols[1],
                    title=f"{numeric_cols[1]} vs {numeric_cols[0]}",
                    template='plotly_dark', height=350
                )
                return fig

        except Exception:
            return None

        return None

    @staticmethod
    def format_row_count(n: int) -> str:
        if n == 0: return "No rows returned"
        if n == 1: return "1 row"
        return f"{n:,} rows"
