"""
Builds the schema context string injected into LLM prompts.
Merges retrieved RAG chunks with full schema for comprehensive SQL context.
"""


class ContextBuilder:

    @staticmethod
    def build_sql_context(
        retrieved_chunks: list[str],
        all_table_schemas: dict[str, dict],
        user_query: str
    ) -> str:
        """
        Build the complete schema context for SQL generation.

        retrieved_chunks: RAG-retrieved schema text chunks
        all_table_schemas: {table_name: schema_dict from SchemaExtractor}
        user_query: original user question (used for relevance scoring)
        """
        lines = ["=== DATABASE SCHEMA ===\n"]

        # Always include full llm_context for all registered tables
        for table_name, schema_dict in all_table_schemas.items():
            lines.append(schema_dict['llm_context'])
            lines.append("")

        # Add RAG-enriched details if available
        if retrieved_chunks:
            lines.append("\n=== RELEVANT SCHEMA DETAILS (from semantic search) ===\n")
            seen = set()
            for chunk in retrieved_chunks:
                if chunk not in seen:
                    lines.append(chunk)
                    lines.append("")
                    seen.add(chunk)

        return '\n'.join(lines)

    @staticmethod
    def build_sample_data_context(all_table_schemas: dict[str, dict], max_rows: int = 5) -> str:
        """Build a sample data section for LLM context."""
        lines = ["=== SAMPLE DATA ===\n"]
        for table_name, schema_dict in all_table_schemas.items():
            lines.append(f"Table: {table_name}")
            for row in schema_dict.get('sample_rows', [])[:max_rows]:
                truncated = dict(list(row.items())[:10])
                lines.append(f"  {truncated}")
            lines.append("")
        return '\n'.join(lines)
