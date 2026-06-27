"""
ChromaDB vector store wrapper for schema RAG.
Manages: adding schema embeddings, querying by natural language, resetting on new upload.
Uses sentence-transformers locally (no embedding API calls = no extra cost).
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
import logging

logger = logging.getLogger(__name__)


class SchemaVectorStore:
    COLLECTION_NAME = "table_schemas"

    def __init__(self, persist_dir: str, embedding_model_name: str):
        self.persist_dir = persist_dir
        self._embedder = None
        self._embedding_model_name = embedding_model_name
        self._client = None
        self._collection = None

    def _get_embedder(self):
        """Lazy-load the embedding model (downloads on first use)."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def _get_client(self):
        """Lazy-init ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        return self._client

    def _get_collection(self):
        """Get or create the schema collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "Table schema embeddings for RAG SQL"}
            )
        return self._collection

    def add_schema(self, table_name: str, schema_text: str) -> None:
        """Embed and store a table schema. Replaces existing entry for same table."""
        collection = self._get_collection()
        embedder = self._get_embedder()

        chunks = self._chunk_schema(table_name, schema_text)

        for i, chunk in enumerate(chunks):
            doc_id = f"{table_name}_chunk_{i}"
            embedding = embedder.encode(chunk, normalize_embeddings=True).tolist()

            try:
                collection.delete(ids=[doc_id])
            except Exception:
                pass

            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                ids=[doc_id],
                metadatas=[{"table_name": table_name, "chunk_index": i}]
            )

        logger.info(f"Stored {len(chunks)} schema chunks for table: {table_name}")

    def query(self, natural_language_query: str, top_k: int = 5) -> list[str]:
        """
        Retrieve the most relevant schema chunks for a given natural language query.
        Returns: list of schema text chunks, ranked by relevance.
        """
        collection = self._get_collection()
        embedder = self._get_embedder()

        if collection.count() == 0:
            return []

        query_embedding = embedder.encode(
            natural_language_query, normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        return results['documents'][0] if results['documents'] else []

    def get_all_schemas(self) -> list[str]:
        """Retrieve all stored schema documents."""
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        results = collection.get(include=["documents"])
        return results['documents']

    def clear_all(self) -> None:
        """Remove all stored schemas. Called when user uploads new files."""
        client = self._get_client()
        try:
            client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self._collection = None

    def count(self) -> int:
        """Return number of stored schema chunks."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def _chunk_schema(self, table_name: str, schema_text: str) -> list[str]:
        """Split schema into overlapping chunks for better retrieval granularity."""
        lines = schema_text.split('\n')
        header = f"TABLE: {table_name}\n"
        chunks = []
        column_lines = [l for l in lines if l.strip().startswith('-')]
        if column_lines:
            chunks.append(header + "COLUMNS:\n" + '\n'.join(column_lines))
        chunks.append(schema_text)
        for col_line in column_lines:
            if len(col_line.strip()) > 3:
                chunks.append(f"TABLE {table_name}, COLUMN: {col_line.strip()}")
        return [c for c in chunks if c.strip()]
