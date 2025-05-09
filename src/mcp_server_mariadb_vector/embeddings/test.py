from typing import List

from mcp_server_mariadb_vector.embeddings.base import EmbeddingProvider


class TestEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider for testing.
    """

    def length_of_embedding(self) -> int:
        return 3

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3]] * len(documents)

    def embed_query(self, query: str) -> List[float]:
        return [0.1, 0.2, 0.3]
