from typing import List

from openai import OpenAI

from mcp_server_mariadb_vector.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI implementation of the embedding provider.

    Args:
        model: The name of the OpenAI model to use.
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def length_of_embedding(self) -> int:
        """Get the length of the embedding for a given model."""
        if self.model == "text-embedding-3-small":
            return 1536
        elif self.model == "text-embedding-3-large":
            return 3072
        else:
            raise ValueError(f"Unknown embedding model: {self.model}")

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        embeddings = [
            self.client.embeddings.create(
                model=self.model,
                input=document,
            )
            .data[0]
            .embedding
            for document in documents
        ]
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        embedding = self.client.embeddings.create(
            model=self.model,
            input=query,
        )
        return embedding.data[0].embedding
