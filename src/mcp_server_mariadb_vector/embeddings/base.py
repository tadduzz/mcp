from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class EmbeddingProviderType(Enum):
    OPENAI = "openai"
    TEST = "test"
    # SENTENCE_TRANSFORMERS = "sentence-transformers"


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def length_of_embedding(self) -> int:
        """Get the length of the embedding for a given model."""
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        pass
