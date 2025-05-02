from mcp_server_mariadb_vector.embeddings.base import (
    EmbeddingProvider,
    EmbeddingProviderType,
)
from mcp_server_mariadb_vector.embeddings.openai import OpenAIEmbeddingProvider
from mcp_server_mariadb_vector.settings import EmbeddingSettings


def create_embedding_provider(settings: EmbeddingSettings) -> EmbeddingProvider:
    """
    Create an instance of the specified embedding provider.

    Args:
        settings: The settings for the embedding provider.
    """
    if settings.provider == EmbeddingProviderType.OPENAI:
        return OpenAIEmbeddingProvider(settings.model, settings.openai_api_key)

    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider}")
