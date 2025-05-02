from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from mcp_server_mariadb_vector.embeddings.base import EmbeddingProviderType


class DatabaseSettings(BaseSettings):
    host: str = Field(default="127.0.0.1", alias="MARIADB_HOST")
    port: int = Field(default=3306, alias="MARIADB_PORT")
    user: str = Field(..., alias="MARIADB_USER")
    password: str = Field(..., alias="MARIADB_PASSWORD")
    database: str = Field(default="mcp", alias="MARIADB_DATABASE")


class EmbeddingSettings(BaseSettings):
    provider: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.OPENAI, alias="EMBEDDING_PROVIDER"
    )
    model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
