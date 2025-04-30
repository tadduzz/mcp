import json
from abc import ABC, abstractmethod
from enum import Enum
from collections.abc import AsyncIterator
from typing import List, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass

import mariadb
from openai import OpenAI
from mcp.server.fastmcp import FastMCP, Context
from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingProviderType(Enum):
    OPENAI = "openai"
    # SENTENCE_TRANSFORMERS = "sentence-transformers"


class DatabaseSettings(BaseSettings):
    host: str = Field(default="127.0.0.1", validation_alias="MARIADB_HOST")
    port: int = Field(default=3306, validation_alias="MARIADB_PORT")
    user: str = Field(validation_alias="MARIADB_USER")
    password: str = Field(validation_alias="MARIADB_PASSWORD")
    database: str = Field(default="mcp", validation_alias="MARIADB_DATABASE")


class EmbeddingSettings(BaseSettings):
    provider: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.OPENAI, validation_alias="EMBEDDING_PROVIDER"
    )
    model: str = Field(
        default="text-embedding-3-small", validation_alias="EMBEDDING_MODEL"
    )
    openai_api_key: Optional[str] = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        pass

    @abstractmethod
    def length_of_embedding(self, model: str) -> int:
        """Get the length of the embedding for a model."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI implementation of the embedding provider.
    Args:
        model_name: The name of the OpenAI model to use.
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = OpenAI(api_key=api_key)

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

    def length_of_embedding(self) -> int:
        """Get the length of the embedding for a model."""
        if self.model == "text-embedding-3-small":
            return 1536
        elif self.model == "text-embedding-3-large":
            return 3072
        else:
            raise ValueError(f"Unknown embedding model: {self.model}")


def create_embedding_provider(settings: EmbeddingSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    Args:
        settings: The settings for the embedding provider.
    Returns:
        An instance of the specified embedding provider.
    """
    if settings.provider == EmbeddingProviderType.OPENAI:
        return OpenAIEmbeddingProvider(settings.model, settings.openai_api_key)

    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider}")


@dataclass
class AppContext:
    conn: mariadb.Connection


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    conn = mariadb.connect(
        host=DatabaseSettings().host,
        port=DatabaseSettings().port,
        user=DatabaseSettings().user,
        password=DatabaseSettings().password,
        database=DatabaseSettings().database,
    )
    conn.autocommit = True
    try:
        yield AppContext(conn=conn)
    finally:
        conn.close()


mcp = FastMCP(
    "Mariadb Vector",
    lifespan=app_lifespan,
    dependencies=["mariadb", "openai", "pydantic", "pydantic-settings"],
)


embedding_provider = create_embedding_provider(EmbeddingSettings())


@mcp.tool()
def mariadb_create_vector_store(
    ctx: Context,
    vector_store_name: str = "vector_store",
    distance_function: str = "euclidean",
) -> str:
    """Create a vector store in MariaDB.

    Args:
        vector_store_name: The name of the vector store. Default is "vector_store".
        embedding_model: The name of the embedding model to use. Default is "openai/text-embedding-3-small".
        distance_function: The distance function to use. Options: 'euclidean', 'cosine'. Default is "euclidean".
    """

    embedding_length = embedding_provider.length_of_embedding()

    schema_query = f"""
    CREATE TABLE `{DatabaseSettings().database}`.`{vector_store_name}` (
        id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
        document LONGTEXT NOT NULL,
        embedding VECTOR({embedding_length}) NOT NULL,
        metadata JSON NOT NULL,
        VECTOR INDEX (embedding) DISTANCE={distance_function}
    );
    """

    try:
        conn = ctx.request_context.lifespan_context.conn
        with conn.cursor() as cursor:
            cursor.execute(schema_query)
    except mariadb.Error as e:
        return f"Error creating vector store `{vector_store_name}`: {e}"

    return f"Vector store `{vector_store_name}` created successfully."


@mcp.tool()
def mariadb_list_vector_stores(ctx: Context) -> str:
    """List all vector stores in a MariaDB database."""
    try:
        conn = ctx.request_context.lifespan_context.conn
        with conn.cursor() as cursor:
            cursor.execute(f"SHOW TABLES IN {DatabaseSettings().database}")
            tables = [table[0] for table in cursor]
    except mariadb.Error as e:
        return f"Error listing vector stores: {e}"

    return "Vector stores: " + ", ".join(tables)


@mcp.tool()
def mariadb_insert_documents(
    ctx: Context,
    vector_store_name: str = "vector_store",
    documents: List[str] = [],
    metadata: List[dict] = [],
) -> str:
    """Insert a document into a vector store.

    Args:
        vector_store_name: The name of the vector store.
        documents: The documents to insert.
        embedding_model: The embedding model to use.
        metadata: The metadata of the documents.
    """

    embeddings = embedding_provider.embed_documents(documents)

    metadata_json = [json.dumps(metadata) for metadata in metadata]

    insert_query = f"""
    INSERT INTO `{DatabaseSettings().database}`.`{vector_store_name}` (document, embedding, metadata) VALUES (%s, VEC_FromText(%s), %s);
    """
    try:
        conn = ctx.request_context.lifespan_context.conn
        with conn.cursor() as cursor:
            cursor.executemany(
                insert_query, list(zip(documents, embeddings, metadata_json))
            )
    except mariadb.Error as e:
        return f"Error inserting documents`{vector_store_name}`: {e}"

    return f"Documents inserted into `{vector_store_name}` successfully."


@mcp.tool()
def mariadb_vector_search(
    ctx: Context, query: str, vector_store_name: str = "vector_store", k: int = 5
) -> str:
    """Search a vector store for the most similar documents to a query.

    Args:
        query: The query to search for.
        vector_store_name: The name of the vector store to search.
        k: The number of results to return. Default is 5.
    """

    embedding = embedding_provider.embed_query(query)

    search_query = f"""
    SELECT 
        document,
        metadata,
        VEC_DISTANCE_EUCLIDEAN(embedding, VEC_FromText(%s)) AS distance
    FROM `{DatabaseSettings().database}`.`{vector_store_name}`
    ORDER BY distance ASC
    LIMIT %s;
    """

    try:
        conn = ctx.request_context.lifespan_context.conn
        with conn.cursor(buffered=True) as cursor:
            cursor.execute(
                search_query,
                (str(embedding), k),
            )
            rows = cursor.fetchall()
    except mariadb.Error as e:
        return f"Error searching vector store`{vector_store_name}`: {e}"

    if not rows:
        return "No similar context found."

    return "\n\n".join(
        f"Document: {row[0]}\nMetadata: {json.loads(row[1])}\nDistance: {row[2]}"
        for row in rows
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
