import argparse
import json
from typing import Annotated, List, Literal

import mariadb
from fastmcp import Context, FastMCP
from pydantic import Field

from mcp_server_mariadb_vector.app_context import app_lifespan
from mcp_server_mariadb_vector.embeddings.factory import create_embedding_provider
from mcp_server_mariadb_vector.settings import EmbeddingSettings

mcp = FastMCP(
    "Mariadb Vector",
    lifespan=app_lifespan,
    dependencies=["mariadb", "openai", "pydantic", "pydantic-settings"],
)


embedding_provider = create_embedding_provider(EmbeddingSettings())


@mcp.tool()
def mariadb_create_vector_store(
    ctx: Context,
    vector_store_name: Annotated[
        str,
        Field(description="The name of the vector store to create"),
    ],
    distance_function: Annotated[
        Literal["euclidean", "cosine"],
        Field(description="The distance function to use."),
    ] = "euclidean",
) -> str:
    """Create a vector store in the MariaDB database."""

    embedding_length = embedding_provider.length_of_embedding()

    schema_query = f"""
    CREATE TABLE `{vector_store_name}` (
        id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
        document LONGTEXT NOT NULL,
        embedding VECTOR({embedding_length}) NOT NULL,
        metadata JSON NOT NULL,
        VECTOR INDEX (embedding) DISTANCE={distance_function}
    )
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
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor]
    except mariadb.Error as e:
        return f"Error listing vector stores: {e}"

    return "Vector stores: " + ", ".join(tables)


@mcp.tool()
def mariadb_delete_vector_store(
    ctx: Context,
    vector_store_name: Annotated[
        str, Field(description="The name of the vector store to delete.")
    ],
) -> str:
    """Delete a vector store in the MariaDB database."""

    try:
        conn = ctx.request_context.lifespan_context.conn
        with conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE `{vector_store_name}`")
    except mariadb.Error as e:
        return f"Error deleting vector store `{vector_store_name}`: {e}"

    return f"Vector store `{vector_store_name}` deleted successfully."


@mcp.tool()
def mariadb_insert_documents(
    ctx: Context,
    vector_store_name: Annotated[
        str, Field(description="The name of the vector store to insert documents into.")
    ],
    documents: Annotated[
        List[str], Field(description="The documents to insert into the vector store.")
    ],
    metadata: Annotated[
        List[dict], Field(description="The metadata of the documents to insert.")
    ],
) -> str:
    """Insert a document into a vector store."""

    embeddings = embedding_provider.embed_documents(documents)

    metadata_json = [json.dumps(metadata) for metadata in metadata]

    insert_query = f"""
    INSERT INTO `{vector_store_name}` (document, embedding, metadata) VALUES (%s, VEC_FromText(%s), %s)
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
    ctx: Context,
    query: Annotated[str, Field(description="The query to search for.")],
    vector_store_name: Annotated[
        str, Field(description="The name of the vector store to search.")
    ],
    k: Annotated[int, Field(gt=0, description="The number of results to return.")] = 5,
) -> str:
    """Search a vector store for the most similar documents to a query."""

    embedding = embedding_provider.embed_query(query)

    search_query = f"""
    SELECT 
        document,
        metadata,
        VEC_DISTANCE_EUCLIDEAN(embedding, VEC_FromText(%s)) AS distance
    FROM `{vector_store_name}`
    ORDER BY distance ASC
    LIMIT %s
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    else:
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
