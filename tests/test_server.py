import mariadb
import pytest
from fastmcp import Client, FastMCP

import mcp_server_mariadb_vector.server as server
from mcp_server_mariadb_vector.app_context import app_lifespan
from mcp_server_mariadb_vector.settings import DatabaseSettings


class MockEmbeddingProvider:
    def __init__(self):
        self.embed_documents_called = 0
        self.embed_query_called = 0

    def length_of_embedding(self):
        return 3

    def embed_documents(self, documents):
        self.embed_documents_called += 1
        print(f"Mock embed_documents called with {len(documents)} documents")
        return [[1.0, 0.0, 0.0] for _ in documents]

    def embed_query(self, query):
        self.embed_query_called += 1
        print(f"Mock embed_query called with query: '{query}'")
        return [1.0, 0.0, 0.0]


@pytest.fixture
def db_settings():
    return DatabaseSettings()


@pytest.fixture
def db_connection(db_settings):
    conn = mariadb.connect(
        host=db_settings.host,
        port=db_settings.port,
        user=db_settings.user,
        password=db_settings.password,
        database=db_settings.database,
    )
    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture
def clean_database(db_connection):
    """Clean up any test tables before and after tests."""
    with db_connection.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor]
        for table in tables:
            if table.startswith("test_"):
                cursor.execute(f"DROP TABLE `{table}`")
    yield
    with db_connection.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor]
        for table in tables:
            if table.startswith("test_"):
                cursor.execute(f"DROP TABLE `{table}`")


@pytest.fixture
def mock_embedding_provider():
    return MockEmbeddingProvider()


@pytest.fixture
async def mcp_test_server(mock_embedding_provider):
    """Set up MCP server with mocked embedding provider and regular app lifespan."""

    server.embedding_provider = mock_embedding_provider

    test_server = FastMCP("TestServer", lifespan=app_lifespan)

    # FastMCP specific: import server with "test" prefix so all tools appear as test_*
    await test_server.import_server("test", server.mcp)
    print("Server imported successfully")
    return test_server


async def test_create_and_delete_vector_store(
    mcp_test_server, clean_database, db_connection
):
    # FastMCP specific: use Client to call tools in-memory
    async with Client(mcp_test_server) as client:
        result = await client.call_tool(
            "test_mariadb_create_vector_store",
            {"vector_store_name": "test_store", "distance_function": "euclidean"},
        )
        assert "Vector store `test_store` created successfully" in str(result[0])

        with db_connection.cursor() as cursor:
            cursor.execute("SHOW TABLES LIKE 'test_store'")
            assert cursor.fetchone() is not None

        result = await client.call_tool(
            "test_mariadb_delete_vector_store", {"vector_store_name": "test_store"}
        )
        assert "Vector store `test_store` deleted successfully" in str(result[0])

        with db_connection.cursor() as cursor:
            cursor.execute("SHOW TABLES LIKE 'test_store'")
            assert cursor.fetchone() is None


async def test_list_vector_stores(mcp_test_server, clean_database, db_connection):
    async with Client(mcp_test_server) as client:
        await client.call_tool(
            "test_mariadb_create_vector_store",
            {"vector_store_name": "test_store1", "distance_function": "euclidean"},
        )
        await client.call_tool(
            "test_mariadb_create_vector_store",
            {"vector_store_name": "test_store2", "distance_function": "euclidean"},
        )

        result = await client.call_tool("test_mariadb_list_vector_stores", {})
        assert "test_store1" in str(result[0])
        assert "test_store2" in str(result[0])


async def test_insert_and_search_documents(
    mcp_test_server, clean_database, db_connection, mock_embedding_provider
):
    documents = ["This is a test document", "This is another test document"]
    metadata = [{"type": "test1"}, {"type": "test2"}]

    async with Client(mcp_test_server) as client:
        await client.call_tool(
            "test_mariadb_create_vector_store",
            {"vector_store_name": "test_store", "distance_function": "euclidean"},
        )

        result = await client.call_tool(
            "test_mariadb_insert_documents",
            {
                "vector_store_name": "test_store",
                "documents": documents,
                "metadata": metadata,
            },
        )
        assert "Documents inserted into `test_store` successfully" in str(result[0])
        assert mock_embedding_provider.embed_documents_called > 0, (
            "Mock embed_documents was not called"
        )

        result = await client.call_tool(
            "test_mariadb_vector_search",
            {"query": "test document", "vector_store_name": "test_store", "k": 2},
        )

        assert "This is a test document" in str(result[0])
        assert "This is another test document" in str(result[0])
        assert "'type': 'test1'" in str(result[0])
        assert "'type': 'test2'" in str(result[0])
        assert mock_embedding_provider.embed_query_called > 0, (
            "Mock embed_query was not called"
        )
