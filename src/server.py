import asyncio
import logging
import argparse
from typing import List, Dict, Any, Optional
from functools import partial # Needed for anyio.run

import asyncmy
import anyio
from fastmcp import FastMCP, Context

# Import configuration settings
from config import (
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME,
    MCP_READ_ONLY, MCP_MAX_POOL_SIZE, logger
)

from asyncmy.errors import Error as AsyncMyError

# --- MariaDB MCP Server Class ---
class MariaDBServer:
    """
    MCP Server exposing tools to interact with a MariaDB database.
    Manages the database connection pool.
    """
    def __init__(self, server_name="MariaDB_Server"):
        self.mcp = FastMCP(server_name)
        self.pool: Optional[asyncmy.Pool] = None
        self.is_read_only = MCP_READ_ONLY
        logger.info(f"Initializing {server_name}...")
        if self.is_read_only:
            logger.warning("Server running in READ-ONLY mode. Write operations are disabled.")

    async def initialize_pool(self):
        """Initializes the asyncmy connection pool within the running event loop."""
        if not all([DB_USER, DB_PASSWORD, DB_NAME]):
             logger.error("Cannot initialize pool due to missing database credentials.")
             raise ConnectionError("Missing database credentials for pool initialization.")

        if self.pool is not None:
            logger.info("Connection pool already initialized.")
            return

        try:
            logger.info(f"Creating connection pool for {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME} (max size: {MCP_MAX_POOL_SIZE})")
            self.pool = await asyncmy.create_pool(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                db=DB_NAME,
                minsize=1,
                maxsize=MCP_MAX_POOL_SIZE,
                autocommit=True,
                pool_recycle=3600
            )
            logger.info("Connection pool initialized successfully.")
        except AsyncMyError as e:
            logger.error(f"Failed to initialize database connection pool: {e}", exc_info=True)
            self.pool = None
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during pool initialization: {e}", exc_info=True)
            self.pool = None
            raise

    async def close_pool(self):
        """Closes the connection pool gracefully."""
        if self.pool:
            logger.info("Closing database connection pool...")
            try:
                self.pool.close()
                await self.pool.wait_closed()
                logger.info("Database connection pool closed.")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}", exc_info=True)
            finally:
                self.pool = None

    async def _execute_query(self, sql: str, params: Optional[tuple] = None, database: Optional[str] = None) -> List[Dict[str, Any]]:
        """Helper function to execute SELECT queries using the pool."""
        if self.pool is None:
            logger.error("Connection pool is not initialized.")
            raise RuntimeError("Database connection pool not available.")

        allowed_prefixes = ('SELECT', 'SHOW', 'DESC', 'DESCRIBE', 'USE')
        query_upper = sql.strip().upper()
        is_allowed_read_query = any(query_upper.startswith(prefix) for prefix in allowed_prefixes)

        if self.is_read_only and not is_allowed_read_query:
             logger.warning(f"Blocked potentially non-read-only query in read-only mode: {sql[:100]}...")
             raise PermissionError("Operation forbidden: Server is in read-only mode.")

        logger.info(f"Executing query (DB: {database or DB_NAME}): {sql[:100]}...")
        if params:
            logger.debug(f"Parameters: {params}")

        conn = None
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(cursor=asyncmy.cursors.DictCursor) as cursor:
                    current_db_query = "SELECT DATABASE()"
                    await cursor.execute(current_db_query)
                    current_db_result = await cursor.fetchone()
                    current_db_name = current_db_result.get('DATABASE()') if current_db_result else None
                    pool_db_name = DB_NAME
                    actual_current_db = current_db_name or pool_db_name

                    if database and database != actual_current_db:
                        logger.info(f"Switching database context from '{actual_current_db}' to '{database}'")
                        await cursor.execute(f"USE `{database}`")

                    await cursor.execute(sql, params or ())
                    results = await cursor.fetchall()
                    logger.info(f"Query executed successfully, {len(results)} rows returned.")
                    return results if results else []
        except AsyncMyError as e:
            conn_state = f"Connection: {'acquired' if conn else 'not acquired'}"
            logger.error(f"Database error executing query ({conn_state}): {e}", exc_info=True)
            raise RuntimeError(f"Database error: {e}") from e
        except PermissionError as e:
             logger.warning(f"Permission denied: {e}")
             raise e
        except Exception as e:
            if isinstance(e, RuntimeError) and 'Event loop is closed' in str(e):
                 logger.critical("Detected closed event loop during query execution!", exc_info=True)
                 raise RuntimeError("Event loop closed unexpectedly during query.") from e
            conn_state = f"Connection: {'acquired' if conn else 'not acquired'}"
            logger.error(f"Unexpected error during query execution ({conn_state}): {e}", exc_info=True)
            raise RuntimeError(f"An unexpected error occurred: {e}") from e
            
    async def _database_exists(self, database_name: str) -> bool:
        """Checks if a database exists."""
        if not database_name or not database_name.isidentifier():
            logger.warning(f"_database_exists called with invalid database_name: {database_name}")
            return False 

        sql = "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = %s"
        try:
            results = await self._execute_query(sql, params=(database_name,), database='information_schema')
            return len(results) > 0
        except Exception as e:
            logger.error(f"Error checking if database '{database_name}' exists: {e}", exc_info=True)
            return False
        
    async def _table_exists(self, database_name: str, table_name: str) -> bool:
        """Checks if a table exists in the given database."""
        if not database_name or not database_name.isidentifier() or \
           not table_name or not table_name.isidentifier():
            logger.warning(f"_table_exists called with invalid names: db='{database_name}', table='{table_name}'")
            return False

        sql = "SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s"
        try:
            results = await self._execute_query(sql, params=(database_name, table_name), database='information_schema')
            return len(results) > 0
        except Exception as e:
            logger.error(f"Error checking if table '{database_name}.{table_name}' exists: {e}", exc_info=True)
            return False
    
    
    # --- MCP Tool Definitions ---

    async def list_databases(self) -> List[str]:
        """Lists all accessible databases on the connected MariaDB server."""
        logger.info("TOOL START: list_databases called.")
        sql = "SHOW DATABASES"
        try:
            results = await self._execute_query(sql)
            db_list = [row['Database'] for row in results if 'Database' in row]
            logger.info(f"TOOL END: list_databases completed. Databases found: {len(db_list)}.")
            return db_list
        except Exception as e:
            logger.error(f"TOOL ERROR: list_databases failed: {e}", exc_info=True)
            raise

    async def list_tables(self, database_name: str) -> List[str]:
        """Lists all tables within the specified database."""
        logger.info(f"TOOL START: list_tables called. database_name={database_name}")
        if not database_name or not database_name.isidentifier():
            logger.warning(f"TOOL WARNING: list_tables called with invalid database_name: {database_name}")
            raise ValueError(f"Invalid database name provided: {database_name}")
        sql = "SHOW TABLES"
        try:
            results = await self._execute_query(sql, database=database_name)
            table_list = [list(row.values())[0] for row in results if row]
            logger.info(f"TOOL END: list_tables completed. Tables found: {len(table_list)}.")
            return table_list
        except Exception as e:
            logger.error(f"TOOL ERROR: list_tables failed for database_name={database_name}: {e}", exc_info=True)
            raise

    async def get_table_schema(self, database_name: str, table_name: str) -> Dict[str, Any]:
        """
        Retrieves the schema (column names, types, nullability, keys, default values)
        for a specific table in a database.
        """
        logger.info(f"TOOL START: get_table_schema called. database_name={database_name}, table_name={table_name}")
        if not database_name or not database_name.isidentifier():
            logger.warning(f"TOOL WARNING: get_table_schema called with invalid database_name: {database_name}")
            raise ValueError(f"Invalid database name provided: {database_name}")
        if not table_name or not table_name.isidentifier():
            logger.warning(f"TOOL WARNING: get_table_schema called with invalid table_name: {table_name}")
            raise ValueError(f"Invalid table name provided: {table_name}")

        sql = f"DESCRIBE `{database_name}`.`{table_name}`"
        try:
            schema_results = await self._execute_query(sql)
            schema_info = {}
            if not schema_results:
                exists_sql = "SELECT COUNT(*) as count FROM information_schema.tables WHERE table_schema = %s AND table_name = %s"
                exists_result = await self._execute_query(exists_sql, params=(database_name, table_name))
                if not exists_result or exists_result[0]['count'] == 0:
                    logger.warning(f"TOOL WARNING: Table '{database_name}'.'{table_name}' not found or inaccessible.")
                    raise FileNotFoundError(f"Table '{database_name}'.'{table_name}' not found or inaccessible.")
                else:
                    logger.warning(f"Could not describe table '{database_name}'.'{table_name}'. It might be a view or lack permissions.")

            for row in schema_results:
                col_name = row.get('Field')
                if col_name:
                    schema_info[col_name] = {
                        'type': row.get('Type'),
                        'nullable': row.get('Null', '').upper() == 'YES',
                        'key': row.get('Key'),
                        'default': row.get('Default'),
                        'extra': row.get('Extra')
                    }
            logger.info(f"TOOL END: get_table_schema completed. Columns found: {len(schema_info)}. Keys: {list(schema_info.keys())}")
            return schema_info
        except FileNotFoundError as e:
            logger.warning(f"TOOL WARNING: get_table_schema table not found: {e}")
            raise e
        except Exception as e:
            logger.error(f"TOOL ERROR: get_table_schema failed for database_name={database_name}, table_name={table_name}: {e}", exc_info=True)
            raise RuntimeError(f"Could not retrieve schema for table '{database_name}.{table_name}'.")

    async def execute_sql(self, sql_query: str, database_name: Optional[str] = None, parameters: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Executes a read-only SQL query (primarily SELECT, SHOW, DESCRIBE) against a specified database
        and returns the results. Uses parameterized queries for safety.
        Example `parameters`: ["value1", 123] corresponding to %s placeholders in `sql_query`.
        """
        logger.info(f"TOOL START: execute_sql called. database_name={database_name}, sql_query={sql_query[:100]}, parameters={parameters}")
        if database_name and not database_name.isidentifier():
            logger.warning(f"TOOL WARNING: execute_sql called with invalid database_name: {database_name}")
            raise ValueError(f"Invalid database name provided: {database_name}")
        param_tuple = tuple(parameters) if parameters is not None else None
        try:
            results = await self._execute_query(sql_query, params=param_tuple, database=database_name)
            logger.info(f"TOOL END: execute_sql completed. Rows returned: {len(results)}.")
            return results
        except Exception as e:
            logger.error(f"TOOL ERROR: execute_sql failed for database_name={database_name}, sql_query={sql_query[:100]}, parameters={parameters}: {e}", exc_info=True)
            raise
            
    async def create_database(self, database_name: str) -> Dict[str, Any]:
        """
        Creates a new database if it doesn't exist.
        """
        logger.info(f"TOOL START: create_database called for database: '{database_name}'")
        if not database_name or not database_name.isidentifier():
            logger.error(f"Invalid database_name for creation: '{database_name}'. Must be a valid identifier.")
            raise ValueError(f"Invalid database_name for creation: '{database_name}'. Must be a valid identifier.")

        if await self._database_exists(database_name):
            message = f"Database '{database_name}' already exists."
            logger.info(f"TOOL END: create_database. {message}")
            return {"status": "exists", "message": message, "database_name": database_name}

        sql = f"CREATE DATABASE IF NOT EXISTS `{database_name}`;"

        try:
            await self._execute_query(sql, database=None)
            message = f"Database '{database_name}' created successfully."
            logger.info(f"TOOL END: create_database. {message}")
            return {"status": "success", "message": message, "database_name": database_name}
        except Exception as e:
            error_message = f"Failed to create database '{database_name}'."
            logger.error(f"TOOL ERROR: create_database. {error_message} Error: {e}", exc_info=True)
            raise RuntimeError(f"{error_message} Reason: {str(e)}")

    async def create_vector_store(self,
                                  database_name: str,
                                  vector_store_name: str,
                                  embedding_length: int,
                                  distance_function: Optional[str] = None) -> Dict[str, Any]:
        """
        THIS TOOL HELPS IN CREATING A TABLE WHICH STORES EMBEDDINGS. 
        
        Creates a new vector store (table) with a predefined schema if it doesn't already exist.
        First checks if the database exists, creating it if necessary.
        Otherwise, it creates the table with id, document, embedding (VECTOR type), and metadata (JSON) columns.
        A VECTOR INDEX is created on the embedding column.

        Parameters:
        - database_name (str): The target database.
        - vector_store_name (str): The name of the table to create.
        - embedding_length (int): The dimension of the vector embeddings.
        - distance_function (str, optional): 'euclidean' or 'cosine'. Defaults to 'cosine'.
        """
        logger.info(f"TOOL START: create_vector_store called. DB: '{database_name}', Store: '{vector_store_name}', Embedding_Length: {embedding_length}, Distance_Requested: '{distance_function}'")

        # --- Input Validation ---
        if not database_name or not database_name.isidentifier():
            logger.error(f"Invalid database_name: '{database_name}'. Must be a valid identifier.")
            raise ValueError(f"Invalid database_name: '{database_name}'. Must be a valid identifier.")
        if not vector_store_name or not vector_store_name.isidentifier():
            logger.error(f"Invalid vector_store_name: '{vector_store_name}'. Must be a valid identifier.")
            raise ValueError(f"Invalid vector_store_name: '{vector_store_name}'. Must be a valid identifier.")

        if not isinstance(embedding_length, int) or embedding_length <= 0:
            logger.error(f"Invalid embedding_length: {embedding_length}. Must be a positive integer.")
            raise ValueError(f"Invalid embedding_length: {embedding_length}. Must be a positive integer.")

        # Validate and set distance_function
        valid_distance_functions_map = {"euclidean": "EUCLIDEAN", "cosine": "COSINE"}
        processed_distance_function_sql = valid_distance_functions_map["cosine"] # Default

        if distance_function:
            df_lower = distance_function.lower()
            if df_lower in valid_distance_functions_map:
                processed_distance_function_sql = valid_distance_functions_map[df_lower]
            else:
                logger.error(f"Invalid distance_function: '{distance_function}'. Must be one of {list(valid_distance_functions_map.keys())}.")
                raise ValueError(f"Invalid distance_function: '{distance_function}'. Must be one of {list(valid_distance_functions_map.keys())}.")
        else: # Not provided, use default
            logger.info(f"Distance function not provided, defaulting to '{processed_distance_function_sql}'.")
        
        logger.info(f"Using SQL distance function: '{processed_distance_function_sql}'.")

        # --- Database Existence Check ---
        if not await self._database_exists(database_name):
            logger.info(f"Database '{database_name}' does not exist. Attempting to create it.")
            try:
                await self.create_database(database_name) # Call the create_database tool/method
            except Exception as db_create_e:
                logger.error(f"Failed to ensure database '{database_name}' existence: {db_create_e}", exc_info=True)
                raise RuntimeError(f"Failed to ensure database '{database_name}' exists before creating vector store. Reason: {str(db_create_e)}")

        # --- Table Existence Check ---
        if await self._table_exists(database_name, vector_store_name):
            message = f"Vector store (table) '{vector_store_name}' already exists in database '{database_name}'. No action taken."
            logger.info(f"TOOL END: create_vector_store. {message}")
            return {
                "status": "exists",
                "message": message,
                "database_name": database_name,
                "vector_store_name": vector_store_name
            }

        # --- SQL Query for Vector Store Table Creation ---
        schema_query = f"""
        CREATE TABLE IF NOT EXISTS `{vector_store_name}` (
            id VARCHAR(36) NOT NULL DEFAULT UUID_v7() PRIMARY KEY,
            document TEXT NOT NULL,
            embedding VECTOR({embedding_length}) NOT NULL,
            metadata JSON NOT NULL,
            VECTOR INDEX (embedding) DISTANCE={processed_distance_function_sql}
        );
        """
        try:
            # --- Execute Query ---
            await self._execute_query(schema_query, database=database_name)
            
            success_message = f"Vector store '{vector_store_name}' created successfully in database '{database_name}' with {processed_distance_function_sql} distance."
            logger.info(f"TOOL END: create_vector_store completed. {success_message}")
            return {
                "status": "success",
                "message": success_message,
                "database_name": database_name,
                "vector_store_name": vector_store_name
            }
        except Exception as e:
            error_message = f"Failed to create vector store '{vector_store_name}' in database '{database_name}'."
            logger.error(f"TOOL ERROR: create_vector_store failed. {error_message} Error: {e}", exc_info=True)
            raise RuntimeError(f"{error_message} Reason: {str(e)}")

    async def list_vector_stores(self, database_name: str) -> List[str]:
        """
        Lists all tables within the specified database that are identified as vector stores.
        A table is considered a vector store if it contains an indexed column named 'embedding'
        with a data type of 'VECTOR'.

        Parameters:
        - database_name (str): The name of the database to scan.

        Returns:
        - List[str]: A list of table names that are identified as vector stores.
                     Returns an empty list if no such tables are found or if the database doesn't exist.
        
        Raises:
        - ValueError: If the database_name is invalid.
        - RuntimeError: For database errors during the operation.
        """
        logger.info(f"TOOL START: list_vector_stores called for database: '{database_name}'")

        # --- Input Validation ---
        if not database_name or not database_name.isidentifier():
            logger.error(f"Invalid database_name: '{database_name}'. Must be a valid identifier.")
            raise ValueError(f"Invalid database_name: '{database_name}'. Must be a valid identifier.")

        if not await self._database_exists(database_name):
            logger.warning(f"Database '{database_name}' does not exist. Cannot list vector stores.")
            return []

        # --- SQL Query ---
        # This query identifies tables that have:
        # 1. A column named 'embedding'.
        # 2. The data type of this 'embedding' column is 'VECTOR'.
        # 3. This 'embedding' column is part of an index (ensured by the JOIN with STATISTICS).
        sql_query = """
        SELECT DISTINCT T1.TABLE_NAME
        FROM information_schema.COLUMNS AS T1
        INNER JOIN information_schema.STATISTICS AS T2
            ON T1.TABLE_SCHEMA = T2.TABLE_SCHEMA
            AND T1.TABLE_NAME = T2.TABLE_NAME
            AND T1.COLUMN_NAME = T2.COLUMN_NAME
        WHERE T1.TABLE_SCHEMA = %s
          AND UPPER(T1.COLUMN_NAME) = 'EMBEDDING'
          AND UPPER(T1.DATA_TYPE) = 'VECTOR' 
        ORDER BY T1.TABLE_NAME;
        """

        try:
            results = await self._execute_query(sql_query, params=(database_name,), database='information_schema')
            store_list = [row['TABLE_NAME'] for row in results if 'TABLE_NAME' in row]
            
            if not store_list:
                logger.info(f"No vector stores found in database '{database_name}'.")
            else:
                logger.info(f"Found {len(store_list)} vector store(s) in database '{database_name}': {store_list}")
            
            logger.info(f"TOOL END: list_vector_stores completed for database '{database_name}'.")
            return store_list

        except Exception as e:
            error_message = f"Failed to list vector stores in database '{database_name}'."
            logger.error(f"TOOL ERROR: list_vector_stores. {error_message} Error: {e}", exc_info=True)
            raise RuntimeError(f"{error_message} Reason: {str(e)}") 

    # --- Tool Registration (Synchronous) ---
    def register_tools(self):
        """Registers the class methods as MCP tools."""
        if self.pool is None:
             logger.error("Cannot register tools: Database pool is not initialized.")
             raise RuntimeError("Database pool must be initialized before registering tools.")

        self.mcp.add_tool(self.list_databases)
        self.mcp.add_tool(self.list_tables)
        self.mcp.add_tool(self.get_table_schema)
        self.mcp.add_tool(self.execute_sql)
        self.mcp.add_tool(self.create_database)
        self.mcp.add_tool(self.create_vector_store)
        self.mcp.add_tool(self.list_vector_stores)
        logger.info("Registered MCP tools explicitly.")

    # --- Async Main Server Logic ---
    async def run_async_server(self, transport="stdio", host="127.0.0.1", port=9001):
        """
        Initializes pool, registers tools, and runs the appropriate async MCP listener.
        This method should be the target for anyio.run().
        """
        try:
            # 1. Initialize pool within the anyio-managed loop
            await self.initialize_pool()

            # 2. Register tools (synchronous part, but called from async context)
            self.register_tools()

            # 3. Prepare transport arguments
            transport_kwargs = {}
            if transport == "sse":
                transport_kwargs = {"host": host, "port": port}
                logger.info(f"Starting MCP server via {transport} on {host}:{port}...")
            elif transport == "stdio":
                 logger.info(f"Starting MCP server via {transport}...")
            else:
                 logger.error(f"Unsupported transport type: {transport}")
                 return # Exit if transport is invalid

            # 4. Run the appropriate async listener from FastMCP
            await self.mcp.run_async(transport=transport, **transport_kwargs)

        except (ConnectionError, AsyncMyError, RuntimeError) as e:
            logger.critical(f"Server setup failed: {e}", exc_info=True)
            # Error during setup, re-raise to stop server
            raise
        except Exception as e:
            logger.critical(f"Server execution failed with an unexpected error: {e}", exc_info=True)
            # Error during execution, re-raise to stop server
            raise
        finally:
            # Ensure pool is closed when run_async_server finishes or raises an error
            await self.close_pool()


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MariaDB MCP Server")
    parser.add_argument('--transport', type=str, default='stdio', choices=['stdio', 'sse'],
                        help='MCP transport protocol (stdio or sse)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host for SSE transport')
    parser.add_argument('--port', type=int, default=9001,
                        help='Port for SSE transport')
    args = parser.parse_args()

    # 1. Create the server instance
    server = MariaDBServer()
    exit_code = 0

    try:
        # 2. Use anyio.run to manage the event loop and call the main async server logic
        anyio.run(
            partial(server.run_async_server, transport=args.transport, host=args.host, port=args.port)
        )
        logger.info("Server finished gracefully.") # Should only be reached if server stops normally

    except KeyboardInterrupt:
         logger.info("Server execution interrupted by user.")
    except Exception as e:
         logger.critical(f"Server failed to start or crashed: {e}", exc_info=True)
         exit_code = 1 # Indicate failure
    finally:
        logger.info(f"Server exiting with code {exit_code}.")
