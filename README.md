# MCP MariaDB Server

This project implements an MCP (Model Context Protocol) server that provides tools for interacting with a MariaDB database via an AI Assistant like Cascade.

## Core Components

*   `server.py`: Contains the main MCP server logic and tool definitions.
*   `config.py`: Handles loading configuration (likely from `.env`).
*   `tests/`: Contains manual test documentation for the MCP tools.

## Available Tools

This server provides the following tools for interacting with MariaDB:

*   **`list_databases`**:
    *   **Functionality:** Lists all accessible databases on the connected MariaDB server.
    *   **Parameters:** None.
*   **`list_tables`**:
    *   **Functionality:** Lists all tables within a specified database.
    *   **Parameters:** `database_name` (string, required).
*   **`get_table_schema`**:
    *   **Functionality:** Retrieves the schema (column names, types, etc.) for a specific table in a database.
    *   **Parameters:** `database_name` (string, required), `table_name` (string, required).
*   **`execute_sql`**:
    *   **Functionality:** Executes a read-only SQL query (e.g., `SELECT`, `SHOW`, `DESCRIBE`) against a specified database. Supports parameterized queries (`%s` placeholders) for safety. If `MCP_READ_ONLY` is enabled in the configuration, it attempts to prevent non-read-only commands.
    *   **Parameters:** `sql_query` (string, required), `database_name` (string, required), `parameters` (list, optional - values corresponding to `%s` placeholders).

## Setup

1.  **Environment:** This project uses `uv` for dependency management (indicated by `uv.lock`). Ensure `uv` is installed.
2.  **Configuration:** Create a `.env` file in the root directory with your MariaDB connection details. Example:
    ```dotenv
    # Example .env content
    DB_HOST=localhost
    DB_USER=your_db_user
    DB_PASSWORD=your_db_password
    # DB_PORT=3306 # Optional, defaults to 3306 if omitted
    # DB_NAME=your_default_database # Optional, can often be specified per query

    # Optional: Enforce read-only mode for execute_sql
    # Set to True or 1 to enable. If omitted or set to False/0, the check might be less strict.
    # MCP_READ_ONLY=True
    ```
3.  **Dependencies:** Install the required Python packages:
    ```bash
    uv pip sync
    ```
4.  **Running the Server:** Start the MCP server (adjust the command if `main.py` is the entry point instead of `server.py`):
    ```bash
    python server.py
    ```

## Testing

Manual tests focusing on the read-only capabilities of the MCP tools were performed using an AI Assistant.

Details about the testing approach, specific test cases, and observed results can be found in the `tests` directory:

*   See `tests/README.md` for an overview.
*   See `tests/test_mariadb_mcp_tools.py` for documented test steps.
