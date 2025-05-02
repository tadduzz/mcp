from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

import mariadb
from mcp.server.fastmcp import FastMCP

from mcp_server_mariadb_vector.settings import DatabaseSettings


@dataclass
class AppContext:
    conn: mariadb.Connection


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Open a MariaDB connection for the duration of the FastMCP session."""

    cfg = DatabaseSettings()
    conn = mariadb.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        database=cfg.database,
    )
    conn.autocommit = True

    try:
        yield AppContext(conn=conn)
    finally:
        conn.close()
