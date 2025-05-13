import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE_PATH = os.getenv("LOG_FILE", "logs/mcp_server.log")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 10 * 1024 * 1024))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Create formatter
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Remove existing handlers to avoid duplication if script is reloaded
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# File Handler - Ensure log directory exists
log_file = Path(LOG_FILE_PATH)
log_file.parent.mkdir(parents=True, exist_ok=True)

file_handler = RotatingFileHandler(
    log_file,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT
)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# --- Database Configuration ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# --- MCP Server Configuration ---
MCP_READ_ONLY = os.getenv("MCP_READ_ONLY", "true").lower() == "true"
MCP_MAX_POOL_SIZE = int(os.getenv("MCP_MAX_POOL_SIZE", 10))

# --- Embedding Configuration ---
# Provider selection ('openai' or 'gemini')
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Validation ---
if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    logger.error("Database credentials (DB_USER, DB_PASSWORD, DB_NAME) not found in environment variables or .env file.")

# Embedding Provider and Keys
logger.info(f"Selected Embedding Provider: {EMBEDDING_PROVIDER}")
if EMBEDDING_PROVIDER == "openai":
    if not OPENAI_API_KEY:
        logger.error("EMBEDDING_PROVIDER is 'openai' but OPENAI_API_KEY is missing.")
        raise ValueError("OpenAI API key is required when EMBEDDING_PROVIDER is 'openai'.")
else:
    logger.error(f"Invalid EMBEDDING_PROVIDER specified: '{EMBEDDING_PROVIDER}'. Use 'openai' or 'gemini'.")
    raise ValueError(f"Invalid EMBEDDING_PROVIDER: '{EMBEDDING_PROVIDER}'.")

logger.info(f"Read-only mode: {MCP_READ_ONLY}")
logger.info(f"Logging to console and to file: {LOG_FILE_PATH} (Level: {LOG_LEVEL}, MaxSize: {LOG_MAX_BYTES}B, Backups: {LOG_BACKUP_COUNT})")
