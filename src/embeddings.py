import logging
from typing import List, Optional, Dict, Any

# Import configuration variables and the logger instance
from config import (
    EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    logger
)

# Import specific client libraries
try:
    from openai import AsyncOpenAI, OpenAIError
except ImportError:
    logger.warning("OpenAI library not installed. OpenAI provider will not be available.")
    AsyncOpenAI = None # type: ignore
    OpenAIError = Exception # type: ignore # Generic exception if library missing

# --- Model Definitions ---
# Define allowed models and defaults for each provider
ALLOWED_OPENAI_MODELS: List[str] = ["text-embedding-3-small", "text-embedding-3-large"]
DEFAULT_OPENAI_MODEL: str = "text-embedding-3-small"
# Mapping of model names to their embedding dimensions (update as needed)
OPENAI_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072
}
# If you add Gemini support, add its model/dimension mapping here as well.

class EmbeddingService:
    """
    Provides an interface to generate text embeddings using a configured provider
    (OpenAI or Google Gemini) and allows model selection at runtime.
    """
    def __init__(self):
        """
        Initializes the embedding service based on configuration.
        Sets up the appropriate asynchronous client for OpenAI or configures Gemini.
        """
        self.provider = EMBEDDING_PROVIDER
        self.openai_client: Optional[AsyncOpenAI] = None
        self.allowed_models: List[str] = []
        self.default_model: str = ""

        logger.info(f"Initializing EmbeddingService with provider: {self.provider}")

        if self.provider == "openai":
            if not AsyncOpenAI:
                 logger.error("OpenAI provider selected, but 'openai' library is not installed.")
                 raise ImportError("OpenAI library not found. Please install it.")
            if not OPENAI_API_KEY:
                 logger.error("OpenAI API key is missing.")
                 raise ValueError("OpenAI API key is required for the OpenAI provider.")
            try:
                self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
                self.allowed_models = ALLOWED_OPENAI_MODELS
                self.default_model = DEFAULT_OPENAI_MODEL
                logger.info(f"OpenAI client initialized. Default model: {self.default_model}. Allowed: {self.allowed_models}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
                raise RuntimeError(f"OpenAI client initialization failed: {e}")
        else:
            logger.error(f"Unsupported embedding provider configured: {self.provider}")
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def get_allowed_models(self) -> List[str]:
        """Returns the list of allowed model names for the current provider."""
        return self.allowed_models

    def get_default_model(self) -> str:
        """Returns the default model name for the current provider."""
        return self.default_model

    async def get_embedding_dimension(self, model_name: Optional[str] = None) -> int:
        """
        Asynchronously returns the embedding vector dimension for the given model (or default model if not specified).
        Raises ValueError if the model is invalid or dimension unknown.
        """
        # If in the future you want to fetch dimensions from an API, this can be awaited
        if self.provider == "openai":
            model = model_name or self.default_model
            if model not in OPENAI_MODEL_DIMENSIONS:
                logger.error(f"Unknown dimension for OpenAI model '{model}'. Known: {list(OPENAI_MODEL_DIMENSIONS.keys())}")
                raise ValueError(f"Unknown dimension for OpenAI model '{model}'.")
            return OPENAI_MODEL_DIMENSIONS[model]
        else:
            logger.error(f"get_embedding_dimension not implemented for provider: {self.provider}")
            raise NotImplementedError(f"Embedding dimension lookup not implemented for provider: {self.provider}")

    from typing import Union

    async def embed(self, text: Union[str, List[str]], model_name: Optional[str] = None) -> Union[List[float], List[List[float]]]:
        """
        Generates embedding(s) for a single document or a list of documents using the configured provider.

        Parameters:
        - text (str or List[str]): The text(s) to embed.
        - model_name (str, optional): The specific model to use. If None, uses the provider's default model.

        Returns:
        - List[float]: The generated embedding vector (if input is str).
        - List[List[float]]: The generated embedding vectors (if input is List[str]).

        Raises:
        - ValueError: If an invalid model_name is provided or input is empty/invalid.
        - RuntimeError: If the embedding API call fails for other reasons.
        """
        # Validate input
        if isinstance(text, str):
            if not text:
                logger.error("Embedding requested for empty string, which is not allowed.")
                raise ValueError("Cannot generate embedding for empty text.")
            texts = [text]
            single_input = True
        elif isinstance(text, list):
            if not text:
                logger.error("Embedding requested for empty list, which is not allowed.")
                raise ValueError("Cannot generate embedding for empty list.")
            if not all(isinstance(t, str) and t for t in text):
                logger.error("Embedding requested for a list containing non-string or empty elements.")
                raise ValueError("All elements in the input list must be non-empty strings.")
            texts = text
            single_input = False
        else:
            logger.error(f"Embedding requested for unsupported input type: {type(text)}")
            raise ValueError("Input must be a string or a list of strings.")

        target_model = model_name
        if target_model:
            if target_model not in self.allowed_models:
                logger.error(f"Invalid model '{target_model}' requested for provider '{self.provider}'. Allowed: {self.allowed_models}")
                raise ValueError(f"Model '{target_model}' is not allowed for the '{self.provider}' provider. Choose from: {self.allowed_models}")
        else:
            target_model = self.default_model
            logger.debug(f"No model specified, using default for {self.provider}: {target_model}")

        logger.debug(f"Requesting embedding using model '{target_model}' for {len(texts)} text(s). Example (first 50 chars): '{texts[0][:50]}...'")

        try:
            if self.provider == "openai":
                if not self.openai_client:
                    logger.critical("OpenAI client not initialized during embed call.")
                    raise RuntimeError("OpenAI client not initialized.")
                response = await self.openai_client.embeddings.create(
                    input=texts,
                    model=target_model
                )
                if response.data and len(response.data) == len(texts):
                    embeddings = [d.embedding for d in response.data]
                    logger.debug(f"OpenAI embedding(s) received. Count: {len(embeddings)}, Dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
                    return embeddings[0] if single_input else embeddings
                else:
                    logger.error("OpenAI embedding API response did not contain expected data or count mismatch.")
                    raise RuntimeError("Invalid response structure from OpenAI embedding API.")
            else:
                raise RuntimeError(f"Embed called with unsupported provider: {self.provider}")

        except OpenAIError as e:
            logger.error(f"OpenAI API error during embedding: {e}", exc_info=True)
            raise RuntimeError(f"OpenAI API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during embedding with {self.provider} model {target_model}: {e}", exc_info=True)
            raise RuntimeError(f"Embedding generation failed: {e}")
