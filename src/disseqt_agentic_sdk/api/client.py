"""
Client helper functions for accessing the global client instance.
"""


from disseqt_agentic_sdk.client import DisseqtAgenticClient
from disseqt_agentic_sdk.utils.logging import get_logger

logger = get_logger()

# Global client instance
_client: DisseqtAgenticClient | None = None


def get_client() -> DisseqtAgenticClient | None:
    """
    Get the global client instance.

    Returns:
        DisseqtAgenticClient or None: The global client, or None if not initialized
    """
    return _client


def set_client(client: DisseqtAgenticClient) -> None:
    """
    Set the global client instance.

    Args:
        client: The client instance to set as global
    """
    global _client
    _client = client


def get_current_client() -> DisseqtAgenticClient:
    """
    Get the current client instance.

    Raises:
        RuntimeError: If SDK has not been initialized (call init() first)

    Returns:
        DisseqtAgenticClient: The global client instance
    """
    if _client is None:
        logger.error("Attempted to get client before SDK initialization")
        raise RuntimeError(
            "SDK not initialized. Call init() first:\n"
            "from disseqt_agentic_sdk import init\n"
            "init(api_key='...', org_id='...', project_id='...', service_name='...')"
        )
    return _client


def flush() -> None:
    """
    Flush all buffered spans to backend immediately.

    Raises:
        RuntimeError: If SDK has not been initialized
    """
    logger.debug("Flushing spans via public API")
    client = get_current_client()
    client.flush()


def shutdown() -> None:
    """
    Shutdown the SDK - flush all buffered spans and cleanup.

    Raises:
        RuntimeError: If SDK has not been initialized
    """
    logger.info("Shutting down SDK via public API")
    client = get_current_client()
    client.shutdown()
    global _client
    _client = None
    logger.info("SDK shutdown complete")


def is_initialized() -> bool:
    """
    Check if SDK has been initialized.

    Returns:
        bool: True if SDK is initialized, False otherwise
    """
    return _client is not None
