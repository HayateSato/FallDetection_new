"""
InfluxDB Data Fetcher - Singleton Client
Provides a persistent connection to prevent port exhaustion on Windows.
"""

from influxdb_client import InfluxDBClient
from config import settings
import logging
import atexit

logger = logging.getLogger(__name__)

# Singleton InfluxDB client to prevent connection exhaustion
_influxdb_client_instance = None


def _get_influxdb_client():
    """
    Returns a singleton InfluxDB client to prevent port exhaustion.

    This reuses the same connection instead of creating a new one for each query,
    which prevents Windows error: [WinError 10048] "Only one usage of each socket
    address is normally permitted"
    """
    global _influxdb_client_instance

    if _influxdb_client_instance is None:
        logger.info("Creating new InfluxDB client connection...")
        _influxdb_client_instance = InfluxDBClient(
            url=settings.INFLUXDB_URL,
            token=settings.INFLUXDB_TOKEN,
            org=settings.INFLUXDB_ORG,
            bucket=settings.INFLUXDB_BUCKET,
            timeout=30_000,
            verify_ssl=True
        )
        # Register cleanup on exit
        atexit.register(_close_influxdb_client)
        logger.info("InfluxDB client connection established")

    return _influxdb_client_instance


def _close_influxdb_client():
    """Close the singleton InfluxDB client on application exit"""
    global _influxdb_client_instance
    if _influxdb_client_instance is not None:
        logger.info("Closing InfluxDB client connection...")
        _influxdb_client_instance.close()
        _influxdb_client_instance = None


def fetch_data(query):
    """
    Fetches data from InfluxDB using a persistent connection.

    This function reuses the same InfluxDB client to avoid connection exhaustion
    issues on Windows (WinError 10048).
    """
    try:
        client = _get_influxdb_client()
        query_api = client.query_api()
        tables = query_api.query(query)
        logger.info(f"Query successful, received {len(tables)} tables")
        return tables
    except Exception as e:
        logger.error(f"InfluxDB connection error: {type(e).__name__}: {str(e)}")
        logger.error(f"URL: {settings.INFLUXDB_URL}")
        logger.error(f"Org: {settings.INFLUXDB_ORG}")
        logger.error(f"Bucket: {settings.INFLUXDB_BUCKET}")

        # Provide specific troubleshooting hints
        error_msg = str(e).lower()
        if "timeout" in error_msg:
            logger.error("TROUBLESHOOTING: Connection timeout detected")
            logger.error("  Check if VPN is connected (if required)")
            logger.error("  Verify firewall settings")
        elif "ssl" in error_msg or "certificate" in error_msg:
            logger.error("TROUBLESHOOTING: SSL certificate issue")
            logger.error("  Try setting verify_ssl=False in data_fetcher.py")
        elif "unauthorized" in error_msg or "401" in error_msg:
            logger.error("TROUBLESHOOTING: Authentication failed")
            logger.error("  Check if token is valid and not expired")

        raise
