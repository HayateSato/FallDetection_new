"""
InfluxDB client manager - Singleton Client
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
