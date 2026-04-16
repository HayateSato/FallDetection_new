"""
Configuration settings for Fall Detection System.
Loads from environment variables with sensible defaults.

Supports dynamic model selection via MODEL_VERSION environment variable.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# MODEL SELECTION
# =============================================================================

# Model version to use (v0, v1, v2, v3, v4, v5, v1_tuned, v3_tuned)
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v0').lower()

# Optional custom model path (overrides default for selected version)
MODEL_PATH_OVERRIDE = os.getenv('MODEL_PATH', None)

# Barometer field name in InfluxDB
BAROMETER_FIELD = os.getenv('BAROMETER_FIELD', 'bmp_pressure')

# =============================================================================
# INFLUXDB SETTINGS
# =============================================================================

INFLUXDB_URL = os.getenv('INFLUXDB_URL', '')
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN', '')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG', '')
INFLUXDB_BUCKET = os.getenv('INFLUXDB_BUCKET', '')

# =============================================================================
# PATIENT SETTINGS
# =============================================================================

# Comma-separated patient identifiers (used by mock_app and retrain/seed_test_data)
PATIENT_IDS = os.getenv('PATIENT_IDS', '')

# Comma-separated MAC addresses, positional 1:1 with PATIENT_IDS
MAC_IDS = os.getenv('MAC_IDS', '')

# =============================================================================
# ACCELEROMETER SENSOR CONFIGURATION
# =============================================================================

# Sensor type: 'bosch' or 'non_bosch'
# - bosch: Uses bosch_acc_x/y/z field names in InfluxDB
# - non_bosch: Uses acc_x/y/z field names in InfluxDB
ACC_SENSOR_TYPE = os.getenv('ACC_SENSOR_TYPE', 'bosch').lower()

# Validate sensor type
if ACC_SENSOR_TYPE not in ('bosch', 'non_bosch'):
    print(f"WARNING: Invalid ACC_SENSOR_TYPE '{ACC_SENSOR_TYPE}', defaulting to 'bosch'")
    ACC_SENSOR_TYPE = 'bosch'

# Set accelerometer field names based on sensor type
if ACC_SENSOR_TYPE == 'bosch':
    ACC_FIELD_X = 'bosch_acc_x'
    ACC_FIELD_Y = 'bosch_acc_y'
    ACC_FIELD_Z = 'bosch_acc_z'
else:
    # non_bosch sensor
    ACC_FIELD_X = 'acc_x'
    ACC_FIELD_Y = 'acc_y'
    ACC_FIELD_Z = 'acc_z'

# Sensor calibration: transforms non_bosch values to bosch-equivalent values
# This is automatically enabled when ACC_SENSOR_TYPE is 'non_bosch'
# The transformation uses a pre-computed matrix from calibration data
SENSOR_CALIBRATION_ENABLED = ACC_SENSOR_TYPE == 'non_bosch'

# =============================================================================
# SAMPLING RATE CONFIGURATION
# =============================================================================

# Hardware accelerometer sampling rate: 25, 50, or 100 Hz
# - 25hz: Upsamples to 50Hz for model compatibility
# - 50hz: No resampling needed (model's native rate)
# - 100hz: Downsamples to 50Hz for model compatibility
HARDWARE_ACC_SAMPLE_RATE = int(os.getenv('HARDWARE_ACC_SAMPLE_RATE', '50'))

# Validate hardware sample rate
if HARDWARE_ACC_SAMPLE_RATE not in (25, 50, 100):
    print(f"WARNING: Invalid HARDWARE_ACC_SAMPLE_RATE '{HARDWARE_ACC_SAMPLE_RATE}', defaulting to 50")
    HARDWARE_ACC_SAMPLE_RATE = 50

# Model's expected sample rate (fixed at 50Hz - what the model was trained on)
MODEL_ACC_SAMPLE_RATE = 50  # Hz - this is fixed, models are trained at 50Hz

# Determine if resampling is needed
UPSAMPLING_ENABLED = HARDWARE_ACC_SAMPLE_RATE < MODEL_ACC_SAMPLE_RATE  # 25Hz -> 50Hz
DOWNSAMPLING_ENABLED = HARDWARE_ACC_SAMPLE_RATE > MODEL_ACC_SAMPLE_RATE  # 100Hz -> 50Hz
RESAMPLING_ENABLED = UPSAMPLING_ENABLED or DOWNSAMPLING_ENABLED

# Resampling method: 'linear' for upsampling, 'decimate' or 'average' for downsampling
RESAMPLING_METHOD = os.getenv('RESAMPLING_METHOD', 'linear')  # 'linear', 'decimate', or 'average'

# The effective sample rate after resampling (always 50Hz for model)
ACC_SAMPLE_RATE = MODEL_ACC_SAMPLE_RATE

# Barometer sampling rate (only used when barometer is enabled)
BARO_SAMPLE_RATE = int(os.getenv('BAROMETER_SAMPLING_RATE', '25'))  # Hz

# Legacy alias
SAMPLING_RATE = ACC_SAMPLE_RATE

# =============================================================================
# BAROMETER CONFIGURATION
# =============================================================================

# Barometer is automatically disabled for V0 model (ACC-only model)
# For other models, barometer is enabled by default but can be manually disabled
_BARO_MANUAL_OVERRIDE = os.getenv('BAROMETER_ENABLED', None)

if MODEL_VERSION == 'v0' or MODEL_VERSION == 'v0_lsb_int':
    # V0 model doesn't use barometer - always disabled
    BAROMETER_ENABLED = False
    BARO_SAMPLE_RATE = 0
elif _BARO_MANUAL_OVERRIDE is not None:
    # Manual override from env file
    BAROMETER_ENABLED = _BARO_MANUAL_OVERRIDE.lower() == 'true'
    if not BAROMETER_ENABLED:
        BARO_SAMPLE_RATE = 0
else:
    # Default: enabled for models that use barometer
    BAROMETER_ENABLED = True

# =============================================================================
# WINDOW CONFIGURATION
# =============================================================================

# Window size in seconds
WINDOW_SIZE_SECONDS = 9  # seconds

# Calculated window samples (based on model's expected rate after resampling)
WINDOW_SAMPLES = WINDOW_SIZE_SECONDS * ACC_SAMPLE_RATE  # 9 * 50 = 450 samples
BARO_WINDOW_SAMPLES = WINDOW_SIZE_SECONDS * BARO_SAMPLE_RATE if BARO_SAMPLE_RATE > 0 else 0

# Hardware window samples (before resampling)
HARDWARE_WINDOW_SAMPLES = WINDOW_SIZE_SECONDS * HARDWARE_ACC_SAMPLE_RATE

# =============================================================================
# PREPROCESSING SETTINGS (for reference, used by model registry)
# =============================================================================

ACC_PREPROCESSING_VERSION = os.getenv('ACC_PREPROCESSING_VERSION', 'v1_features')
BARO_PREPROCESSING_VERSION = os.getenv('BARO_PREPROCESSING_VERSION', 'v1_ema')
ACC_IMPACT_THRESHOLD_G = float(os.getenv('ACC_IMPACT_THRESHOLD_G', '4.0'))
BARO_SLOPE_LIMIT = float(os.getenv('BARO_SLOPE_LIMIT', '25'))
BARO_MA_WINDOW_SECONDS = float(os.getenv('BARO_MA_WINDOW_SECONDS', '1.0'))

# =============================================================================
# MODEL PATH RESOLUTION
# =============================================================================

def get_model_path() -> str:
    """
    Get the model file path based on MODEL_VERSION setting.

    If MODEL_PATH is set in .env, that override takes priority.
    Otherwise, the path is looked up from the model registry (single source of truth).

    Returns:
        Path to the model file
    """
    if MODEL_PATH_OVERRIDE:
        return MODEL_PATH_OVERRIDE

    # Fall back to registry's canonical path (avoids maintaining a duplicate dict here)
    from app.core.model_registry import get_model_name, get_model_path as registry_get_model_path
    model_type = get_model_name(MODEL_VERSION)
    return registry_get_model_path(model_type)

# Set MODEL_PATH for backward compatibility
MODEL_PATH = get_model_path()

# =============================================================================
# PUBLIC ENDPOINT / API SECURITY SETTINGS
# =============================================================================

# Enable public endpoint mode (adds authentication, rate limiting, production settings)
PUBLIC_ENDPOINT_ENABLED = os.getenv('PUBLIC_ENDPOINT_ENABLED', 'false').lower() == 'true'

# API Keys for authentication (comma-separated list of valid keys)
# Generate keys with: python -c "import secrets; print(secrets.token_urlsafe(32))"
API_KEYS = [k.strip() for k in os.getenv('API_KEYS', '').split(',') if k.strip()]

# Rate limiting: requests per minute per IP
RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '30'))

# CORS allowed origins (comma-separated, or * for all)
CORS_ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', '*')

# Tunnel mode — not active in current deployment; reserved for future ngrok/cloudflare use
# TUNNEL_MODE = os.getenv('TUNNEL_MODE', 'local').lower()
# NGROK_REGION = os.getenv('NGROK_REGION', 'eu')
# CLOUDFLARE_TUNNEL_TOKEN = os.getenv('CLOUDFLARE_TUNNEL_TOKEN', '')
