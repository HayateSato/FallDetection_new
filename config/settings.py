"""
Configuration settings for Fall Detection System.
Loads from environment variables with sensible defaults.

Supports dynamic model selection via MODEL_VERSION environment variable.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# MODEL SELECTION
# =============================================================================

# Model version to use (v0, v1, v2, v3, v4, v5, v1_tuned, v3_tuned)
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v3').lower()

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

if MODEL_VERSION == 'v0':
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
# MONITORING SETTINGS
# =============================================================================

MONITORING_ENABLED = os.getenv('MONITORING_ENABLED', 'true').lower() == 'true'
MONITORING_INTERVAL_SECONDS = int(os.getenv('MONITORING_INTERVAL_SECONDS', '5'))
MONITORING_LOOKBACK_SECONDS = int(os.getenv('MONITORING_LOOKBACK_SECONDS', '15'))

# Additional sensor data collection
COLLECT_ADDITIONAL_SENSORS = os.getenv('COLLECT_ADDITIONAL_SENSORS', 'false').lower() == 'true'

# =============================================================================
# EXPORT SETTINGS
# =============================================================================

TIMEZONE_OFFSET_HOURS = float(os.getenv('TIMEZONE_OFFSET_HOURS', '0'))
TODAY = datetime.today().strftime('%Y%m%d')
BASE_DIR = os.getenv('FALL_DATA_EXPORT_DIR', 'results/')
FALL_DATA_EXPORT_DIR = os.path.join(BASE_DIR, 'exported_flaskApp_data', TODAY)

# Create export directory if it doesn't exist
export_path = Path(FALL_DATA_EXPORT_DIR)
export_path.mkdir(parents=True, exist_ok=True)

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

    Returns:
        Path to the model file
    """
    if MODEL_PATH_OVERRIDE:
        return MODEL_PATH_OVERRIDE

    # Default paths for each version
    default_paths = {
        'v0': 'model/model_v0/model_v0_xgboost.pkl',
        'v1': 'model/model_v1/model_v1_xgboost.pkl',
        'v2': 'model/model_v2/model_v2_xgboost.pkl',
        'v3': 'model/model_v3/model_v3_xgboost.pkl',
        'v4': 'model/model_v4/model_v4_xgboost.pkl',
        'v5': 'model/model_v5/model_v5_xgboost.pkl',
        'v1_tuned': 'model/model_v1_tuned/model_v1_tuned.pkl',
        'v3_tuned': 'model/model_v3_tuned/model_v3_tuned.pkl',
    }

    version_lower = MODEL_VERSION.lower()
    return default_paths.get(version_lower, default_paths['v3'])

# Set MODEL_PATH for backward compatibility
MODEL_PATH = get_model_path()

# =============================================================================
# PUBLIC ENDPOINT / API SECURITY SETTINGS
# =============================================================================

# Enable public endpoint mode (adds authentication, rate limiting, production settings)
PUBLIC_ENDPOINT_ENABLED = os.getenv('PUBLIC_ENDPOINT_ENABLED', 'false').lower() == 'true'

# Tunnel mode: 'local', 'ngrok', or 'cloudflare'
TUNNEL_MODE = os.getenv('TUNNEL_MODE', 'local').lower()
if TUNNEL_MODE not in ('local', 'ngrok', 'cloudflare'):
    print(f"WARNING: Invalid TUNNEL_MODE '{TUNNEL_MODE}', defaulting to 'local'")
    TUNNEL_MODE = 'local'

# API Keys for authentication (comma-separated list of valid keys)
# Generate keys with: python -c "import secrets; print(secrets.token_urlsafe(32))"
API_KEYS = [k.strip() for k in os.getenv('API_KEYS', '').split(',') if k.strip()]

# Rate limiting: requests per minute per IP
RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '30'))

# CORS allowed origins (comma-separated, or * for all)
CORS_ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', '*')

# Flask debug mode (automatically disabled when PUBLIC_ENDPOINT_ENABLED=true)
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
if PUBLIC_ENDPOINT_ENABLED:
    FLASK_DEBUG = False  # Force disable debug in public mode

# ngrok settings
NGROK_REGION = os.getenv('NGROK_REGION', 'eu')

# Cloudflare Tunnel settings
CLOUDFLARE_TUNNEL_TOKEN = os.getenv('CLOUDFLARE_TUNNEL_TOKEN', '')

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

def print_config():
    """Print current configuration"""
    print("="*60)
    print("Fall Detection System Configuration")
    print("="*60)
    print(f"  Model Version:     {MODEL_VERSION.upper()}")
    print(f"  Model Path:        {MODEL_PATH}")
    print(f"  ACC Sensor Type:   {ACC_SENSOR_TYPE}")
    print(f"  ACC Fields:        {ACC_FIELD_X}, {ACC_FIELD_Y}, {ACC_FIELD_Z}")
    print(f"  Hardware ACC Rate: {HARDWARE_ACC_SAMPLE_RATE} Hz")
    print(f"  Model ACC Rate:    {MODEL_ACC_SAMPLE_RATE} Hz")
    if UPSAMPLING_ENABLED:
        print(f"  Resampling:        Upsampling ({HARDWARE_ACC_SAMPLE_RATE}Hz -> {MODEL_ACC_SAMPLE_RATE}Hz, method={RESAMPLING_METHOD})")
    elif DOWNSAMPLING_ENABLED:
        print(f"  Resampling:        Downsampling ({HARDWARE_ACC_SAMPLE_RATE}Hz -> {MODEL_ACC_SAMPLE_RATE}Hz, method={RESAMPLING_METHOD})")
    else:
        print(f"  Resampling:        Disabled (no conversion needed)")
    print(f"  Window Size:       {WINDOW_SIZE_SECONDS}s ({HARDWARE_WINDOW_SAMPLES} HW samples -> {WINDOW_SAMPLES} model samples)")
    print(f"  Barometer:         {'Enabled' if BAROMETER_ENABLED else 'Disabled' + (' (V0 model)' if MODEL_VERSION == 'v0' else '')}")
    if BAROMETER_ENABLED:
        print(f"  BARO Sample Rate:  {BARO_SAMPLE_RATE} Hz")
        print(f"  Barometer Field:   {BAROMETER_FIELD}")
    print(f"  Monitoring:        {'Enabled' if MONITORING_ENABLED else 'Disabled'}")
    print(f"  Monitor Interval:  {MONITORING_INTERVAL_SECONDS}s")
    print("="*60)
