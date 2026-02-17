"""
Fall Detection System - Entry Point

Initializes logging, model, Flask app, registers route blueprints,
and starts continuous monitoring.

Set MODEL_VERSION and DATA_SOURCE in .env file.
"""
from flask import Flask, send_from_directory, Response
import sys
import os
import logging
from datetime import datetime

# Import app modules
from app.core.inference_engine import PipelineSelector
from app.core.model_registry import get_model_type, get_model_config
from app.services.continuous_monitoring import ContinuousMonitor
from app.data_input.data_loader.csv_dataloader import process_csv_file

# Import shared state and helpers
from app.utils import shared_state
from app.middleware.api_security import setup_cors
from app.data_output.data_exporter import export_detection_data

# Import settings
from config.settings import (
    MODEL_VERSION,
    MODEL_PATH,
    MONITORING_ENABLED,
    MONITORING_INTERVAL_SECONDS,
    WINDOW_SIZE_SECONDS,
    TODAY,
    ACC_SENSOR_TYPE,
    HARDWARE_ACC_SAMPLE_RATE,
    MODEL_ACC_SAMPLE_RATE,
    BAROMETER_ENABLED,
    UPSAMPLING_ENABLED,
    DOWNSAMPLING_ENABLED,
    RESAMPLING_METHOD,
    ACC_FIELD_X,
    ACC_FIELD_Y,
    ACC_FIELD_Z,
    WINDOW_SAMPLES,
    HARDWARE_WINDOW_SAMPLES,
    SENSOR_CALIBRATION_ENABLED,
    PUBLIC_ENDPOINT_ENABLED,
    TUNNEL_MODE,
    API_KEYS,
    RATE_LIMIT_PER_MINUTE,
    FLASK_DEBUG,
)

# Data source settings
DATA_SOURCE = os.getenv('DATA_SOURCE', 'influx').lower()
CSV_FILE_PATH = os.getenv('CSV_FILE_PATH', '')
CSV_WINDOW_INTERVAL_SECONDS = float(os.getenv('CSV_WINDOW_INTERVAL_SECONDS', '1.0'))


# ------------------------------------------------------------------------
# SETUP LOGGING
# ------------------------------------------------------------------------

log_dir = os.path.join("results", "logs", TODAY)
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"fall_detection_{MODEL_VERSION}_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    ]
)

class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

logger = logging.getLogger(__name__)
logger.info(f"=== Logging started ===")
logger.info(f"Logs will be saved to: {log_filename}")


# ------------------------------------------------------------------------
# MODEL INITIALIZATION
# ------------------------------------------------------------------------

model_type = get_model_type(MODEL_VERSION)
model_config = get_model_config(model_type)
inference_engine = PipelineSelector(MODEL_VERSION, MODEL_PATH)
model_info = inference_engine.get_model_info()


# ------------------------------------------------------------------------
# FLASK APP + BLUEPRINTS
# ------------------------------------------------------------------------

app = Flask(__name__, static_folder='app/static')

# Store shared objects on app.config so blueprints can access via current_app
app.config['inference_engine'] = inference_engine
app.config['model_info'] = model_info
app.config['model_config'] = model_config

# Setup CORS (only active when PUBLIC_ENDPOINT_ENABLED=true)
setup_cors(app)

# Register route blueprints
from app.routes.detection import detection_bp
from app.routes.recording import recording_bp
from app.routes.monitoring import monitoring_bp

app.register_blueprint(detection_bp)
app.register_blueprint(recording_bp)
app.register_blueprint(monitoring_bp)


@app.route('/')
def index() -> Response:
    """Serve the frontend HTML page."""
    return send_from_directory('app/static', 'index.html')


# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------

if __name__ == '__main__':
    logger.info(f"="*70)
    logger.info(f"Starting Fall Detection System")
    logger.info(f"  Data Source:     {DATA_SOURCE}")
    logger.info(f"  Monitoring Interval: {MONITORING_INTERVAL_SECONDS} seconds")
    if DATA_SOURCE == 'csv':
        logger.info(f"  CSV Path:        {CSV_FILE_PATH}")
        logger.info(f"  Window Interval: {CSV_WINDOW_INTERVAL_SECONDS} seconds")
    logger.info(f"  Model Version:   {MODEL_VERSION.upper()}")
    logger.info(f"  Model Path:      {MODEL_PATH}")
    logger.info(f"  ACC Sensor Type: {ACC_SENSOR_TYPE}")
    logger.info(f"  ACC Fields:      {ACC_FIELD_X}, {ACC_FIELD_Y}, {ACC_FIELD_Z}")
    logger.info(f"  Hardware Rate:   {HARDWARE_ACC_SAMPLE_RATE} Hz")
    logger.info(f"  Model Rate:      {MODEL_ACC_SAMPLE_RATE} Hz")
    if UPSAMPLING_ENABLED:
        logger.info(f"  Resampling:      Upsampling ({HARDWARE_ACC_SAMPLE_RATE}Hz -> {MODEL_ACC_SAMPLE_RATE}Hz, method={RESAMPLING_METHOD})")
    elif DOWNSAMPLING_ENABLED:
        logger.info(f"  Resampling:      Downsampling ({HARDWARE_ACC_SAMPLE_RATE}Hz -> {MODEL_ACC_SAMPLE_RATE}Hz, method={RESAMPLING_METHOD})")
    else:
        logger.info(f"  Resampling:      Disabled (no conversion needed)")
    logger.info(f"  Window Size:     {WINDOW_SIZE_SECONDS}s ({HARDWARE_WINDOW_SAMPLES} HW -> {WINDOW_SAMPLES} model samples)")
    logger.info(f"  Barometer:       {'Enabled' if BAROMETER_ENABLED else 'Disabled'}")
    logger.info(f"  Uses Barometer:  {model_info['uses_barometer'] and BAROMETER_ENABLED}")
    logger.info(f"  Calibration:     {'Enabled (non_bosch -> bosch)' if SENSOR_CALIBRATION_ENABLED else 'Disabled'}")
    logger.info(f"  ACC Preprocess:  {model_info['acc_preprocessing']}")
    logger.info(f"  BARO Preprocess: {model_info['baro_preprocessing'] if BAROMETER_ENABLED else 'disabled'}")
    logger.info(f"  Features:        {model_info['num_features']} ({model_info['acc_features']} ACC + {model_info['baro_features']} BARO)")
    logger.info(f"="*70)

    # If CSV mode, run validation immediately
    if DATA_SOURCE == 'csv' and CSV_FILE_PATH:
        logger.info(f"\nCSV mode detected - running validation...")
        result = process_csv_file(CSV_FILE_PATH, CSV_WINDOW_INTERVAL_SECONDS, inference_engine, MODEL_VERSION)
        if result['success']:
            logger.info(f"Validation complete. Results saved to: {result['output_file']}")
        else:
            logger.error(f"Validation failed: {result.get('error')}")
    else:
        # InfluxDB mode - start continuous monitoring if enabled
        if MONITORING_ENABLED:
            logger.info(f"\nStarting continuous monitoring...")

            shared_state.continuous_monitor = ContinuousMonitor(
                inference_engine=inference_engine,
                notification_queue=shared_state.fall_notification_queue,
                export_callback=export_detection_data,
                notification_callback=shared_state.add_poll_notification
            )

            shared_state.continuous_monitor.start()
            logger.info("Continuous monitoring started (CSV export requires recording active)")
        else:
            logger.info("\nContinuous monitoring is disabled. Use /trigger endpoint for manual detection.")

        # Start Flask app
        if PUBLIC_ENDPOINT_ENABLED:
            logger.info(f"PUBLIC ENDPOINT MODE ENABLED")
            logger.info(f"  Tunnel mode: {TUNNEL_MODE}")
            logger.info(f"  API Keys configured: {len(API_KEYS)}")
            logger.info(f"  Rate limit: {RATE_LIMIT_PER_MINUTE} req/min")
            logger.info(f"  Debug mode: {FLASK_DEBUG}")

        app.run(host="0.0.0.0", port=8000, debug=FLASK_DEBUG, use_reloader=False, threaded=True)
