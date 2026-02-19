"""
Detection routes: /trigger, /validate-csv, /model/info

Core fall detection pipeline - fetches sensor data, preprocesses,
runs inference, and returns results.
"""
import os
import logging
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, current_app
from typing import Tuple

from app.utils.model_logger import model_logger
from app.data_input.data_loader.csv_dataloader import process_csv_file
from app.data_input.sensor_data_reader import fetch_and_preprocess_sensor_data
from app.middleware.api_security import require_api_key
from app.data_output.data_exporter import convert_to_dataframe, extract_window, export_detection_data

from config.settings import (
    MODEL_VERSION,
    WINDOW_SIZE_SECONDS,
    ACC_SAMPLE_RATE,
    ACC_SENSOR_SENSITIVITY,
)

logger = logging.getLogger(__name__)

detection_bp = Blueprint('detection', __name__)

# Data source settings
DATA_SOURCE = os.getenv('DATA_SOURCE', 'influx').lower()
CSV_FILE_PATH = os.getenv('CSV_FILE_PATH', '')
CSV_WINDOW_INTERVAL_SECONDS = float(os.getenv('CSV_WINDOW_INTERVAL_SECONDS', '1.0'))


@detection_bp.route('/trigger', methods=['POST'])
@require_api_key
def trigger() -> Tuple:
    """
    Main endpoint for fall detection.

    Authentication (when PUBLIC_ENDPOINT_ENABLED=true):
        - Header: X-API-Key: <your-api-key>
        - Or query param: ?api_key=<your-api-key>
    """
    inference_engine = current_app.config['inference_engine']
    model_info = current_app.config['model_info']
    model_config = current_app.config['model_config']

    # CSV mode
    if DATA_SOURCE == 'csv':
        if not CSV_FILE_PATH:
            return jsonify({
                "message": "CSV mode selected but CSV_FILE_PATH not set in .env",
                "error": "Configuration error"
            }), 400

        result = process_csv_file(CSV_FILE_PATH, CSV_WINDOW_INTERVAL_SECONDS, inference_engine, MODEL_VERSION)

        if result['success']:
            return jsonify({
                "message": f"CSV validation completed ({model_info['name']})",
                "data_source": "csv",
                "model_version": MODEL_VERSION,
                "model_name": model_info['name'],
                "csv_path": result['csv_path'],
                "total_windows": result['total_windows'],
                "falls_detected": result['falls_detected'],
                "output_file": result['output_file']
            }), 200
        else:
            return jsonify({
                "message": "CSV validation failed",
                "error": result.get('error', 'Unknown error')
            }), 500

    # InfluxDB mode - real-time detection
    request_data = request.get_json() if request.is_json else {}
    participant_name = request_data.get('participant_name', 'unknown')
    participant_gender = request_data.get('participant_gender', 'unknown')
    manual_truth_fall = request_data.get('manual_truth_fall', 0)

    print("="*70)
    print(f"FALL DETECTION PIPELINE - {model_info['name']}")
    print("="*70)
    print(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {model_info['description']}")
    print(f"Features: {model_info['num_features']} ({model_info['acc_features']} ACC + {model_info['baro_features']} BARO)")
    print(f"Participant: {participant_name} ({participant_gender})")
    print(f"Manual truth: {manual_truth_fall}")
    print("="*70)

    try:
        # STEP 1+2: FETCH AND PREPROCESS DATA FROM INFLUXDB
        query_start_time = datetime.now(timezone.utc)
        model_logger.log_data_fetch_start(query_start_time)

        acc_data, acc_time, pressure, pressure_time, flux_objects = fetch_and_preprocess_sensor_data(
            uses_barometer=inference_engine.uses_barometer(),
            lookback_seconds=30,
        )

        if acc_data is None:
            error_msg = "No data found in InfluxDB for this time range"
            model_logger.log_error(error_msg)
            return jsonify({
                "message": error_msg,
                "error": "No sensor data available"
            }), 404

        acc_duration = (acc_time[-1] - acc_time[0]) / 1000.0 if len(acc_time) > 1 else 0
        model_logger.log_data_fetch_complete(len(flux_objects), len(flux_objects))
        model_logger.log_preprocessing_complete(acc_samples=acc_data.shape[1], duration=acc_duration)
        print(f"  ACC samples: {acc_data.shape[1]}, duration: {acc_duration:.1f}s")
        print(f"  BARO samples: {len(pressure)}")

        # STEP 3: CONVERT TO DATAFRAME
        model_logger.log_dataframe_conversion_start()

        try:
            acc_scale_factor = 1.0 / ACC_SENSOR_SENSITIVITY
            df = convert_to_dataframe(acc_data, acc_time, acc_scale_factor)

            time_diffs = df['Device_Timestamp_[ms]'].diff().dropna()
            actual_acc_rate = 1000 / time_diffs.median()

            print(f"  ACC sampling rate: {actual_acc_rate:.1f} Hz")

            model_logger.log_dataframe_conversion_complete(df, actual_acc_rate)

        except Exception as e:
            model_logger.log_error("Error converting to DataFrame", e)
            return jsonify({
                "message": "DataFrame conversion failed",
                "error": str(e)
            }), 500

        # STEP 4: EXTRACT WINDOW
        model_logger.log_windowing_start()

        try:
            required_samples = int(WINDOW_SIZE_SECONDS * ACC_SAMPLE_RATE)

            window_df, window_pressure, window_pressure_time = extract_window(
                df, required_samples, pressure, pressure_time
            )

            baro_count = len(window_pressure) if window_pressure is not None else 0
            print(f"  Window: {len(window_df)} ACC samples, {baro_count} BARO samples")

            model_logger.log_windowing_complete(len(window_df))

        except Exception as e:
            model_logger.log_error("Error extracting window", e)
            return jsonify({
                "message": "Window extraction failed",
                "error": str(e)
            }), 500

        # STEP 5: RUN INFERENCE
        model_logger.log_inference_start()

        try:
            result = inference_engine.predict(
                window_df,
                pressure=window_pressure,
                pressure_timestamps=window_pressure_time
            )

            is_fall = result['is_fall']
            confidence = result['confidence']
            features_dict = result['features']

            model_logger.log_feature_extraction_complete(
                num_features=len(features_dict),
                sample_features=features_dict
            )

            threshold = model_config.threshold

            model_logger.log_inference_complete(is_fall, confidence, threshold)

        except Exception as e:
            model_logger.log_error("Error during prediction", e)
            return jsonify({
                "message": "Model prediction failed",
                "error": str(e)
            }), 500

        # SUMMARY
        model_logger.log_summary(
            num_records=len(flux_objects),
            sampling_rate=actual_acc_rate,
            num_samples=len(window_df),
            is_fall=is_fall,
            confidence=confidence
        )

        export_detection_data(
            flux_records=flux_objects,
            is_fall=is_fall,
            confidence=confidence,
            participant_name=participant_name,
            participant_gender=participant_gender,
            manual_truth_fall=manual_truth_fall,
            timestamp_utc=query_start_time
        )

        # Prepare result message
        if is_fall:
            if confidence > 0.75:
                result_message = "High confidence fall detection"
            elif confidence > 0.60:
                result_message = "Moderate confidence fall detection"
            else:
                result_message = "Low confidence fall detection"
        else:
            if confidence > 0.40:
                result_message = "Close to threshold - consider manual review"
            elif confidence > 0.25:
                result_message = "Borderline case"
            else:
                result_message = "Clear negative"

        return jsonify({
            "message": f"Fall detection completed ({model_info['name']}).",
            "data_source": "influx",
            "fall_detected": is_fall,
            "model_version": MODEL_VERSION,
            "model_name": model_info['name'],
            "result": result_message,
            "confidence": float(confidence),
            "threshold": float(threshold),
            "sampling_rate": float(actual_acc_rate),
            "window_size": len(window_df),
            "window_duration_seconds": WINDOW_SIZE_SECONDS,
            "num_features": model_info['num_features'],
            "acc_features": model_info['acc_features'],
            "baro_features": model_info['baro_features'],
            "baro_samples": baro_count,
            "participant_name": participant_name,
            "participant_gender": participant_gender,
            "manual_truth_marker": manual_truth_fall
        }), 200

    except ConnectionError as e:
        logger.error(f"Database connection error: {str(e)}")
        return jsonify({
            "message": "Failed to connect to InfluxDB.",
            "error": str(e)
        }), 503
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            "message": "An error occurred during fall detection.",
            "error": str(e)
        }), 500

    finally:
        logger.info("=== Fall detection process completed ===")


@detection_bp.route('/validate-csv', methods=['POST'])
def validate_csv() -> Tuple:
    """Endpoint for validating a specific CSV file."""
    inference_engine = current_app.config['inference_engine']
    model_info = current_app.config['model_info']

    request_data = request.get_json() if request.is_json else {}

    csv_path = request_data.get('csv_path', CSV_FILE_PATH)
    interval = request_data.get('interval_seconds', CSV_WINDOW_INTERVAL_SECONDS)

    if not csv_path:
        return jsonify({
            "message": "No CSV path provided",
            "error": "csv_path required in request or CSV_FILE_PATH in .env"
        }), 400

    result = process_csv_file(csv_path, interval, inference_engine, MODEL_VERSION)

    if result['success']:
        return jsonify({
            "message": "CSV validation completed",
            "model_version": MODEL_VERSION,
            "model_name": model_info['name'],
            "csv_path": result['csv_path'],
            "total_windows": result['total_windows'],
            "falls_detected": result['falls_detected'],
            "output_file": result['output_file'],
            "fall_windows": [r for r in result['results'] if r['is_fall']]
        }), 200
    else:
        return jsonify({
            "message": "CSV validation failed",
            "error": result.get('error', 'Unknown error')
        }), 500


@detection_bp.route('/model/info', methods=['GET'])
def get_model_info_endpoint():
    """Get information about the currently loaded model."""
    inference_engine = current_app.config['inference_engine']
    return jsonify(inference_engine.get_model_info())
