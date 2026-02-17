"""
Monitoring routes: /events, /notification/config, /notifications/poll,
                   /monitoring/status, /monitoring/stop

Handles SSE streaming, polling notifications, and monitoring control.
"""
import os
import json
import queue
import logging
from flask import Blueprint, jsonify, Response

from app.utils import shared_state

from config.settings import (
    MODEL_VERSION,
    MONITORING_ENABLED,
    ACC_SAMPLE_RATE,
    BARO_SAMPLE_RATE,
    ACC_SENSOR_TYPE,
    HARDWARE_ACC_SAMPLE_RATE,
    MODEL_ACC_SAMPLE_RATE,
    BAROMETER_ENABLED,
    UPSAMPLING_ENABLED,
    DOWNSAMPLING_ENABLED,
    RESAMPLING_ENABLED,
    RESAMPLING_METHOD,
    ACC_FIELD_X,
    ACC_FIELD_Y,
    ACC_FIELD_Z,
    WINDOW_SAMPLES,
    NOTIFICATION_MODE,
    POLLING_INTERVAL_SECONDS,
)

logger = logging.getLogger(__name__)

monitoring_bp = Blueprint('monitoring', __name__)

DATA_SOURCE = os.getenv('DATA_SOURCE', 'influx').lower()
CSV_FILE_PATH = os.getenv('CSV_FILE_PATH', '')


@monitoring_bp.route('/events')
def sse_stream():
    """Server-Sent Events stream for real-time fall notifications."""
    def event_stream():
        while True:
            try:
                fall_data = shared_state.fall_notification_queue.get(timeout=30)

                event_data = json.dumps({
                    'fall_detected': fall_data.get('is_fall', False),
                    'timestamp': fall_data['timestamp'],
                    'confidence': fall_data.get('confidence', 0),
                    'model_type': MODEL_VERSION,
                    'message': f"Fall detected with {fall_data.get('confidence', 0):.0%} confidence" if fall_data.get('is_fall', False) else "No fall detected"
                })

                yield f"data: {event_data}\n\n"

            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    response = Response(event_stream(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@monitoring_bp.route('/notification/config')
def notification_config():
    """Tell the frontend which notification mode to use."""
    return jsonify({
        'mode': NOTIFICATION_MODE,
        'polling_interval_seconds': POLLING_INTERVAL_SECONDS
    })


@monitoring_bp.route('/notifications/poll')
def poll_notifications():
    """Return and clear any pending fall notifications (for polling mode)."""
    with shared_state.poll_lock:
        events = list(shared_state.poll_notifications)
        shared_state.poll_notifications.clear()

    formatted = []
    for fall_data in events:
        formatted.append({
            'fall_detected': fall_data.get('is_fall', False),
            'timestamp': fall_data.get('timestamp', ''),
            'confidence': fall_data.get('confidence', 0),
            'model_type': fall_data.get('model_version', MODEL_VERSION),
            'message': f"Fall detected with {fall_data.get('confidence', 0):.0%} confidence"
                       if fall_data.get('is_fall', False) else "No fall detected"
        })

    return jsonify({'events': formatted})


@monitoring_bp.route('/monitoring/status', methods=['GET'])
def monitoring_status():
    """Get current monitoring status."""
    from flask import current_app
    model_info = current_app.config['model_info']

    return jsonify({
        'monitoring_enabled': MONITORING_ENABLED,
        'is_running': shared_state.continuous_monitor is not None and getattr(shared_state.continuous_monitor, 'is_running', False),
        'data_source': DATA_SOURCE,
        'csv_file': CSV_FILE_PATH if DATA_SOURCE == 'csv' else None,
        'model_version': MODEL_VERSION,
        'model_name': model_info['name'],
        'uses_barometer': model_info['uses_barometer'] and BAROMETER_ENABLED,
        'acc_sample_rate': ACC_SAMPLE_RATE,
        'baro_sample_rate': BARO_SAMPLE_RATE,
        'features': model_info['num_features'],
        'acc_sensor_type': ACC_SENSOR_TYPE,
        'hardware_acc_rate': HARDWARE_ACC_SAMPLE_RATE,
        'model_acc_rate': MODEL_ACC_SAMPLE_RATE,
        'barometer_enabled': BAROMETER_ENABLED,
        'upsampling_enabled': UPSAMPLING_ENABLED,
        'downsampling_enabled': DOWNSAMPLING_ENABLED,
        'resampling_method': RESAMPLING_METHOD if RESAMPLING_ENABLED else None,
        'acc_fields': [ACC_FIELD_X, ACC_FIELD_Y, ACC_FIELD_Z],
        'window_samples': WINDOW_SAMPLES,
    })


@monitoring_bp.route('/monitoring/stop', methods=['POST'])
def stop_monitoring():
    """Stop continuous monitoring."""
    if not shared_state.continuous_monitor or not getattr(shared_state.continuous_monitor, 'is_running', False):
        return jsonify({'message': 'Monitoring not running'}), 200

    shared_state.continuous_monitor.stop()
    return jsonify({'message': 'Continuous monitoring stopped'}), 200
