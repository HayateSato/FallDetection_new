"""
Recording routes: /recording/state, /manual_truth/marker, /fall_feedback

Handles participant recording state, manual truth markers,
and user feedback on fall detections.
"""
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from flask import Blueprint, jsonify, request

from app.utils import shared_state

logger = logging.getLogger(__name__)

recording_bp = Blueprint('recording', __name__)


@recording_bp.route('/recording/state', methods=['POST'])
def update_recording_state():
    """Update recording state and participant information."""
    from app.core.recording_state import recording_state

    try:
        data = request.get_json()
        recording_active = data.get('recording_active', False)
        participant_name = data.get('participant_name', 'unknown')
        participant_gender = data.get('participant_gender', 'unknown')
        manual_truth_fall = data.get('manual_truth_fall', 0)

        recording_state.update_participant_name(participant_name)
        recording_state.update_participant_gender(participant_gender)
        recording_state.update_manual_truth(manual_truth_fall)
        recording_state.set_recording_active(recording_active)

        logger.info(f"Recording state updated: active={recording_active}, name={participant_name}")

        return jsonify({
            'message': 'Recording state updated',
            'state': recording_state.get_current_state()
        }), 200

    except Exception as e:
        logger.error(f"Error updating recording state: {e}")
        return jsonify({'error': str(e)}), 500


@recording_bp.route('/manual_truth/marker', methods=['POST'])
def write_manual_truth_marker():
    """
    Write manual truth marker directly to InfluxDB at current timestamp.
    Called when user presses the manual truth button.
    """
    from app.data_output.marker_injection import write_manual_truth_marker as write_marker

    try:
        data = request.get_json()
        value = data.get('value', 1)  # 1 = fall event, 0 = no fall

        success = write_marker(value)

        if success:
            logger.info(f"Manual truth marker written to InfluxDB: value={value}")
            return jsonify({
                'message': 'Manual truth marker written',
                'value': value,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200
        else:
            return jsonify({
                'error': 'Failed to write manual truth marker'
            }), 500

    except Exception as e:
        logger.error(f"Error writing manual truth marker: {e}")
        return jsonify({'error': str(e)}), 500


@recording_bp.route('/fall_feedback', methods=['POST'])
def record_fall_feedback():
    """
    Record user feedback when fall alert popup is shown.
    Writes user_feedback marker to InfluxDB and updates the last exported CSV.

    Expected JSON body:
        feedback_value: 0 (No/not a fall), 1 (Yes/confirmed fall), 3 (timeout/no response)
        detection_timestamp: ISO timestamp of the original detection
        confidence: detection confidence value
    """
    from app.core.recording_state import recording_state
    from app.data_output.marker_injection import write_user_feedback_marker
    from config.settings import FALL_DATA_EXPORT_DIR, TIMEZONE_OFFSET_HOURS

    try:
        data = request.get_json()
        feedback_value = data.get('feedback_value', 3)  # 0=No, 1=Yes, 3=timeout
        detection_timestamp = data.get('detection_timestamp', '')
        confidence = data.get('confidence', 0)

        participant_name = recording_state.get_current_state().get('participant_name', 'unknown')

        now_utc = datetime.now(timezone.utc)
        now_local = now_utc + timedelta(hours=TIMEZONE_OFFSET_HOURS)

        feedback_labels = {0: "NO_FALL", 1: "CONFIRMED_FALL", 3: "TIMEOUT_NO_RESPONSE"}
        feedback_type = feedback_labels.get(feedback_value, f"UNKNOWN({feedback_value})")

        # 1. Write user_feedback marker to InfluxDB
        influx_ok = write_user_feedback_marker(feedback_value)
        if influx_ok:
            logger.info(f"User feedback written to InfluxDB: {feedback_type} ({feedback_value})")
        else:
            logger.warning(f"Failed to write user feedback to InfluxDB")

        # 2. Retroactively update the last exported CSV with user_feedback value
        csv_updated = False
        csv_path = None
        with shared_state.csv_path_lock:
            csv_path = shared_state.last_exported_csv_path

        if csv_path and Path(csv_path).exists():
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for i, line in enumerate(lines):
                    if 'user_feedback' in line and i + 1 < len(lines):
                        data_line = lines[i + 1]
                        cols = line.strip().split(',')
                        vals = data_line.strip().split(',')
                        if 'user_feedback' in cols and len(vals) == len(cols):
                            fb_idx = cols.index('user_feedback')
                            vals[fb_idx] = str(feedback_value)
                            lines[i + 1] = ','.join(vals) + '\n'
                        break

                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

                csv_updated = True
                logger.info(f"CSV updated with user_feedback={feedback_value}: {csv_path}")
            except Exception as csv_err:
                logger.error(f"Failed to update CSV with feedback: {csv_err}")

        # 3. Append to text log for human readability
        export_dir = Path(FALL_DATA_EXPORT_DIR)
        export_dir.mkdir(parents=True, exist_ok=True)
        feedback_file = export_dir / "fall_feedback_log.txt"

        feedback_line = (
            f"{now_local.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{feedback_type} (value={feedback_value}) | "
            f"Participant: {participant_name} | "
            f"Confidence: {confidence} | "
            f"Detection Time: {detection_timestamp}\n"
        )

        if not feedback_file.exists():
            with open(feedback_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("Fall Detection User Feedback Log\n")
                f.write("=" * 80 + "\n")
                f.write("Format: Timestamp | Feedback Type | Participant | Confidence | Detection Time\n")
                f.write("-" * 80 + "\n")

        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(feedback_line)

        logger.info(f"Fall feedback recorded: {feedback_type} for {participant_name}")

        return jsonify({
            'message': 'Feedback recorded',
            'feedback_type': feedback_type,
            'feedback_value': feedback_value,
            'influx_written': influx_ok,
            'csv_updated': csv_updated,
            'csv_path': csv_path,
            'timestamp': now_local.isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error recording fall feedback: {e}")
        return jsonify({'error': str(e)}), 500
