## Updated the InfluxDB field name across all files:

- `manual_truth_writer.py:43` — .field("manual_truth_marker", value)
- `continuous_monitoring.py:242` — InfluxDB query filter
- `query_manager.py:37,49` — both query variants
- `main.py:711` — trigger endpoint query filter
- `main.py:620` — CSV metadata column name


## user_feedback marker system (0/1/3)

- New function `write_user_feedback_marker()` — writes `user_feedback` field to InfluxDB
- CSV export `save_detection_window_to_csv()` — adds `user_feedback: -1 (pending)` to metadata, tracks the filepath - in `_last_exported_csv_path`
- Updated /fall_feedback endpoint — now:
  1. Writes `user_feedback` marker to InfluxDB
  2. Retroactively updates the last exported CSV (replaces -1 with actual value)
  3. Still appends to the text log for human readability
- All InfluxDB queries now also fetch the `user_feedback` field


## Updated frontend popup

- `index.html:457` — Timeout changed from **30s → 10s**
- **On timeout**: automatically sends `feedback_value=3` (unknown/no response)
- `respondToFallAlert()` — sends `feedback_value=1` (Yes) or 0 (No)
- `recordFallFeedback()` — sends numeric `feedback_value` instead of boolean is_fall
  

#### Feedback values reference:
| Value	   | Meaning                                     |
|----------|---------------------------------------------|
|   0	   | User pressed No (not a fall)                |
|   1	   | User pressed Yes (confirmed fall)           |
|   3	   | Timeout — no response within 10 seconds     |
|  -1	   | Pending — feedback not yet received (in CSV)|

