"""
Model Logger Utility
Provides structured logging for the fall detection pipeline with visual formatting.
"""

import logging
from datetime import datetime
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)


class ModelLogger:
    """Logger for fall detection model pipeline with structured output"""

    def __init__(self):
        self.logger = logging.getLogger('fall_detection')

    def log_data_fetch_start(self, query_time: datetime):
        """Log start of data fetch"""
        print(f"\n[1/7] Data Fetch")
        print(f"  Query time: {query_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    def log_data_fetch_complete(self, num_tables: int, num_records: int):
        """Log completion of data fetch"""
        print(f"  Received {num_tables} tables, {num_records} records")

    def log_preprocessing_start(self):
        """Log start of preprocessing"""
        print(f"\n[2/7] Preprocessing")

    def log_preprocessing_complete(self, acc_samples: int, duration: float):
        """Log completion of preprocessing"""
        print(f"  Processed {acc_samples} accelerometer samples")
        print(f"  Duration: {duration:.2f} seconds")

    def log_dataframe_conversion_start(self):
        """Log start of DataFrame conversion"""
        print(f"\n[3/7] DataFrame Conversion")

    def log_dataframe_conversion_complete(self, df: pd.DataFrame, sampling_rate: float):
        """Log completion of DataFrame conversion"""
        print(f"  Converted to DataFrame: {len(df)} samples")
        print(f"  Sampling rate: {sampling_rate:.1f} Hz")

    def log_windowing_start(self):
        """Log start of windowing"""
        print(f"\n[4/7] Window Extraction")

    def log_windowing_complete(self, window_size: int):
        """Log completion of windowing"""
        print(f"  Extracted {window_size} samples")

    def log_feature_extraction_start(self):
        """Log start of feature extraction"""
        print(f"\n[5/7] Feature Extraction")

    def log_feature_extraction_complete(self, num_features: int, sample_features: dict):
        """Log completion of feature extraction"""
        print(f"  Extracted {num_features} features")
        print(f"  Sample features (first 5):")
        for i, (key, value) in enumerate(list(sample_features.items())[:5]):
            print(f"    {key}: {value:.6f}")

    def log_inference_start(self):
        """Log start of inference"""
        print(f"\n[6/7] Model Inference")

    def log_inference_complete(self, is_fall: bool, confidence: float, threshold: float):
        """Log completion of inference"""
        result = "FALL" if is_fall else "NO FALL"
        symbol = "!" if is_fall else "-"
        print(f"  {symbol} Prediction: {result}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Threshold: {threshold:.2%}")

    def log_data_quality_analysis(self, sampling_rate: float, num_samples: int):
        """Log data quality analysis"""
        print(f"\n[7/7] Data Quality Analysis")
        print(f"  Sampling rate: {sampling_rate:.1f} Hz")
        print(f"  Total samples: {num_samples}")

        # Quality indicators (50Hz target)
        target_rate = 50.0
        rate_deviation = abs(sampling_rate - target_rate) / target_rate * 100

        if rate_deviation < 5:
            quality = "Excellent"
        elif rate_deviation < 10:
            quality = "Good"
        elif rate_deviation < 20:
            quality = "Fair"
        else:
            quality = "Poor"

        print(f"  Data quality: {quality} ({rate_deviation:.1f}% deviation from 50Hz)")

    def log_summary(self, num_records: int, sampling_rate: float, num_samples: int,
                   is_fall: bool, confidence: float):
        """Log pipeline summary"""
        print(f"\n{'='*70}")
        print(f"DETECTION SUMMARY")
        print(f"{'='*70}")
        print(f"  Records processed: {num_records}")
        print(f"  Sampling rate: {sampling_rate:.1f} Hz")
        print(f"  Window size: {num_samples} samples")
        print(f"  Result: {'FALL DETECTED' if is_fall else 'NO FALL'}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"{'='*70}\n")

    def log_error(self, message: str, error: Exception = None):
        """Log an error"""
        print(f"\n! Error: {message}")
        if error:
            print(f"  Details: {str(error)}")
        if error:
            self.logger.error(f"{message}: {error}", exc_info=True)
        else:
            self.logger.error(message)


# Singleton instance
model_logger = ModelLogger()
