"""
Feature Registry per Model Version 

Each model version uses a specific set of features derived from the raw sensor data. 
This module defines the feature sets for each model version, 


Accelerometer --- 
  _STAT_ACC   - Statistical features (min, max, mean, variance for each axis and magnitude)
  _PAPER_ACC  - Features inspired by academic fall detection papers (magnitude stats + impact events)
  _RAW_ACC    - Raw accelerometer features (min, max, mean, variance for each axis and magnitude)
Barometer ---
  _EMA_BARO   - Features from dual-path EMA barometer processing (delta height stats, slope)
  _PAPER_BARO - Features from paper-inspired barometer processing (pressure shift, slopes, variance



Supported Models:
                |   (No Barometer)   | (With Barometer)   
                |   V0    v1    v2   |  v3    v4    v5   
    ----------------------------------------------------               
    _STAT_ACC   |    X     X     -   |   X     -     -
    _PAPER_ACC  |    -     -     X   |   -     X     -
    _RAW_ACC    |    -     -     -   |   -     -     X
    -----------------------------------------------------
    _EMA_BARO   |    -     -     -   |   -     X     -
    _PAPER_BARO |    -     -     -   |   X     X     - 
    _RAW_BARO   |    -     -     -   |   -     -     X

"""

# Shared feature name lists (reused across models with identical feature sets)
_STAT_ACC = [
    "acc_x_min", "acc_x_max", "acc_x_mean", "acc_x_var",
    "acc_y_min", "acc_y_max", "acc_y_mean", "acc_y_var",
    "acc_z_min", "acc_z_max", "acc_z_mean", "acc_z_var",
    "acc_mag_min", "acc_mag_max", "acc_mag_mean", "acc_mag_var",
]
_PAPER_ACC = [
    "acc_mag_max", "acc_mag_mean", "acc_mag_var",
    "impact_count", "max_impact_g", "has_high_impact",
]
_RAW_ACC = [
    "raw_acc_x_min", "raw_acc_x_max", "raw_acc_x_mean", "raw_acc_x_var",
    "raw_acc_y_min", "raw_acc_y_max", "raw_acc_y_mean", "raw_acc_y_var",
    "raw_acc_z_min", "raw_acc_z_max", "raw_acc_z_mean", "raw_acc_z_var",
    "raw_acc_mag_min", "raw_acc_mag_max", "raw_acc_mag_mean", "raw_acc_mag_var",
]
_EMA_BARO = [
    "delta_h_min", "delta_h_max", "delta_h_mean", "delta_h_var",
    "delta_h_range", "delta_h_slope",
]
_PAPER_BARO = [
    "pressure_shift", "middle_slope", "post_fall_slope", "filtered_pressure_var",
]
_RAW_BARO = [
    "raw_pressure_min", "raw_pressure_max", "raw_pressure_mean", "raw_pressure_var",
    "raw_pressure_range", "raw_pressure_slope",
]
