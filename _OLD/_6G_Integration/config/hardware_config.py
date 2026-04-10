"""
Hardware Data Profiles for Fall Detection System.

A data profile bundles all hardware-specific settings for a known
sensor/recording setup so you only need to set DATA_PROFILE in .env
instead of configuring ACC_SENSOR_TYPE, HARDWARE_ACC_SAMPLE_RATE, etc.
individually.

Usage in .env:
    DATA_PROFILE=SmarKo          # use a preset profile
    DATA_PROFILE=Custom          # all settings come from .env as before

Available profiles
------------------
FALL_TEST     - Bosch 50Hz + barometer 25Hz. No resampling.
SmarKo        - Bosch or non_bosch 25Hz + barometer 25Hz. Upsample to 50Hz.
                Set ACC_SENSOR_TYPE=bosch|non_bosch in .env.
                Set RESAMPLING_METHOD in .env (default: linear).
AIDAPT-Trial  - Bosch 25Hz (upsample) OR non_bosch 100Hz (downsample) + barometer 25Hz.
                Set ACC_SENSOR_TYPE=bosch|non_bosch in .env to select the sensor.
                Set RESAMPLING_METHOD in .env (default: linear / decimate).
Custom        - Every setting must be provided explicitly in .env.
"""

from dataclasses import dataclass
from typing import Dict, Optional


ACC_SENSOR_SENSITIVITY = 16384  # LSB/g for ±2g range (common for many IMUs, but can be overridden in .env)
# ACC_SENSOR_SENSITIVITY = 4096  ±8g range

@dataclass(frozen=True)
class HardwareProfile:
    """
    Hardware configuration for one data-collection setup.

    A field set to None means the profile does not constrain that setting —
    the value must come from .env (or its default).
    """
    description: str

    # 'bosch' or 'non_bosch'. None = user must set ACC_SENSOR_TYPE in .env.
    acc_sensor_type: Optional[str]

    # Hardware accelerometer sample rate in Hz.
    # None = derive from sensor_rate_map (if provided) or read from .env.
    hardware_acc_sample_rate: Optional[int]

    # Barometer sample rate in Hz.
    baro_sample_rate: int = 25

    # Used when the sample rate differs per sensor type (e.g. AIDAPT-Trial).
    # Maps sensor type string -> hardware rate.  Ignored when
    # hardware_acc_sample_rate is already set.
    sensor_rate_map: Optional[Dict[str, int]] = None

    # Accelerometer sensitivity in LSB/g (e.g. 16384 for ±2g, 4096 for ±8g).
    # Used to convert raw ADC integers to g units.
    sensor_sensitivity: int = ACC_SENSOR_SENSITIVITY

    # Default resampling method for this profile.  Can always be overridden
    # by RESAMPLING_METHOD in .env.  None = no profile-level default.
    default_resampling_method: Optional[str] = None


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

HARDWARE_PROFILES: Dict[str, HardwareProfile] = {

    # ------------------------------------------------------------------
    # FALL_TEST
    # Bosch IMU sampled at 50 Hz — exactly what the model was trained on.
    # No resampling is required.
    # ------------------------------------------------------------------
    'FALL_TEST': HardwareProfile(
        description="Lab fall-test rig: Bosch IMU at 50 Hz, barometer at 25 Hz. No resampling needed.",
        acc_sensor_type='bosch',
        hardware_acc_sample_rate=50,
        baro_sample_rate=25,
        sensor_sensitivity=ACC_SENSOR_SENSITIVITY,  # LSB/g for ±8g range
    ),

    # ------------------------------------------------------------------
    # SmarKo
    # Both the Bosch and non_bosch sensors run at 25 Hz on the SmarKo watch.
    # Select which sensor to use with ACC_SENSOR_TYPE in .env.
    # Upsampling (25 Hz → 50 Hz) is always required; set RESAMPLING_METHOD
    # in .env to choose the interpolation strategy (default: linear).
    # ------------------------------------------------------------------
    'SmarKo': HardwareProfile(
        description="SmarKo watch: Bosch or non_bosch at 25 Hz, barometer at 25 Hz. Upsamples to 50 Hz.",
        acc_sensor_type=None,           # set ACC_SENSOR_TYPE=bosch|non_bosch in .env
        hardware_acc_sample_rate=25,    # both sensors are 25 Hz on this device
        baro_sample_rate=25,
        default_resampling_method='linear',
        sensor_sensitivity=ACC_SENSOR_SENSITIVITY,  # LSB/g for ±8g range
    ),

    # ------------------------------------------------------------------
    # AIDAPT-Trial
    # Bosch runs at 25 Hz (needs upsampling) while the non_bosch sensor
    # runs at 100 Hz (needs downsampling).  Which sensor is active is
    # selected with ACC_SENSOR_TYPE in .env.
    # Set RESAMPLING_METHOD in .env (suggested: 'linear' for bosch,
    # 'decimate' or 'average' for non_bosch).
    # ------------------------------------------------------------------
    'AIDAPT-Trial': HardwareProfile(
        description=(
            "AIDAPT clinical trial: Bosch at 25 Hz (upsample) "
            "or non_bosch at 100 Hz (downsample), barometer at 25 Hz. "
            "Set ACC_SENSOR_TYPE and RESAMPLING_METHOD in .env."
        ),
        acc_sensor_type=None,           # set ACC_SENSOR_TYPE=bosch|non_bosch in .env
        hardware_acc_sample_rate=None,  # derived from sensor_rate_map below
        sensor_sensitivity=ACC_SENSOR_SENSITIVITY,  # LSB/g for ±8g range
        baro_sample_rate=25,
        sensor_rate_map={
            'bosch':     25,   # → upsampling required
            'non_bosch': 100,  # → downsampling required
        },
    ),

    # ------------------------------------------------------------------
    # Custom
    # All hardware settings are read from .env, exactly as before.
    # Use this if none of the presets match your setup.
    # ------------------------------------------------------------------
    'Custom': HardwareProfile(
        description="Custom setup: all hardware settings read individually from .env.",
        acc_sensor_type=None,
        hardware_acc_sample_rate=None,
        baro_sample_rate=25,
        sensor_sensitivity=ACC_SENSOR_SENSITIVITY,  # LSB/g for ±8g range
    ),
}


def resolve_profile(profile_name: str, env_sensor_type: str) -> HardwareProfile:
    """
    Look up a profile and, for profiles where the sample rate depends on the
    active sensor (AIDAPT-Trial), return a fully-resolved copy with
    hardware_acc_sample_rate filled in.

    Args:
        profile_name:    Value of DATA_PROFILE from .env.
        env_sensor_type: Value of ACC_SENSOR_TYPE from .env (used for
                         sensor_rate_map look-up).

    Returns:
        HardwareProfile with hardware_acc_sample_rate set (or None if Custom).

    Raises:
        ValueError: Unknown profile name or missing sensor mapping.
    """
    name = profile_name.strip()
    if name not in HARDWARE_PROFILES:
        valid = list(HARDWARE_PROFILES.keys())
        raise ValueError(
            f"Unknown DATA_PROFILE '{name}'. Valid options: {valid}"
        )

    profile = HARDWARE_PROFILES[name]

    # If the profile uses a per-sensor rate map, resolve the rate now.
    if profile.sensor_rate_map is not None and profile.hardware_acc_sample_rate is None:
        if env_sensor_type not in profile.sensor_rate_map:
            valid_sensors = list(profile.sensor_rate_map.keys())
            raise ValueError(
                f"DATA_PROFILE='{name}' requires ACC_SENSOR_TYPE to be one of "
                f"{valid_sensors}, got '{env_sensor_type}'."
            )
        # Return a new (frozen) instance with the rate filled in.
        return HardwareProfile(
            description=profile.description,
            acc_sensor_type=profile.acc_sensor_type,
            hardware_acc_sample_rate=profile.sensor_rate_map[env_sensor_type],
            baro_sample_rate=profile.baro_sample_rate,
            sensor_rate_map=profile.sensor_rate_map,
            default_resampling_method=profile.default_resampling_method,
        )

    return profile
