"""
Run the multi-point demo over selected groups of model parameters.

Edit SWEEP_GROUPS below. Values are combined only inside one group, and groups
are appended to the same CSV. This avoids accidentally testing everything with
everything.
"""

from itertools import product
import csv
import json
from pathlib import Path
import time

import numpy as np

from run_demo import ON_MOCK_DATA_DIR, make_default_config, run_experiment


# Default values for every parameter that can appear in SWEEP_GROUPS.
# Each case starts with these values, then applies group "base", then one
# generated "sweep" combination.
SWEEP_DEFAULTS = {
    "MEASUREMENT_DURATION_SECONDS": 1.0,  # Duration of one measurement; rates below are multiplied by this.
    "BACKGROUND_MEAN_PER_SECOND": 1000.0,  # Mean background counts per channel per second.
    "BACKGROUND_CHANNEL_STD_PER_SECOND": 100.0,  # Channel-to-channel background std per second.
    "CHANNEL_SCALE_MEAN": 1.0,  # Mean multiplicative channel sensitivity used during data generation.
    "CHANNEL_SCALE_STD": 0.08,  # Channel-to-channel sensitivity mismatch std used during data generation.
    "GENERATION_DISTANCE_POWER": 0.0,  # Extra empirical distance falloff; keep 0 with solid angle, test 0-1 if calibration needs more falloff.
    "GENERATION_ANGULAR_PEAK_DEG": 0.0,  # Peak of detector angular acceptance curve [deg]; typical test range: 0-45.
    "GENERATION_ANGULAR_WIDTH_DEG": 35.0,  # Intrinsic angular width before collimator [deg]; broad detector 30-80, narrow crystal response 15-35.
    "GENERATION_COLLIMATOR_HALF_ANGLE_DEG": 18.0,  # Collimator cone half-angle [deg]; narrow probe 8-20, broad/weak collimation 20-35.
    "GENERATION_COLLIMATOR_EDGE_WIDTH_DEG": 3.0,  # Softness of collimator cutoff [deg]; sharp lead collimator 1-4, soft/uncertain 4-8.
    "GENERATION_COLLIMATOR_LEAKAGE": 0.002,  # Side-response floor from imperfect shielding/scatter; ideal 0, realistic test 0.001-0.02.
    "GENERATION_SKIN_THICKNESS_CM": 0.2,  # Skin slab thickness [cm]; typical adult skin: 0.1-0.3, stress-test up to 0.5.
    "GENERATION_SKIN_MU_PER_CM": 0.15,  # Gamma attenuation coefficient for skin/soft tissue [1/cm]; rough low-energy range: 0.05-0.30.
    "GENERATION_DETECTOR_INTRINSIC_EFFICIENCY": 1.0,  # Detection probability scale after entering active material; physical range: 0-1.
    "GENERATION_RESPONSE_SCALE": 1.0,  # Calibration scale from source units to counts; tune over orders of magnitude, e.g. 1-1e5.
    "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,  # Background rate used by reconstruction model.
    "ASSUMED_CHANNEL_SCALE": 1.0,  # Global channel sensitivity used by reconstruction model.
    "ASSUMED_DISTANCE_POWER": 0.0,  # Extra empirical distance falloff; keep 0 with solid angle, test 0-1 if calibration needs more falloff.
    "ASSUMED_ANGULAR_PEAK_DEG": 0.0,  # Peak of detector angular acceptance curve [deg]; typical test range: 0-45.
    "ASSUMED_ANGULAR_WIDTH_DEG": 35.0,  # Intrinsic angular width before collimator [deg]; broad detector 30-80, narrow crystal response 15-35.
    "ASSUMED_COLLIMATOR_HALF_ANGLE_DEG": 18.0,  # Collimator cone half-angle assumed by reconstruction [deg]; try 10-25.
    "ASSUMED_COLLIMATOR_EDGE_WIDTH_DEG": 3.0,  # Assumed cutoff softness [deg]; smaller = stronger angular rejection.
    "ASSUMED_COLLIMATOR_LEAKAGE": 0.002,  # Assumed side-response floor; increase only if measured side lobes are visible.
    "ASSUMED_SKIN_THICKNESS_CM": 0.2,  # Skin thickness assumed by reconstruction [cm]; typical adult skin: 0.1-0.3.
    "ASSUMED_SKIN_MU_PER_CM": 0.15,  # Skin attenuation coefficient assumed by reconstruction [1/cm]; rough low-energy range: 0.05-0.30.
    "ASSUMED_DETECTOR_INTRINSIC_EFFICIENCY": 1.0,  # Detection probability scale assumed by reconstruction; physical range: 0-1.
    "ASSUMED_RESPONSE_SCALE": 1.0,  # Calibration scale assumed by reconstruction; tune to match measured source excess, e.g. 1-1e5.
    "ANGULAR_JITTER_DEG": 10.0,  # Random orientation noise per measurement; simulates imperfect manual sensor motion.
    "FIXED_SWEEP_Z_RANGE_DEG": 0.0,  # random left/right angle range around world Z axis, sampled as +/- this value.
    "FIXED_SWEEP_X_RANGE_DEG": 0.0,  # random up/down angle range around world X axis, sampled as +/- this value.
}


# Add groups here to test more cases. Lists are combined only inside a group.
# Use "base" to pin one parameter while sweeping a smaller set of another.
# Use "cases" for paired values, e.g. background and assumed background.
SWEEP_GROUPS = [
    # {
    #     "name": "ratio_table_best_angles",
    #     "base": {
    #         "ANGULAR_JITTER_DEG": 20.0,  # Random orientation noise per measurement; simulates imperfect manual sensor motion.
    #         "FIXED_SWEEP_Z_RANGE_DEG": 60.0,  # Best tested left/right random angle range around world Z axis.
    #         "FIXED_SWEEP_X_RANGE_DEG": 60.0,  # Best tested up/down random angle range around world X axis.
    #         "CHANNEL_SCALE_MEAN": 1.0,  # Generated global channel sensitivity.
    #         "ASSUMED_CHANNEL_SCALE": 1.0,  # Matched global channel sensitivity used by reconstruction.
    #     },
    #     "cases": [
    #         {
    #             "SEED": [11, 12, 13, 14, 15],
    #             "BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #             "BACKGROUND_CHANNEL_STD_PER_SECOND": 100.0,
    #             "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #             "TRUE_SOURCE_SPECS": ({"position": (18.0, -20.0, 20.0), "strength_per_second": 5000.0},),
    #         },  # background/source = 0.2
    #         {
    #             "SEED": [11, 12, 13, 14, 15],
    #             "BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #             "BACKGROUND_CHANNEL_STD_PER_SECOND": 100.0,
    #             "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #             "TRUE_SOURCE_SPECS": ({"position": (18.0, -20.0, 20.0), "strength_per_second": 2000.0},),
    #         },  # background/source = 0.5
    #         {
    #             "SEED": [11, 12, 13, 14, 15],
    #             "BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #             "BACKGROUND_CHANNEL_STD_PER_SECOND": 100.0,
    #             "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #             "TRUE_SOURCE_SPECS": ({"position": (18.0, -20.0, 20.0), "strength_per_second": 1000.0},),
    #         },  # background/source = 1.0
    #         {
    #             "SEED": [11, 12, 13, 14, 15],
    #             "BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #             "BACKGROUND_CHANNEL_STD_PER_SECOND": 100.0,
    #             "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #             "TRUE_SOURCE_SPECS": ({"position": (18.0, -20.0, 20.0), "strength_per_second": 500.0},),
    #         },  # background/source = 2.0
    #         {
    #             "SEED": [11, 12, 13, 14, 15],
    #             "BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #             "BACKGROUND_CHANNEL_STD_PER_SECOND": 100.0,
    #             "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #             "TRUE_SOURCE_SPECS": ({"position": (18.0, -20.0, 20.0), "strength_per_second": 200.0},),
    #         },  # background/source = 5.0
    #     ],
    # },
    {
        "name": "assumed_background_mismatch_best_case",
        "base": {
            "ANGULAR_JITTER_DEG": 10.0,  # Random orientation noise per measurement; simulates imperfect manual sensor motion.
            "FIXED_SWEEP_Z_RANGE_DEG": 60.0,  # Best tested left/right random angle range around world Z axis.
            "FIXED_SWEEP_X_RANGE_DEG": 60.0,  # Best tested up/down random angle range around world X axis.
            "BACKGROUND_MEAN_PER_SECOND": 1000.0,  # True background level.
            "BACKGROUND_CHANNEL_STD_PER_SECOND": 0.0,  # Channel-to-channel background std per second.
            "GENERATION_DISTANCE_POWER": 0.0,  # Extra generated distance falloff beyond solid angle.
            "ASSUMED_DISTANCE_POWER": 0.0,  # Matched extra distance falloff used by reconstruction.
            "CHANNEL_SCALE_MEAN": 1.0,  # Generated global channel sensitivity.
            "ASSUMED_CHANNEL_SCALE": 1.0,  # Matched global channel sensitivity used by reconstruction.
        },
        "sweep": {
            "SEED": [11, 12, 13, 14, 15],
            "ASSUMED_BACKGROUND_MEAN_PER_SECOND": [500.0, 700.0, 900.0, 1000.0, 1100.0, 1300.0, 1500.0, 2000.0],
        },
    },
    # {
    #     "name": "background_1000",
    #     "base": {
    #         "BACKGROUND_MEAN_PER_SECOND": 1000.0,  # Mean background counts per channel per second.
    #         "BACKGROUND_CHANNEL_STD_PER_SECOND": 100.0,  # Channel-to-channel background std per second.
    #     },
    #     "sweep": {
    #         "ASSUMED_BACKGROUND_MEAN_PER_SECOND": [700.0, 1000.0, 3000.0],  # Background rate used by reconstruction model.
    #     },
    # },
    # {
    #     "name": "background_5000",
    #     "base": {
    #         "BACKGROUND_MEAN_PER_SECOND": 5000.0,  # Mean background counts per channel per second.
    #         "BACKGROUND_CHANNEL_STD_PER_SECOND": 300.0,  # Channel-to-channel background std per second.
    #     },
    #     "sweep": {
    #         "ASSUMED_BACKGROUND_MEAN_PER_SECOND": [3000.0, 5000.0, 8000.0],  # Background rate used by reconstruction model.
    #     },
    # },
    # {
    #     "name": "background_20000",
    #     "base": {
    #         "BACKGROUND_MEAN_PER_SECOND": 20000.0,  # Mean background counts per channel per second.
    #         "BACKGROUND_CHANNEL_STD_PER_SECOND": 1000.0  # Channel-to-channel background std per second.
    #     },
    #     "sweep": {
    #         "ASSUMED_BACKGROUND_MEAN_PER_SECOND": [15000.0, 20000.0, 25000.0],  # Background rate used by reconstruction model.
    #     },
    # },
    # {
    #     "name": "measurement_duration",
    #     "base": {},
    #     "sweep": {
    #         "MEASUREMENT_DURATION_SECONDS": [1.0, 5.0, 10.0, 30.0],  # Duration of one measurement; rates below are multiplied by this.
    #     },
    # },
    # {
    #     "name": "source_signal",
    #     "base": {},
    #     "sweep": {
    #         "TRUE_SOURCE_SPECS": [
    #             ({"position": (18.0, -20.0, 20.0), "strength_per_second": value},)
    #             for value in [500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0]
    #         ],
    #     },
    # },
    # {
    #     "name": "channel_model",
    #     "base": {},
    #     "sweep": {
    #         "CHANNEL_SCALE_MEAN": [0.8, 1.0],  # Mean multiplicative channel sensitivity used during data generation.
    #         "CHANNEL_SCALE_STD": [0.08, 0.1],  # Channel-to-channel sensitivity mismatch std used during data generation.
    #         "ASSUMED_CHANNEL_SCALE": [0.8, 0.9, 1.0, 1.1, 1.2, 1.5],  # Global channel sensitivity used by reconstruction model.
    #     },
    # },
    # {
    #     "name": "motion_uncertainty",
    #     "base": {},
    #     "sweep": {
    #         "ANGULAR_JITTER_DEG": [0.0, 2.5, 5.0, 10.0, 20.0],  # Random orientation noise per measurement; simulates imperfect manual sensor motion.
    #         "FIXED_SWEEP_Z_RANGE_DEG": [0.0, 10.0, 25.0, 45.0, 60.0],  # random left/right angle range around world Z axis, sampled as +/- this value.
    #         "FIXED_SWEEP_X_RANGE_DEG": [0.0, 10.0, 25.0, 45.0, 60.0],  # random up/down angle range around world X axis, sampled as +/- this value.
    #     },
    # },
    # {
    #     "name": "background_1000_source_strenght_500_different_angles",
    #     "base": {
    #         "ANGULAR_JITTER_DEG": 10.0,
    #         "BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #         "BACKGROUND_CHANNEL_STD_PER_SECOND": 100.0,
    #         "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #         "TRUE_SOURCE_SPECS": ({"position": (18.0, -20.0, 20.0), "strength_per_second": 500.0},),
    #     },
    #     "sweep": {
    #         "FIXED_SWEEP_Z_RANGE_DEG": [45.0, 60.0],  # random left/right angle range around world Z axis, sampled as +/- this value.
    #         "FIXED_SWEEP_X_RANGE_DEG": [25.0, 45.0, 60.0],  # random up/down angle range around world X axis, sampled as +/- this value.
    #     },
    # },
    # {
    #     "name": "background_1000_source_strenght_1000_different_angles",
    #     "base": {
    #         "ANGULAR_JITTER_DEG": 10.0,
    #         "BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #         "BACKGROUND_CHANNEL_STD_PER_SECOND": 100.0,
    #         "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #         "TRUE_SOURCE_SPECS": ({"position": (18.0, -20.0, 20.0), "strength_per_second": 1000.0},),
    #     },
    #     "sweep": {
    #         "FIXED_SWEEP_Z_RANGE_DEG": [45.0, 60.0],  # random left/right angle range around world Z axis, sampled as +/- this value.
    #         "FIXED_SWEEP_X_RANGE_DEG": [25.0, 45.0, 60.0],  # random up/down angle range around world X axis, sampled as +/- this value.
    #     },
    # },
    # {
    #     "name": "background_1000_source_strenght_2000_different_angles",
    #     "base": {
    #         "ANGULAR_JITTER_DEG": 10.0,
    #         "BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #         "BACKGROUND_CHANNEL_STD_PER_SECOND": 100.0,
    #         "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #         "TRUE_SOURCE_SPECS": ({"position": (18.0, -20.0, 20.0), "strength_per_second": 2000.0},),
    #     },
    #     "sweep": {
    #         "FIXED_SWEEP_Z_RANGE_DEG": [45.0, 60.0],  # random left/right angle range around world Z axis, sampled as +/- this value.
    #         "FIXED_SWEEP_X_RANGE_DEG": [25.0, 45.0, 60.0],  # random up/down angle range around world X axis, sampled as +/- this value.
    #     },
    # },
    # {
    #     "name": "background_1000_source_strenght_5000_different_angles",
    #     "base": {
    #         "ANGULAR_JITTER_DEG": 10.0,
    #         "BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #         "BACKGROUND_CHANNEL_STD_PER_SECOND": 100.0,
    #         "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,
    #         "TRUE_SOURCE_SPECS": ({"position": (18.0, -20.0, 20.0), "strength_per_second": 5000.0},),
    #     },
    #     "sweep": {
    #         "FIXED_SWEEP_Z_RANGE_DEG": [45.0, 60.0],  # random left/right angle range around world Z axis, sampled as +/- this value.
    #         "FIXED_SWEEP_X_RANGE_DEG": [25.0, 45.0, 60.0],  # random up/down angle range around world X axis, sampled as +/- this value.
    #     },
    # },
    # {
    #     "name": "background_5000_source_strenght_1000_different_angles",
    #     "base": {
    #         "ANGULAR_JITTER_DEG": 10.0,
    #         "BACKGROUND_MEAN_PER_SECOND": 5000.0,
    #         "BACKGROUND_CHANNEL_STD_PER_SECOND": 300.0,
    #         "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 5000.0,
    #         "TRUE_SOURCE_SPECS": ({"position": (18.0, -20.0, 20.0), "strength_per_second": 1000.0},),
    #     },
    #     "sweep": {
    #         "FIXED_SWEEP_Z_RANGE_DEG": [45.0, 60.0],  # random left/right angle range around world Z axis, sampled as +/- this value.
    #         "FIXED_SWEEP_X_RANGE_DEG": [25.0, 45.0, 60.0],  # random up/down angle range around world X axis, sampled as +/- this value.
    #     },
    # },
    # {
    #     "name": "background_5000_source_strenght_2500_different_angles",
    #     "base": {
    #         "ANGULAR_JITTER_DEG": 10.0,
    #         "BACKGROUND_MEAN_PER_SECOND": 5000.0,
    #         "BACKGROUND_CHANNEL_STD_PER_SECOND": 300.0,
    #         "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 5000.0,
    #         "TRUE_SOURCE_SPECS": ({"position": (18.0, -20.0, 20.0), "strength_per_second": 2500.0},),
    #     },
    #     "sweep": {
    #         "FIXED_SWEEP_Z_RANGE_DEG": [45.0, 60.0],  # random left/right angle range around world Z axis, sampled as +/- this value.
    #         "FIXED_SWEEP_X_RANGE_DEG": [25.0, 45.0, 60.0],  # random up/down angle range around world X axis, sampled as +/- this value.
    #     },
    # },
    # {
    #     "name": "background_5000_source_strenght_5000_different_angles",
    #     "base": {
    #         "ANGULAR_JITTER_DEG": 10.0,
    #         "BACKGROUND_MEAN_PER_SECOND": 5000.0,
    #         "BACKGROUND_CHANNEL_STD_PER_SECOND": 300.0,
    #         "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 5000.0,
    #         "TRUE_SOURCE_SPECS": ({"position": (18.0, -20.0, 20.0), "strength_per_second": 5000.0},),
    #     },
    #     "sweep": {
    #         "FIXED_SWEEP_Z_RANGE_DEG": [45.0, 60.0],  # random left/right angle range around world Z axis, sampled as +/- this value.
    #         "FIXED_SWEEP_X_RANGE_DEG": [25.0, 45.0, 60.0],  # random up/down angle range around world X axis, sampled as +/- this value.
    #     },
    # },
    # {
    #     "name": "background_5000_source_strenght_10000_different_angles",
    #     "base": {
    #         "ANGULAR_JITTER_DEG": 10.0,
    #         "BACKGROUND_MEAN_PER_SECOND": 5000.0,
    #         "BACKGROUND_CHANNEL_STD_PER_SECOND": 300.0,
    #         "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 5000.0,
    #         "TRUE_SOURCE_SPECS": ({"position": (18.0, -20.0, 20.0), "strength_per_second": 10000.0},),
    #     },
    #     "sweep": {
    #         "FIXED_SWEEP_Z_RANGE_DEG": [45.0, 60.0],  # random left/right angle range around world Z axis, sampled as +/- this value.
    #         "FIXED_SWEEP_X_RANGE_DEG": [25.0, 45.0, 60.0],  # random up/down angle range around world X axis, sampled as +/- this value.
    #     },
    # },
]


# All non-swept run_demo settings live here so every run is explicit and every
# setting can be written to the output CSV.
FIXED_OVERRIDES = {
    "reconstruction_method": "mlem",  # Available: "mlem", "map_em".
    "SEED": 11,  # Random seed for mock data noise, channel scales, backgrounds, and trajectory jitter.
    "NUM_MEASUREMENTS": 50,  # Number of detector positions / generated measurements.
    "TRUE_SOURCE_SPECS": [  # Ground-truth point source positions [cm] and source strengths [/s].
        {"position": (18.0, -20.0, 20.0), "strength_per_second": 5000.0},
    ],
    "NUM_CHANNELS": 16,  # Number of detector channels.
    "DIODE_PITCH": 0.15875,  # Channel width and center-to-center pitch [cm]; 16 channels span 2.54 cm.
    "SEARCH_RADIUS": 90.0,  # Reconstruction cylinder radius around TRAJECTORY_CENTER [cm].
    "CYLINDER_HEIGHT": 36.0,  # Reconstruction cylinder height along world Z [cm].
    "VOXEL_SIZE_CM": 8.0,  # Reconstruction voxel spacing [cm]; smaller = denser 3D activity grid.
    "TRAJECTORY_MODE": "grid",  # "grid" or "around_center".
    "LATERAL_SPAN": 60.0,  # grid only: max left/right position offset from TRAJECTORY_CENTER [cm].
    "VERTICAL_SPAN": 36.0,  # grid only: max up/down position offset from TRAJECTORY_CENTER [cm].
    "BASE_DISTANCE": 100.0,  # grid only: nominal detector distance from TRAJECTORY_CENTER [cm].
    "DISTANCE_SPAN": 18.0,  # grid only: sinusoidal variation of detector distance [cm].
    "ORBIT_RADIUS": 130.0,  # around_center only: nominal radius of detector orbit around TRAJECTORY_CENTER [cm].
    "ORBIT_RADIUS_SPAN": 0,  # around_center only: sinusoidal radius variation during orbit [cm].
    "ORBIT_ANGLE_SPAN_DEG": 360.0,  # around_center only: covered orbit angle; 360 means full circle.
    "ORBIT_START_ANGLE_DEG": 0.0,  # around_center only: starting angle of orbit in XY plane.
    "ORBIT_VERTICAL_SPAN": 0.0,  # around_center only: sinusoidal up/down motion during orbit [cm].
    "ORIENTATION_MODE": "fixed",  # "fixed" or "towards_target"; use "towards_target" for around_center.
    "FIXED_VIEW_DIRECTION": (0, 1, 0),  # e.g. (1, 0, 0) to "look right" in world coords.
    "TRAJECTORY_CENTER": (0.0, 0.0, 0.0),  # Reference center of the detector trajectory [cm].
    "NUM_ITERATIONS": 10000,  # ML-EM iteration count. Set back to 10000 if you want to match run_demo exactly.
    "HISTORY_INTERVAL": 1000,  # How often ML-EM stores history snapshots.
    "PRINT_GENERATED_COUNTS_TABLE": False,  # Sweep runs skip the long generated-count table.
    "UPDATE_FRAME_MAX_VOXELS": 1000,  # Max strongest voxels plotted in one history frame.
    "GIF_FPS": 1.0,  # GIF frame rate if visualizations are enabled elsewhere.
    "AXIS_LIMITS_3D": (  # 3D plot axis limits if visualizations are enabled elsewhere.
        (-100, 100),
        (-100, 100),
        (-100, 100),
    ),
}


OUTPUT_CSV = ON_MOCK_DATA_DIR / "multiPoint" / "visualizations" / "ass_background_mismatch.csv"


def sweep_parameter_columns():
    """Return all parameters that appear in any sweep group."""
    columns = list(SWEEP_DEFAULTS)
    for group in SWEEP_GROUPS:
        for key in [*group.get("base", {}), *group.get("sweep", {})]:
            if key not in columns:
                columns.append(key)
        for case in group.get("cases", []):
            for key in case:
                if key not in columns:
                    columns.append(key)
    return columns


def iter_parameter_cases(sweep_groups):
    """Yield one (group_name, override dict, display dict) for every configured case."""
    for group in sweep_groups:
        group_name = group["name"]
        base_overrides = {**SWEEP_DEFAULTS, **dict(group.get("base", {}))}
        explicit_cases = list(group.get("cases", []))
        if explicit_cases:
            for case_overrides in explicit_cases:
                keys = list(case_overrides)
                value_lists = [
                    value if isinstance(value, list) else [value]
                    for value in case_overrides.values()
                ]
                for values in product(*value_lists):
                    expanded_case = dict(zip(keys, values))
                    display_parameters = {**dict(group.get("base", {})), **expanded_case}
                    yield group_name, {**base_overrides, **expanded_case}, display_parameters
            continue

        sweep_values = dict(group.get("sweep", {}))
        keys = list(sweep_values)

        if not keys:
            yield group_name, base_overrides, dict(group.get("base", {}))
            continue

        value_lists = [sweep_values[key] for key in keys]
        for values in product(*value_lists):
            sweep_overrides = dict(zip(keys, values))
            yield group_name, {**base_overrides, **sweep_overrides}, sweep_overrides


def _csv_value(value):
    """Convert nested settings to compact CSV-safe values."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


def format_fixed_overrides():
    """Format fixed settings as one compact name=value CSV field."""
    parts = []
    for key, value in FIXED_OVERRIDES.items():
        parts.append(f"{key}={_csv_value(value)}")
    return ", ".join(parts)


def format_case_parameters(parameter_overrides):
    """Format the parameters used by one sweep case for terminal progress."""
    parts = []
    for key, value in parameter_overrides.items():
        parts.append(f"{key}={_csv_value(value)}")
    return ", ".join(parts)


def summarize_run(case_index, group_name, parameter_overrides, experiment, elapsed_seconds):
    """Flatten one run into a CSV row."""
    config = experiment["config"]
    similarity = experiment["activity_map_similarity"]

    total_source_strength_per_second = sum(
        float(source["strength_per_second"])
        for source in config["TRUE_SOURCE_SPECS"]
    )

    row = {
        "case_index": int(case_index),
        "sweep_group": group_name,
        "elapsed_seconds": float(elapsed_seconds),
        "background_source_ratio": float(
            config["BACKGROUND_MEAN_PER_SECOND"]
            / total_source_strength_per_second
        ),
        "true_total_activity": float(similarity["true_total_activity"]),
        "reconstructed_total_activity": float(similarity["reconstructed_total_activity"]),
        "scale_to_true_total": float(similarity["scale_to_true_total"]),
        "cosine_similarity": float(similarity["cosine_similarity"]),
        "pearson_correlation": float(similarity["pearson_correlation"]),
        "active_mask_overlap": float(similarity["active_mask_overlap"]),
        "active_mask_recall": float(similarity["active_voxel_recall"]),
        "active_mask_precision": float(similarity["active_voxel_precision"]),
        "true_active_voxel_count": int(similarity["true_active_voxel_count"]),
        "reconstructed_active_voxel_count": int(similarity["reconstructed_active_voxel_count"]),
        "activity_threshold_fraction": float(similarity["threshold_fraction"]),
    }

    row["fixed_overrides"] = format_fixed_overrides()
    for key in sweep_parameter_columns():
        row[key] = _csv_value(config[key])

    return row


def write_rows(path, rows):
    """Write rows to CSV with stable column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        "case_index",
        "sweep_group",
        "fixed_overrides",
        *sweep_parameter_columns(),
        "background_source_ratio",
        "elapsed_seconds",
        "true_total_activity",
        "reconstructed_total_activity",
        "scale_to_true_total",
        "cosine_similarity",
        "pearson_correlation",
        "active_mask_overlap",
        "active_mask_recall",
        "active_mask_precision",
        "true_active_voxel_count",
        "reconstructed_active_voxel_count",
        "activity_threshold_fraction",
    ]

    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=metric_columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    default_config = make_default_config()
    rows = []
    cases = list(iter_parameter_cases(SWEEP_GROUPS))

    print(f"Running {len(cases)} parameter-sweep cases")
    print(f"Output CSV: {OUTPUT_CSV}")

    for case_index, (group_name, parameter_overrides, display_parameters) in enumerate(cases, start=1):
        overrides = {**FIXED_OVERRIDES, **parameter_overrides}
        started_at = time.perf_counter()
        experiment = run_experiment(overrides=overrides)
        elapsed_seconds = time.perf_counter() - started_at

        row = summarize_run(
            case_index=case_index,
            group_name=group_name,
            parameter_overrides=parameter_overrides,
            experiment=experiment,
            elapsed_seconds=elapsed_seconds,
        )
        rows.append(row)

        print(
            f"[{case_index}/{len(cases)}] "
            f"group={group_name}, "
            f"{format_case_parameters(display_parameters)} -> "
            f"cosine={row['cosine_similarity']:.4f}, "
            f"mask_overlap={row['active_mask_overlap']:.4f}, "
            f"elapsed={elapsed_seconds:.1f}s"
        )

    write_rows(OUTPUT_CSV, rows)

    print()
    print("Done.")
    print(f"Rows written: {len(rows)}")
    print(f"CSV path: {OUTPUT_CSV}")
    print(f"Base SEED: {default_config['SEED']}")


if __name__ == "__main__":
    main()
