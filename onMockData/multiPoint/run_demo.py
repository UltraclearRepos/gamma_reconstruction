"""
Entrypoint for the synthetic multi-point source demo.
"""

from pathlib import Path
import sys

import numpy as np

ON_MOCK_DATA_DIR = Path(__file__).resolve().parents[1]
if str(ON_MOCK_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(ON_MOCK_DATA_DIR))

from reconstruction_algorithms import reconstruct_sources
from metrics import print_activity_localization_summary, summarize_activity_localization
from simulation import generate_synthetic_dataset
from visualization import (
    visualize_history_score,
    visualize_mlem_activity,
    visualize_multi_source_scene,
)


def print_summary(dataset):
    """Print a short summary of the generated multi-source dataset."""
    counts = dataset["counts"]

    print("=== Synthetic multi-point source demo ===")
    print()
    print("True sources:")
    for index, source in enumerate(dataset["true_sources"], start=1):
        print(
            " "
            f"S{index}: position={np.round(source['position'], 4)}, "
            f"strength={source['strength']:.2f}"
        )

    print()
    print(
        "Counts min/mean/max:        "
        f"{counts.min():.1f} / {counts.mean():.1f} / {counts.max():.1f}"
    )
    print(
        "Measurement mean min/max:   "
        f"{counts.mean(axis=1).min():.1f} / {counts.mean(axis=1).max():.1f}"
    )
    print(f"Measurement duration [s]:   {dataset['measurement_duration_seconds']:.2f}")
    print(f"Background mean [/s]:       {dataset['background_mean_per_second']:.1f}")
    print(f"Background mean / sample:   {dataset['background_mean']:.1f}")
    print(f"Peak source signal [/s]:    {dataset['peak_total_source_signal_per_second']:.1f}")
    print(f"Peak source signal/sample:  {dataset['peak_total_source_signal']:.1f}")


def print_generated_counts_table(dataset):
    """Print generated count samples for every measurement and detector channel."""
    counts = np.asarray(dataset["counts"], dtype=float)
    num_measurements, num_channels = counts.shape
    channel_headers = [f"ch{channel_index + 1:02d}" for channel_index in range(num_channels)]
    header = ["idx", *channel_headers, "sum", "mean"]

    print()
    print(f"Generated count samples ({num_measurements} measurements x {num_channels} channels):")
    print(" ".join(f"{column:>7}" for column in header))
    for measurement_index, measurement_counts in enumerate(counts, start=1):
        row = [
            f"{measurement_index:7d}",
            *[f"{value:7.0f}" for value in measurement_counts],
            f"{np.sum(measurement_counts):7.0f}",
            f"{np.mean(measurement_counts):7.1f}",
        ]
        print(" ".join(row))


def print_reconstruction_summary(method, result):
    """Print compact ML-EM reconstruction output."""
    print()
    print(f"Reconstruction method:       {method}")
    print()
    print("Estimated activity peaks:")
    for index, source in enumerate(result["estimated_sources"], start=1):
        print(
            " "
            f"P{index}: position={np.round(source['position'], 4)}, "
            f"activity={source['activity']:.2f}"
        )

    print()
    print("Reconstruction history:")
    for index, item in enumerate(result["history"], start=1):
        print(f" step={index}")
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {np.round(value, 4).tolist()}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def print_localization_metrics(result, true_sources, relative_drop):
    """Print localization metrics for the activity map."""
    metrics = summarize_activity_localization(
        voxel_positions=result["candidate_positions"],
        activity=result["activity"],
        true_sources=true_sources,
        max_relative_drop=relative_drop,
    )
    print_activity_localization_summary(metrics)


def main():
    # Change only this value while testing multi-source reconstruction methods.
    # Currently available: "mlem".
    reconstruction_method = "mlem"

    # Mock data generation.
    SEED = 11
    NUM_MEASUREMENTS = 50
    TRUE_SOURCE_SPECS = [
        {"position": (0.18, -0.2, 0.2), "relative_strength": 1.0},
        # {"position": (-0.10, -0.18, 0.16), "relative_strength": 0.65},
        # {"position": (0.34, -0.07, 0.38), "relative_strength": 0.45},
    ]
    NUM_CHANNELS = 16
    DIODE_PITCH = 0.006
    MEASUREMENT_DURATION_SECONDS = 10.0
    BACKGROUND_MEAN_PER_SECOND = 0.0
    BACKGROUND_CHANNEL_STD_PER_SECOND = 0.0
    CHANNEL_SCALE_MEAN = 1.0
    CHANNEL_SCALE_STD = 0.08
    GENERATION_K = 4.5  # >= 0; sensible ~0..10. Higher = narrower angular response, so off-axis sources give fewer counts.
    PEAK_TOTAL_SOURCE_SIGNAL_PER_SECOND = 5000.0

    # Reconstruction volume.
    SEARCH_MIN = (-0.55, -0.55, -0.55)
    SEARCH_MAX = (0.55, 0.55, 0.55)
    GRID_SIZE = 13

    # Mock detector trajectory.
    TRAJECTORY_MODE = "grid_scan"  # "grid_scan" or "around_center"
    ORIENTATION_MODE = "fixed"  # "fixed" or "towards_target" | "towards_target" recommended for TRAJECTORY_MODE="around_center"
    FIXED_VIEW_DIRECTION = (0, 1, 0)  # e.g. (1, 0, 0) to "look right" in world coords
    FIXED_SWEEP_Z_DEG = 0.0  # fixed mode only: left/right viewing sweep around world Z axis.
    FIXED_SWEEP_X_DEG = 0.0  # fixed mode only: up/down viewing sweep around world X axis.
    SCAN_CENTER = tuple((np.array(SEARCH_MIN) + np.array(SEARCH_MAX)) / 2.0)  # Center of scan path; should match reconstruction volume center.
    ANGULAR_JITTER_DEG = 3.0  # Random orientation noise per measurement; simulates imperfect manual sensor motion.
    LATERAL_SPAN = 0.60  # grid_scan only: max left/right position offset from SCAN_CENTER [m].
    VERTICAL_SPAN = 0.36  # grid_scan only: max up/down position offset from SCAN_CENTER [m].
    BASE_DISTANCE = 0.65  # grid_scan only: nominal detector distance from SCAN_CENTER [m].
    DISTANCE_SPAN = 0.18  # grid_scan only: sinusoidal variation of detector distance [m].
    ORBIT_RADIUS = 1.2  # around_center only: nominal radius of detector orbit around SCAN_CENTER [m].
    ORBIT_RADIUS_SPAN = 0  # around_center only: sinusoidal radius variation during orbit [m].
    ORBIT_ANGLE_SPAN_DEG = 360.0  # around_center only: covered orbit angle; 360 means full circle.
    ORBIT_START_ANGLE_DEG = 0.0  # around_center only: starting angle of orbit in XY plane.
    ORBIT_VERTICAL_SPAN = 0.2  # around_center only: sinusoidal up/down motion during orbit [m].

    # Assumed reconstruction model.
    ASSUMED_BACKGROUND_MEAN_PER_SECOND = BACKGROUND_MEAN_PER_SECOND
    ASSUMED_CHANNEL_SCALE = CHANNEL_SCALE_MEAN
    ASSUMED_K = GENERATION_K

    # Reconstruction algorithm.
    NUM_ITERATIONS = 10000
    HISTORY_INTERVAL = 1000
    MAX_SOURCES_TO_REPORT = 1
    MIN_SOURCE_DISTANCE = 0.08

    # Localization metric.
    ACTIVITY_MAX_RELATIVE_DROP = 0.5

    # Visualization output.
    PRINT_GENERATED_COUNTS_TABLE = True
    AXIS_LIMITS_3D = (
        (-1, 1),
        (-1, 1),
        (-1, 1),
    )

    dataset = generate_synthetic_dataset(
        num_measurements=NUM_MEASUREMENTS,
        seed=SEED,
        orientation_mode=ORIENTATION_MODE,
        fixed_view_direction=FIXED_VIEW_DIRECTION,
        source_specs=TRUE_SOURCE_SPECS,
        num_channels=NUM_CHANNELS,
        diode_pitch=DIODE_PITCH,
        background_mean_per_second=BACKGROUND_MEAN_PER_SECOND,
        background_channel_std_per_second=BACKGROUND_CHANNEL_STD_PER_SECOND,
        channel_scale_mean=CHANNEL_SCALE_MEAN,
        channel_scale_std=CHANNEL_SCALE_STD,
        generation_k=GENERATION_K,
        peak_total_source_signal_per_second=PEAK_TOTAL_SOURCE_SIGNAL_PER_SECOND,
        measurement_duration_seconds=MEASUREMENT_DURATION_SECONDS,
        scan_center=SCAN_CENTER,
        trajectory_mode=TRAJECTORY_MODE,
        angular_jitter_deg=ANGULAR_JITTER_DEG,
        lateral_span=LATERAL_SPAN,
        vertical_span=VERTICAL_SPAN,
        base_distance=BASE_DISTANCE,
        distance_span=DISTANCE_SPAN,
        orbit_radius=ORBIT_RADIUS,
        orbit_radius_span=ORBIT_RADIUS_SPAN,
        orbit_angle_span_deg=ORBIT_ANGLE_SPAN_DEG,
        orbit_start_angle_deg=ORBIT_START_ANGLE_DEG,
        orbit_vertical_span=ORBIT_VERTICAL_SPAN,
        fixed_sweep_z_deg=FIXED_SWEEP_Z_DEG,
        fixed_sweep_x_deg=FIXED_SWEEP_X_DEG,
    )
    print_summary(dataset)
    if PRINT_GENERATED_COUNTS_TABLE:
        print_generated_counts_table(dataset)

    num_channels = dataset["counts"].shape[1]

    assumed_model = {
        "assumed_backgrounds": np.full(
            num_channels,
            ASSUMED_BACKGROUND_MEAN_PER_SECOND * MEASUREMENT_DURATION_SECONDS,
            dtype=float,
        ),
        "assumed_channel_scales": np.full(num_channels, ASSUMED_CHANNEL_SCALE, dtype=float),
        "assumed_k": ASSUMED_K,
    }

    result = reconstruct_sources(
        counts=dataset["counts"],
        geometry=dataset["geometry"],
        rotations=dataset["rotations"],
        translations=dataset["translations"],
        **assumed_model,
        search_min=SEARCH_MIN,
        search_max=SEARCH_MAX,
        method=reconstruction_method,
        grid_size=GRID_SIZE,
        num_iterations=NUM_ITERATIONS,
        history_interval=HISTORY_INTERVAL,
        max_sources=MAX_SOURCES_TO_REPORT,
        min_source_distance=MIN_SOURCE_DISTANCE,
    )
    print_reconstruction_summary(reconstruction_method, result)
    print_localization_metrics(
        result=result,
        true_sources=dataset["true_sources"],
        relative_drop=ACTIVITY_MAX_RELATIVE_DROP,
    )

    output_dir = ON_MOCK_DATA_DIR / "multiPoint" / "visualizations"
    visualize_multi_source_scene(
        dataset,
        save_path=output_dir / "multi_source_scene.png",
        show=False,
        axis_limits=AXIS_LIMITS_3D,
    )
    visualize_mlem_activity(
        dataset,
        candidate_positions=result["candidate_positions"],
        activity=result["activity"],
        estimated_sources=result["estimated_sources"],
        save_path=output_dir / "mlem_activity_reconstruction.png",
        show=False,
        axis_limits=AXIS_LIMITS_3D,
        highlight_used_trajectory=False,
    )
    visualize_history_score(
        result["history"],
        save_path=output_dir / "mlem_score_history.png",
        show=False,
    )

    print()
    print("Saved plots:")
    print(" ", output_dir / "multi_source_scene.png")
    print(" ", output_dir / "mlem_activity_reconstruction.png")
    print(" ", output_dir / "mlem_score_history.png")


if __name__ == "__main__":
    main()
