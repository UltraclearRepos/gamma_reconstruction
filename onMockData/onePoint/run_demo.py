"""
Short entrypoint for the synthetic point-source reconstruction demo.
"""

from pathlib import Path
import sys

import numpy as np

ON_MOCK_DATA_DIR = Path(__file__).resolve().parents[1]
if str(ON_MOCK_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(ON_MOCK_DATA_DIR))

from reconstruction_algorithms import reconstruct_source
from simulation import generate_synthetic_dataset
from visualization import (
    visualize_detector_geometry,
    visualize_detector_trajectory_world,
    visualize_reconstruction_history_3d,
)


def print_summary(dataset, method, estimated_position, estimated_strength, history):
    """Print a short summary to the terminal."""
    position_error = np.linalg.norm(estimated_position - dataset["true_source_position"])
    num_channels = dataset["counts"].shape[1]
    best_score = history[-1].get("best_score") if history else None

    print("=== Synthetic point-source reconstruction demo ===")
    print(f"Reconstruction method:       {method}")
    print()
    print("True source position:       ", np.round(dataset["true_source_position"], 4))
    print("Estimated source position:  ", np.round(estimated_position, 4))
    print(f"Position error:              {100.0 * position_error:.2f} cm")
    print()
    print(f"True source strength A:      {dataset['true_source_strength']:.2f}")
    print(f"Estimated source strength A: {estimated_strength:.2f}")
    if best_score is not None:
        print(f"Best score:                  {best_score:.2f}")
    print()
    print(f"Measurement duration [s]:    {dataset['measurement_duration_seconds']:.2f}")
    print(f"Background mean [/s]:        {dataset['background_mean_per_second']:.1f}")
    print(f"Background mean / sample:    {dataset['background_mean']:.1f}")
    print(f"Peak source signal [/s]:     {dataset['peak_source_signal_per_second']:.1f}")
    print(f"Peak source signal/sample:   {dataset['peak_source_signal']:.1f}")
    print()
    print(f"First 5 measurements [channels 0..{num_channels - 1}]:")
    for row in dataset["counts"][:5]:
        print(" ", row.astype(int).tolist())


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


def format_history_value(key, value):
    """Format one history value without dumping huge numpy arrays."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return f"array(shape={value.shape}, empty)"

        if value.ndim == 1 and value.size <= 6:
            return str(np.round(value, 4).tolist())

        if key.endswith("_scores"):
            return (
                f"array(shape={value.shape}, "
                f"min={np.min(value):.2f}, max={np.max(value):.2f})"
            )

        return f"array(shape={value.shape})"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def print_history(history):
    print("Search history:")
    for index, item in enumerate(history, start=1):
        print(f" step={index}")
        for key, value in item.items():
            print(f"  {key}: {format_history_value(key, value)}")


def main():
    # Change only this value while testing reconstruction methods:
    # "coarse_to_fine", "grid_search", or "random_search".
    reconstruction_method = "coarse_to_fine"

    # Mock data generation.
    SEED = 7  # Random seed for mock data noise, channel scales, backgrounds, and trajectory jitter.
    NUM_MEASUREMENTS = 24  # Number of detector positions / generated measurements.
    TRUE_SOURCE_POSITION = (0.18, -0.12, 0.26)  # Ground-truth point source position [m].
    NUM_CHANNELS = 16  # Number of detector channels.
    DIODE_PITCH = 0.006  # Distance between neighboring detector channels [m].
    MEASUREMENT_DURATION_SECONDS = 1.0  # Duration of one measurement; rates below are multiplied by this.
    BACKGROUND_MEAN_PER_SECOND = 12000.0  # Mean background counts per channel per second.
    BACKGROUND_CHANNEL_STD_PER_SECOND = 350.0  # Channel-to-channel background std per second.
    CHANNEL_SCALE_MEAN = 1.0  # Mean multiplicative channel sensitivity.
    CHANNEL_SCALE_STD = 0.08  # Channel-to-channel sensitivity mismatch std.
    GENERATION_K = 4.5  # >= 0; sensible ~0..10. Higher = narrower angular response, so off-axis sources give fewer counts.
    PEAK_SOURCE_SIGNAL_PER_SECOND = 4000.0  # Peak source signal rate in the best measurement.

    # Reconstruction volume.
    SEARCH_MIN = (-0.40, -0.50, -0.10)  # Minimum x/y/z corner of searched source position volume [m].
    SEARCH_MAX = (0.60, 0.30, 0.60)  # Maximum x/y/z corner of searched source position volume [m].

    # Mock detector trajectory.
    TRAJECTORY_MODE = "grid"  # "grid" or "around_center"
    ORIENTATION_MODE = "fixed"  # "fixed" or "towards_target" | "towards_target" recommended for TRAJECTORY_MODE="around_center"
    FIXED_VIEW_DIRECTION = (0, 1, 0)  # e.g. (1, 0, 0) to "look right" in world coords
    FIXED_SWEEP_Z_DEG = 10.0  # fixed mode only: left/right viewing sweep around world Z axis.
    FIXED_SWEEP_X_DEG = 6.0  # fixed mode only: up/down viewing sweep around world X axis.
    SCAN_CENTER = tuple((np.array(SEARCH_MIN) + np.array(SEARCH_MAX)) / 2.0)  # Center of scan path; should match search volume center.
    ANGULAR_JITTER_DEG = 3.0  # Random orientation noise per measurement; simulates imperfect manual sensor motion.
    LATERAL_SPAN = 0.45  # grid only: max left/right position offset from SCAN_CENTER [m].
    VERTICAL_SPAN = 0.28  # grid only: max up/down position offset from SCAN_CENTER [m].
    BASE_DISTANCE = 0.65  # grid only: nominal detector distance from SCAN_CENTER [m].
    DISTANCE_SPAN = 0.18  # grid only: sinusoidal variation of detector distance [m].
    ORBIT_RADIUS = 0.90  # around_center only: nominal radius of detector orbit around SCAN_CENTER [m].
    ORBIT_RADIUS_SPAN = 0.05  # around_center only: sinusoidal radius variation during orbit [m].
    ORBIT_ANGLE_SPAN_DEG = 360.0  # around_center only: covered orbit angle; 360 means full circle.
    ORBIT_START_ANGLE_DEG = 0.0  # around_center only: starting angle of orbit in XY plane.
    ORBIT_VERTICAL_SPAN = 0.18  # around_center only: sinusoidal up/down motion during orbit [m].

    # Assumed reconstruction model.
    ASSUMED_BACKGROUND_MEAN_PER_SECOND = BACKGROUND_MEAN_PER_SECOND  # Background rate used by reconstruction model.
    ASSUMED_CHANNEL_SCALE = CHANNEL_SCALE_MEAN  # Global channel sensitivity used by reconstruction model.
    ASSUMED_K = GENERATION_K  # Angular sensitivity exponent used by reconstruction model.

    # Reconstruction algorithm.
    GRID_SIZES = (13, 11, 9)  # coarse_to_fine grid sizes for successive refinement levels.
    GRID_SIZE = 21  # grid_search grid size per axis.
    RANDOM_SAMPLES = 4000  # random_search candidate count.
    RANDOM_SEED = 0  # random_search seed.

    # Visualization output.
    PRINT_GENERATED_COUNTS_TABLE = True  # Print full generated measurement x channel table.
    HISTORY_MAX_POINTS = 500000  # Max candidate points plotted in reconstruction history.

    dataset = generate_synthetic_dataset(
        num_measurements=NUM_MEASUREMENTS,
        seed=SEED,
        orientation_mode=ORIENTATION_MODE,
        fixed_view_direction=FIXED_VIEW_DIRECTION,
        true_source_position=TRUE_SOURCE_POSITION,
        num_channels=NUM_CHANNELS,
        diode_pitch=DIODE_PITCH,
        background_mean_per_second=BACKGROUND_MEAN_PER_SECOND,
        background_channel_std_per_second=BACKGROUND_CHANNEL_STD_PER_SECOND,
        channel_scale_mean=CHANNEL_SCALE_MEAN,
        channel_scale_std=CHANNEL_SCALE_STD,
        generation_k=GENERATION_K,
        peak_source_signal_per_second=PEAK_SOURCE_SIGNAL_PER_SECOND,
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
    num_channels = dataset["counts"].shape[1]

    # Assumed model used during reconstruction.
    # Change this dict to test how reconstruction behaves with different
    # detector/background assumptions.
    assumed_model = {
        "assumed_backgrounds": np.full(
            num_channels,
            ASSUMED_BACKGROUND_MEAN_PER_SECOND * MEASUREMENT_DURATION_SECONDS,
            dtype=float,
        ),
        "assumed_channel_scales": np.full(num_channels, ASSUMED_CHANNEL_SCALE, dtype=float),
        "assumed_k": ASSUMED_K,
    }

    estimated_position, estimated_strength, history = reconstruct_source(
        counts=dataset["counts"],
        geometry=dataset["geometry"],
        rotations=dataset["rotations"],
        translations=dataset["translations"],
        **assumed_model,
        search_min=SEARCH_MIN,
        search_max=SEARCH_MAX,
        method=reconstruction_method,
        grid_sizes=GRID_SIZES,
        grid_size=GRID_SIZE,
        random_samples=RANDOM_SAMPLES,
        random_seed=RANDOM_SEED,
    )

    print_summary(
        dataset,
        reconstruction_method,
        estimated_position,
        estimated_strength,
        history,
    )
    if PRINT_GENERATED_COUNTS_TABLE:
        print_generated_counts_table(dataset)
    print()
    print_history(history)

    output_dir = ON_MOCK_DATA_DIR / "onePoint" / "visualizations"
    visualize_detector_geometry(
        dataset["geometry"],
        save_path=output_dir / "detector_geometry.png",
        show=False,
    )
    visualize_detector_trajectory_world(
        dataset,
        save_path=output_dir / "detector_trajectory_world.png",
        show=False,
    )

    visualize_reconstruction_history_3d(
        history=history,
        estimated_position=estimated_position,
        true_source_position=dataset["true_source_position"],
        detector_positions=dataset["translations"],
        save_path=output_dir / f"reconstruction_history_3d_{reconstruction_method}.png",
        max_points=HISTORY_MAX_POINTS,
        show=False,
    )

    print()
    print("Saved plots:")
    print(" ", output_dir / "detector_geometry.png")
    print(" ", output_dir / "detector_trajectory_world.png")
    print(" ", output_dir / f"reconstruction_history_3d_{reconstruction_method}.png")


if __name__ == "__main__":
    main()
