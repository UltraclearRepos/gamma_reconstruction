"""
Entrypoint for streaming ML-EM reconstruction on synthetic multi-point data.
"""

import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ON_MOCK_DATA_DIR = Path(__file__).resolve().parents[1]
if str(ON_MOCK_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(ON_MOCK_DATA_DIR))

from simulation import generate_synthetic_dataset
from streaming_algorithms import reconstruct_sources
from visualization import (
    create_gif_from_frames,
    visualize_history_score,
    visualize_mlem_activity,
    visualize_multi_source_scene,
)


def print_dataset_summary(dataset):
    """Print short summary for the generated multi-point dataset."""
    counts = dataset["counts"]
    print("=== Streaming multi-point source demo ===")
    print()
    print("Trajectory center:")
    print(dataset["trajectory_center"])
    print()
    print("True sources:")
    for index, source in enumerate(dataset["true_sources"], start=1):
        print(
            " "
            f"S{index}: position={np.round(source['position'], 4)}, "
            f"strength_per_second={source['strength_per_second']:.2f}, "
            f"strength_per_sample={source['strength']:.2f}"
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


def print_streaming_summary(result):
    """Print prediction after each streaming update."""
    print()
    print("Streaming updates:")
    for item in result["stream_history"]:
        print(
            " "
            f"used_measurements={int(item['used_measurements'])}, "
            f"total_activity={item['total_activity']:.2f}, "
            f"max_activity={item['max_activity']:.2f}, "
            f"signal_scale={item['signal_normalization_scale']:.3f}, "
            f"score={item['score']:.2f}"
        )


def to_builtin_value(value):
    """Convert numpy values nested in lists/dicts into plain Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_builtin_value(item) for item in value]
    if isinstance(value, dict):
        return {key: to_builtin_value(item) for key, item in value.items()}
    return value


def to_csv_value(value):
    """Convert numpy and structured values into CSV-friendly scalars/JSON."""
    value = to_builtin_value(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def append_streaming_run_csv(csv_path, parameters, final_history_item):
    """Append run parameters and final streaming map summary to a growing CSV."""

    row = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "final_update_index": int(final_history_item["update_index"]),
        "final_used_measurements": int(final_history_item["used_measurements"]),
        "final_score": float(final_history_item["score"]),
        "final_total_activity": float(final_history_item["total_activity"]),
        "final_max_activity": float(final_history_item["max_activity"]),
        "final_observed_signal": float(final_history_item["observed_signal"]),
        "final_predicted_signal": float(final_history_item["predicted_signal"]),
        "final_signal_normalization_scale": float(final_history_item["signal_normalization_scale"]),
    }
    for key, value in parameters.items():
        row[f"param_{key.lower()}"] = to_csv_value(value)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        merged_fieldnames = existing_fieldnames + [
            fieldname for fieldname in fieldnames if fieldname not in existing_fieldnames
        ]
        if merged_fieldnames != existing_fieldnames:
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=merged_fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
        fieldnames = merged_fieldnames

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    # Mock data generation.
    SEED = 11
    NUM_MEASUREMENTS = 50
    TRUE_SOURCE_SPECS = [
        {"position": (18.0, -20.0, 20.0), "strength_per_second": 5000.0},
        # {"position": (-10.0, -18.0, 16.0), "strength_per_second": 3250.0},
        # {"position": (34.0, -7.0, 38.0), "strength_per_second": 2250.0},
    ]
    NUM_CHANNELS = 16
    DIODE_PITCH = 0.15875  # Channel width and center-to-center pitch [cm]; 16 channels span 2.54 cm.
    MEASUREMENT_DURATION_SECONDS = 10.0
    BACKGROUND_MEAN_PER_SECOND = 0.0
    BACKGROUND_CHANNEL_STD_PER_SECOND = 0.0
    CHANNEL_SCALE_MEAN = 1.0
    CHANNEL_SCALE_STD = 0.08
    GENERATION_K = 4.5  # >= 0; sensible ~0..10. Higher = narrower angular response, so off-axis sources give fewer counts.
    GENERATION_DISTANCE_POWER = 1.0  # Distance falloff exponent: response ~= 1 / distance^power.

    # Reconstruction volume.
    SEARCH_START_DISTANCE = 10.0  # Start searching this far from each detector channel along detector forward axis [cm].
    SEARCH_STOP_DISTANCE = 120.0  # Stop searching this far from each detector channel; window length = stop - start [cm].
    SEARCH_WIDTH = 45.0  # Full width of the detector-relative rectangular search box [cm].
    SEARCH_HEIGHT = 36.0  # Full height of the detector-relative rectangular search box [cm].
    VOXEL_SIZE_CM = 8.0

    # Mock detector trajectory.
    TRAJECTORY_MODE = "grid"  # "grid" or "around_center"
    ORIENTATION_MODE = "fixed"  # "fixed" or "towards_target" | "towards_target" recommended for TRAJECTORY_MODE="around_center"
    FIXED_VIEW_DIRECTION = (0, 1, 0)  # e.g. (1, 0, 0) to "look right" in world coords
    FIXED_SWEEP_Z_RANGE_DEG = 20.0  # random left/right angle range around world Z axis, sampled as +/- this value.
    FIXED_SWEEP_X_RANGE_DEG = 20.0  # random up/down angle range around world X axis, sampled as +/- this value.
    TRAJECTORY_CENTER = (0.0, 0.0, 0.0)  # Reference center of the detector trajectory [cm].
    ANGULAR_JITTER_DEG = 3.0  # Random orientation noise per measurement; simulates imperfect manual sensor motion.
    LATERAL_SPAN = 60.0  # grid only: max left/right position offset from TRAJECTORY_CENTER [cm].
    VERTICAL_SPAN = 36.0  # grid only: max up/down position offset from TRAJECTORY_CENTER [cm].
    BASE_DISTANCE = 65.0  # grid only: nominal detector distance from TRAJECTORY_CENTER [cm].
    DISTANCE_SPAN = 18.0  # grid only: sinusoidal variation of detector distance [cm].
    ORBIT_RADIUS = 120.0  # around_center only: nominal radius of detector orbit around TRAJECTORY_CENTER [cm].
    ORBIT_RADIUS_SPAN = 0  # around_center only: sinusoidal radius variation during orbit [cm].
    ORBIT_ANGLE_SPAN_DEG = 360.0  # around_center only: covered orbit angle; 360 means full circle.
    ORBIT_START_ANGLE_DEG = 0.0  # around_center only: starting angle of orbit in XY plane.
    ORBIT_VERTICAL_SPAN = 20.0  # around_center only: sinusoidal up/down motion during orbit [cm].

    # Assumed reconstruction model.
    ASSUMED_BACKGROUND_MEAN_PER_SECOND = BACKGROUND_MEAN_PER_SECOND
    ASSUMED_CHANNEL_SCALE = CHANNEL_SCALE_MEAN
    ASSUMED_K = GENERATION_K
    ASSUMED_DISTANCE_POWER = GENERATION_DISTANCE_POWER

    # Streaming reconstruction algorithm.
    STREAMING_METHOD = "incremental_mlem"  # "prefix_mlem" or "incremental_mlem"
    RELAXATION = 0.95  # used by incremental_mlem; valid range: (0, 1];  1.0 = no relaxation (standard MLEM); smaller = more conservative updates
    INNER_ITERATIONS_PER_UPDATE = 50
    MEASUREMENTS_PER_UPDATE = 2
    SPARSITY_LAMBDA = 0.0  # L1-like activity penalty; increase to suppress diffuse weak voxels.
    AVOID_DETECTOR_STRENGTH = 0.2  # 0 disables; how much distance is penalized
    AVOID_DETECTOR_DECAY_PER_UPDATE = 0  # penalty strength is multiplied by this after each update; bigger = faster decay
    ACTIVITY_DECAY = 0.9  # 1.0 disables; try 0.95..0.995 to "forget" old spikes
    NORMALIZE_ACTIVITY_TO_SIGNAL = True  # Rescale activity so predicted source counts match observed counts above background.

    # Visualization output.
    PRINT_GENERATED_COUNTS_TABLE = True
    UPDATE_FRAME_MAX_VOXELS = 700
    GIF_FPS = 1.0
    AXIS_LIMITS_3D = (
        (-100, 100),
        (-100, 100),
        (-100, 100),
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
        generation_distance_power=GENERATION_DISTANCE_POWER,
        measurement_duration_seconds=MEASUREMENT_DURATION_SECONDS,
        trajectory_center=TRAJECTORY_CENTER,
        search_start_distance=SEARCH_START_DISTANCE,
        search_stop_distance=SEARCH_STOP_DISTANCE,
        search_width=SEARCH_WIDTH,
        search_height=SEARCH_HEIGHT,
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
        fixed_sweep_z_range_deg=FIXED_SWEEP_Z_RANGE_DEG,
        fixed_sweep_x_range_deg=FIXED_SWEEP_X_RANGE_DEG,
    )

    print_dataset_summary(dataset)
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
        "assumed_distance_power": ASSUMED_DISTANCE_POWER,
    }

    result = reconstruct_sources(
        counts=dataset["counts"],
        geometry=dataset["geometry"],
        rotations=dataset["rotations"],
        translations=dataset["translations"],
        **assumed_model,
        search_min=dataset["search_min"],
        search_max=dataset["search_max"],
        search_start_distance=dataset["search_start_distance"],
        search_stop_distance=dataset["search_stop_distance"],
        search_width=dataset["search_width"],
        search_height=dataset["search_height"],
        true_sources=dataset["true_sources"],
        method=STREAMING_METHOD,
        voxel_size_cm=VOXEL_SIZE_CM,
        inner_iterations_per_update=INNER_ITERATIONS_PER_UPDATE,
        measurements_per_update=MEASUREMENTS_PER_UPDATE,
        sparsity_lambda=SPARSITY_LAMBDA,
        relaxation=RELAXATION,
        avoid_detector_strength=AVOID_DETECTOR_STRENGTH,
        avoid_detector_decay_per_update=AVOID_DETECTOR_DECAY_PER_UPDATE,
        activity_decay=ACTIVITY_DECAY,
        normalize_activity_to_signal=NORMALIZE_ACTIVITY_TO_SIGNAL,
    )
    print_streaming_summary(result)
    max_activity_for_visualization = max(
        float(np.max(activity_snapshot))
        for activity_snapshot in result["activity_history"]
    )
    ACTIVITY_COLOR_LIMITS = (0.0, float(np.log1p(max(max_activity_for_visualization, 0.0))))

    output_dir = ON_MOCK_DATA_DIR / "multiPoint" / "visualizations"
    run_parameters = {
        "SEED": SEED,
        "NUM_MEASUREMENTS": NUM_MEASUREMENTS,
        "TRUE_SOURCE_SPECS": TRUE_SOURCE_SPECS,
        "NUM_CHANNELS": NUM_CHANNELS,
        "DIODE_PITCH": DIODE_PITCH,
        "MEASUREMENT_DURATION_SECONDS": MEASUREMENT_DURATION_SECONDS,
        "BACKGROUND_MEAN_PER_SECOND": BACKGROUND_MEAN_PER_SECOND,
        "BACKGROUND_CHANNEL_STD_PER_SECOND": BACKGROUND_CHANNEL_STD_PER_SECOND,
        "CHANNEL_SCALE_MEAN": CHANNEL_SCALE_MEAN,
        "CHANNEL_SCALE_STD": CHANNEL_SCALE_STD,
        "GENERATION_K": GENERATION_K,
        "GENERATION_DISTANCE_POWER": GENERATION_DISTANCE_POWER,
        "SEARCH_START_DISTANCE": SEARCH_START_DISTANCE,
        "SEARCH_STOP_DISTANCE": SEARCH_STOP_DISTANCE,
        "SEARCH_WIDTH": SEARCH_WIDTH,
        "SEARCH_HEIGHT": SEARCH_HEIGHT,
        "VOXEL_SIZE_CM": VOXEL_SIZE_CM,
        "TRAJECTORY_MODE": TRAJECTORY_MODE,
        "ORIENTATION_MODE": ORIENTATION_MODE,
        "FIXED_VIEW_DIRECTION": FIXED_VIEW_DIRECTION,
        "FIXED_SWEEP_Z_RANGE_DEG": FIXED_SWEEP_Z_RANGE_DEG,
        "FIXED_SWEEP_X_RANGE_DEG": FIXED_SWEEP_X_RANGE_DEG,
        "TRAJECTORY_CENTER": TRAJECTORY_CENTER,
        "ANGULAR_JITTER_DEG": ANGULAR_JITTER_DEG,
        "LATERAL_SPAN": LATERAL_SPAN,
        "VERTICAL_SPAN": VERTICAL_SPAN,
        "BASE_DISTANCE": BASE_DISTANCE,
        "DISTANCE_SPAN": DISTANCE_SPAN,
        "ORBIT_RADIUS": ORBIT_RADIUS,
        "ORBIT_RADIUS_SPAN": ORBIT_RADIUS_SPAN,
        "ORBIT_ANGLE_SPAN_DEG": ORBIT_ANGLE_SPAN_DEG,
        "ORBIT_START_ANGLE_DEG": ORBIT_START_ANGLE_DEG,
        "ORBIT_VERTICAL_SPAN": ORBIT_VERTICAL_SPAN,
        "ASSUMED_BACKGROUND_MEAN_PER_SECOND": ASSUMED_BACKGROUND_MEAN_PER_SECOND,
        "ASSUMED_CHANNEL_SCALE": ASSUMED_CHANNEL_SCALE,
        "ASSUMED_K": ASSUMED_K,
        "ASSUMED_DISTANCE_POWER": ASSUMED_DISTANCE_POWER,
        "STREAMING_METHOD": STREAMING_METHOD,
        "RELAXATION": RELAXATION,
        "INNER_ITERATIONS_PER_UPDATE": INNER_ITERATIONS_PER_UPDATE,
        "MEASUREMENTS_PER_UPDATE": MEASUREMENTS_PER_UPDATE,
        "SPARSITY_LAMBDA": SPARSITY_LAMBDA,
        "AVOID_DETECTOR_STRENGTH": AVOID_DETECTOR_STRENGTH,
        "AVOID_DETECTOR_DECAY_PER_UPDATE": AVOID_DETECTOR_DECAY_PER_UPDATE,
        "ACTIVITY_DECAY": ACTIVITY_DECAY,
        "NORMALIZE_ACTIVITY_TO_SIGNAL": NORMALIZE_ACTIVITY_TO_SIGNAL,
        "PRINT_GENERATED_COUNTS_TABLE": PRINT_GENERATED_COUNTS_TABLE,
        "UPDATE_FRAME_MAX_VOXELS": UPDATE_FRAME_MAX_VOXELS,
        "GIF_FPS": GIF_FPS,
        "AXIS_LIMITS_3D": AXIS_LIMITS_3D,
    }
    STREAMING_RUNS_CSV = ON_MOCK_DATA_DIR / "multiPoint" / "visualizations" / "streaming_runs.csv"
    append_streaming_run_csv(
        csv_path=STREAMING_RUNS_CSV,
        parameters=run_parameters,
        final_history_item=result["stream_history"][-1],
    )
    updates_3d_dir = output_dir / "streaming_activity_updates_3d"
    updates_3d_dir.mkdir(parents=True, exist_ok=True)

    visualize_multi_source_scene(
        dataset,
        save_path=output_dir / "multi_source_scene_streaming.png",
        show=False,
        axis_limits=AXIS_LIMITS_3D,
    )
    visualize_mlem_activity(
        dataset,
        candidate_positions=result["candidate_positions"],
        activity=result["activity"],
        used_measurements=len(dataset["translations"]),
        save_path=output_dir / "streaming_mlem_activity_reconstruction.png",
        show=False,
        axis_limits=AXIS_LIMITS_3D,
        activity_color_limits=ACTIVITY_COLOR_LIMITS,
        activity_size_reference=max_activity_for_visualization,
    )
    visualize_history_score(
        result["stream_history"],
        save_path=output_dir / "streaming_mlem_score_history.png",
        show=False,
    )

    previous_used_measurements = 0
    for update_index, (activity_snapshot, history_item) in enumerate(
        zip(result["activity_history"], result["stream_history"]),
        start=1,
    ):
        used_measurements = int(history_item.get("used_measurements", previous_used_measurements))
        visualize_mlem_activity(
            dataset,
            candidate_positions=result["candidate_positions"],
            activity=activity_snapshot,
            used_measurements=used_measurements,
            update_start=previous_used_measurements,
            update_end=used_measurements,
            save_path=updates_3d_dir / f"update_{update_index:03d}.png",
            show=False,
            max_voxels=UPDATE_FRAME_MAX_VOXELS,
            axis_limits=AXIS_LIMITS_3D,
            activity_color_limits=ACTIVITY_COLOR_LIMITS,
            activity_size_reference=max_activity_for_visualization,
        )
        previous_used_measurements = used_measurements

    updates_gif_path = create_gif_from_frames(
        frames_dir=updates_3d_dir,
        output_path=output_dir / "streaming_activity_updates_3d.gif",
        pattern="update_*.png",
        fps=GIF_FPS,
    )

    print()
    print("Saved plots:")
    print(" ", output_dir / "multi_source_scene_streaming.png")
    print(" ", output_dir / "streaming_mlem_activity_reconstruction.png")
    print(" ", output_dir / "streaming_mlem_score_history.png")
    print(" ", STREAMING_RUNS_CSV)
    print(" ", updates_3d_dir / "update_001.png ...")
    print(" ", updates_gif_path)


if __name__ == "__main__":
    main()
