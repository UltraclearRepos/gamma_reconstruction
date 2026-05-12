"""
Entrypoint for the synthetic multi-point source demo.
"""

from pathlib import Path
import sys

import numpy as np

ON_MOCK_DATA_DIR = Path(__file__).resolve().parents[1]
if str(ON_MOCK_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(ON_MOCK_DATA_DIR))

from reconstruction_algorithms import pick_activity_peaks, reconstruct_sources
from metrics import (
    print_activity_localization_summary,
    print_estimated_source_distance_summary,
    summarize_activity_localization,
    summarize_estimated_source_distances,
)
from simulation import generate_synthetic_dataset
from visualization import (
    create_gif_from_frames,
    visualize_history_score,
    visualize_mlem_activity,
    visualize_multi_source_scene,
)


def make_default_config():
    """Return default settings for the synthetic multi-point demo."""
    search_min = (-50.0, -50.0, -50.0)
    search_max = (50.0, 50.0, 50.0)

    return {
        # Reconstruction method.
        "reconstruction_method": "mlem",

        # Mock data generation.
        "SEED": 11,
        "NUM_MEASUREMENTS": 30,
        "TRUE_SOURCE_SPECS": [
            {"position": (18.0, -20.0, 20.0), "relative_strength": 1.0},
            # {"position": (-10.0, -18.0, 16.0), "relative_strength": 0.65},
            # {"position": (34.0, -7.0, 38.0), "relative_strength": 0.45},
        ],
        "NUM_CHANNELS": 16,
        "DIODE_PITCH": 0.6,
        "MEASUREMENT_DURATION_SECONDS": 1.0,
        "BACKGROUND_MEAN_PER_SECOND": 1000.0,
        "BACKGROUND_CHANNEL_STD_PER_SECOND": 200.0,
        "CHANNEL_SCALE_MEAN": 1.0,
        "CHANNEL_SCALE_STD": 0.08,
        "GENERATION_K": 4.5,
        "PEAK_TOTAL_SOURCE_SIGNAL_PER_SECOND": 5000.0,  

        # Assumed reconstruction model.
        "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,
        "ASSUMED_CHANNEL_SCALE": 1.0,
        "ASSUMED_K": 4.5,

        # Reconstruction volume.
        "SEARCH_MIN": search_min,
        "SEARCH_MAX": search_max,
        "GRID_SIZE": 13,

        # Mock detector trajectory.
        "TRAJECTORY_MODE": "line",
        "LATERAL_SPAN": 60.0,
        "VERTICAL_SPAN": 0.0,
        "BASE_DISTANCE": 100.0,
        "DISTANCE_SPAN": 0.0,
        "ORBIT_RADIUS": 130.0,
        "ORBIT_RADIUS_SPAN": 0,
        "ORBIT_ANGLE_SPAN_DEG": 360.0,
        "ORBIT_START_ANGLE_DEG": 0.0,
        "ORBIT_VERTICAL_SPAN": 0.0,

        "ORIENTATION_MODE": "fixed",
        "FIXED_VIEW_DIRECTION": (0, 1, 0),
        "FIXED_SWEEP_Z_DEG": 60.0,
        "FIXED_SWEEP_X_DEG": -60.0,
        "SCAN_CENTER": tuple((np.array(search_min) + np.array(search_max)) / 2.0),
        "ANGULAR_JITTER_DEG": 20.0,

        # Reconstruction algorithm.
        "NUM_ITERATIONS": 10000,
        "HISTORY_INTERVAL": 1000,
        "MAX_SOURCES_TO_REPORT": 1,
        "MIN_SOURCE_DISTANCE": 8.0,

        # Localization metric.
        "ACTIVITY_MAX_RELATIVE_DROP": 0.5,

        # Visualization output.
        "PRINT_GENERATED_COUNTS_TABLE": True,
        "UPDATE_FRAME_MAX_VOXELS": 1000000,
        "GIF_FPS": 1.0,
        "AXIS_LIMITS_3D": (
            (-100, 100),
            (-100, 100),
            (-100, 100),
        ),
    }


def build_config(overrides=None):
    """Merge a small override dict into the demo defaults."""
    config = make_default_config()
    if overrides:
        unknown_keys = sorted(set(overrides) - set(config))
        if unknown_keys:
            raise KeyError(f"Unknown config keys: {unknown_keys}")
        config.update(overrides)

    if "SCAN_CENTER" not in (overrides or {}):
        config["SCAN_CENTER"] = tuple(
            (np.array(config["SEARCH_MIN"]) + np.array(config["SEARCH_MAX"])) / 2.0
        )

    return config


def run_experiment(overrides=None):
    """Generate one dataset, run reconstruction, and return metrics."""
    config = build_config(overrides)

    dataset = generate_synthetic_dataset(
        num_measurements=config["NUM_MEASUREMENTS"],
        seed=config["SEED"],
        orientation_mode=config["ORIENTATION_MODE"],
        fixed_view_direction=config["FIXED_VIEW_DIRECTION"],
        source_specs=config["TRUE_SOURCE_SPECS"],
        num_channels=config["NUM_CHANNELS"],
        diode_pitch=config["DIODE_PITCH"],
        background_mean_per_second=config["BACKGROUND_MEAN_PER_SECOND"],
        background_channel_std_per_second=config["BACKGROUND_CHANNEL_STD_PER_SECOND"],
        channel_scale_mean=config["CHANNEL_SCALE_MEAN"],
        channel_scale_std=config["CHANNEL_SCALE_STD"],
        generation_k=config["GENERATION_K"],
        peak_total_source_signal_per_second=config["PEAK_TOTAL_SOURCE_SIGNAL_PER_SECOND"],
        measurement_duration_seconds=config["MEASUREMENT_DURATION_SECONDS"],
        scan_center=config["SCAN_CENTER"],
        trajectory_mode=config["TRAJECTORY_MODE"],
        angular_jitter_deg=config["ANGULAR_JITTER_DEG"],
        lateral_span=config["LATERAL_SPAN"],
        vertical_span=config["VERTICAL_SPAN"],
        base_distance=config["BASE_DISTANCE"],
        distance_span=config["DISTANCE_SPAN"],
        orbit_radius=config["ORBIT_RADIUS"],
        orbit_radius_span=config["ORBIT_RADIUS_SPAN"],
        orbit_angle_span_deg=config["ORBIT_ANGLE_SPAN_DEG"],
        orbit_start_angle_deg=config["ORBIT_START_ANGLE_DEG"],
        orbit_vertical_span=config["ORBIT_VERTICAL_SPAN"],
        fixed_sweep_z_deg=config["FIXED_SWEEP_Z_DEG"],
        fixed_sweep_x_deg=config["FIXED_SWEEP_X_DEG"],
    )

    num_channels = dataset["counts"].shape[1]
    assumed_model = {
        "assumed_backgrounds": np.full(
            num_channels,
            config["ASSUMED_BACKGROUND_MEAN_PER_SECOND"]
            * config["MEASUREMENT_DURATION_SECONDS"],
            dtype=float,
        ),
        "assumed_channel_scales": np.full(
            num_channels,
            config["ASSUMED_CHANNEL_SCALE"],
            dtype=float,
        ),
        "assumed_k": config["ASSUMED_K"],
    }

    result = reconstruct_sources(
        counts=dataset["counts"],
        geometry=dataset["geometry"],
        rotations=dataset["rotations"],
        translations=dataset["translations"],
        **assumed_model,
        search_min=config["SEARCH_MIN"],
        search_max=config["SEARCH_MAX"],
        method=config["reconstruction_method"],
        grid_size=config["GRID_SIZE"],
        num_iterations=config["NUM_ITERATIONS"],
        history_interval=config["HISTORY_INTERVAL"],
        max_sources=config["MAX_SOURCES_TO_REPORT"],
        min_source_distance=config["MIN_SOURCE_DISTANCE"],
    )

    localization_metrics = summarize_activity_localization(
        voxel_positions=result["candidate_positions"],
        activity=result["activity"],
        true_sources=dataset["true_sources"],
        max_relative_drop=config["ACTIVITY_MAX_RELATIVE_DROP"],
    )
    estimated_distance_summary = summarize_estimated_source_distances(
        true_sources=dataset["true_sources"],
        estimated_sources=result["estimated_sources"],
    )

    return {
        "config": config,
        "dataset": dataset,
        "result": result,
        "localization_metrics": localization_metrics,
        "estimated_distance_summary": estimated_distance_summary,
    }


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


def print_reconstruction_summary(method, result, true_sources, relative_drop):
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
    for index, (item, activity_snapshot) in enumerate(
        zip(result["history"], result["activity_history"]),
        start=1,
    ):
        print(f" step={index}")
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {np.round(value, 4).tolist()}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        metrics = summarize_activity_localization(
            voxel_positions=result["candidate_positions"],
            activity=activity_snapshot,
            true_sources=true_sources,
            max_relative_drop=relative_drop,
        )
        print(
            "  "
            f"threshold_metric: selected_voxels={metrics['selected_voxel_count']}, "
            f"mean={metrics['all_source_distance_mean']:.4f} cm, "
            f"wmean={metrics['all_source_distance_weighted_mean']:.4f} cm, "
            f"p90={metrics['all_source_distance_p90']:.4f} cm"
        )


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
    experiment = run_experiment()
    config = experiment["config"]
    dataset = experiment["dataset"]
    result = experiment["result"]

    print_summary(dataset)
    if config["PRINT_GENERATED_COUNTS_TABLE"]:
        print_generated_counts_table(dataset)
    print_reconstruction_summary(
        method=config["reconstruction_method"],
        result=result,
        true_sources=dataset["true_sources"],
        relative_drop=config["ACTIVITY_MAX_RELATIVE_DROP"],
    )
    print_localization_metrics(
        result=result,
        true_sources=dataset["true_sources"],
        relative_drop=config["ACTIVITY_MAX_RELATIVE_DROP"],
    )

    print_estimated_source_distance_summary(experiment["estimated_distance_summary"])

    output_dir = ON_MOCK_DATA_DIR / "multiPoint" / "visualizations"
    history_3d_dir = output_dir / "mlem_activity_history_3d"
    history_3d_dir.mkdir(parents=True, exist_ok=True)
    max_activity_for_visualization = max(
        float(np.max(activity_snapshot))
        for activity_snapshot in result["activity_history"]
    )
    ACTIVITY_COLOR_LIMITS = (0.0, float(np.log1p(max(max_activity_for_visualization, 0.0))))

    visualize_multi_source_scene(
        dataset,
        save_path=output_dir / "multi_source_scene.png",
        show=False,
        axis_limits=config["AXIS_LIMITS_3D"],
    )
    visualize_mlem_activity(
        dataset,
        candidate_positions=result["candidate_positions"],
        activity=result["activity"],
        estimated_sources=result["estimated_sources"],
        save_path=output_dir / "mlem_activity_reconstruction.png",
        show=False,
        max_voxels=config["UPDATE_FRAME_MAX_VOXELS"],
        axis_limits=config["AXIS_LIMITS_3D"],
        highlight_used_trajectory=False,
        activity_color_limits=ACTIVITY_COLOR_LIMITS,
        activity_size_reference=max_activity_for_visualization,
    )
    visualize_history_score(
        result["history"],
        save_path=output_dir / "mlem_score_history.png",
        show=False,
    )

    num_history_frames = len(result["history"])
    for frame_index, (activity_snapshot, history_item) in enumerate(
        zip(result["activity_history"], result["history"]),
        start=1,
    ):
        estimated_sources_step = pick_activity_peaks(
            candidate_positions=result["candidate_positions"],
            activity=activity_snapshot,
            max_sources=config["MAX_SOURCES_TO_REPORT"],
            min_distance=config["MIN_SOURCE_DISTANCE"],
        )
        iteration = int(history_item["iteration"])
        visualize_mlem_activity(
            dataset,
            candidate_positions=result["candidate_positions"],
            activity=activity_snapshot,
            estimated_sources=estimated_sources_step,
            save_path=history_3d_dir / f"iteration_{iteration:06d}.png",
            show=False,
            max_voxels=config["UPDATE_FRAME_MAX_VOXELS"],
            axis_limits=config["AXIS_LIMITS_3D"],
            activity_color_limits=ACTIVITY_COLOR_LIMITS,
            activity_size_reference=max_activity_for_visualization,
            frame_label=f"frame {frame_index}/{num_history_frames} | iteration {iteration}",
            highlight_used_trajectory=False,
        )

    history_gif_path = create_gif_from_frames(
        frames_dir=history_3d_dir,
        output_path=output_dir / "mlem_activity_history_3d.gif",
        pattern="iteration_*.png",
        fps=config["GIF_FPS"],
    )

    print()
    print("Saved plots:")
    print(" ", output_dir / "multi_source_scene.png")
    print(" ", output_dir / "mlem_activity_reconstruction.png")
    print(" ", output_dir / "mlem_score_history.png")
    print(" ", history_3d_dir / "iteration_000001.png ...")
    print(" ", history_gif_path)


if __name__ == "__main__":
    main()
