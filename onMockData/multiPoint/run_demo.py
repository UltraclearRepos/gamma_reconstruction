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
    return {
        # Reconstruction method.
        "reconstruction_method": "mlem",  # Currently available: "mlem".

        # Mock data generation.
        "SEED": 11,  # Random seed for mock data noise, channel scales, backgrounds, and trajectory jitter.
        "NUM_MEASUREMENTS": 30,  # Number of detector positions / generated measurements.
        "TRUE_SOURCE_SPECS": [  # Ground-truth point source positions [cm] and source strengths [/s].
            {"position": (0.0, -5.0, 0.0), "strength_per_second": 5000.0},
            # {"position": (-10.0, -18.0, 16.0), "strength_per_second": 3250.0},
            # {"position": (34.0, -7.0, 38.0), "strength_per_second": 2250.0},
        ],
        "NUM_CHANNELS": 16,  # Number of detector channels.
        "DIODE_PITCH": 0.15875,  # Channel width and center-to-center pitch [cm]; 16 channels span 2.54 cm.
        "MEASUREMENT_DURATION_SECONDS": 1.0,  # Duration of one measurement; rates below are multiplied by this.
        "BACKGROUND_MEAN_PER_SECOND": 1000.0,  # Mean background counts per channel per second.
        "BACKGROUND_CHANNEL_STD_PER_SECOND": 100.0,  # Channel-to-channel background std per second.
        "CHANNEL_SCALE_MEAN": 1.0,  # Mean multiplicative channel sensitivity.
        "CHANNEL_SCALE_STD": 0.08,  # Channel-to-channel sensitivity mismatch std.
        "GENERATION_K": 4.5,  # >= 0; higher = narrower angular response, so off-axis sources give fewer counts.
        "GENERATION_DISTANCE_POWER": 1.0,  # Distance falloff exponent: response ~= 1 / distance^power.

        # Assumed reconstruction model.
        "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,  # Background rate used by reconstruction model.
        "ASSUMED_CHANNEL_SCALE": 1.0,  # Global channel sensitivity used by reconstruction model.
        "ASSUMED_K": 4.5,  # Angular sensitivity exponent used by reconstruction model.
        "ASSUMED_DISTANCE_POWER": 1.0,  # Distance falloff exponent used by reconstruction model.

        # Reconstruction volume.
        "SEARCH_START_DISTANCE": 5.0,  # Start searching this far from each detector channel along detector forward axis [cm].
        "SEARCH_STOP_DISTANCE": 20.0,  # Stop searching this far from each detector channel; window length = stop - start [cm].
        "SEARCH_START_RADIUS": 3.0,  # Search tube/cone radius at SEARCH_START_DISTANCE [cm].
        "SEARCH_STOP_RADIUS": 12.0,  # Search tube/cone radius at SEARCH_STOP_DISTANCE [cm].
        # The dataset derives the axis-aligned reconstruction box from this detector-distance cone/tube.
        "GRID_SIZE": 14,  # ML-EM candidate grid size per axis.

        # Mock detector trajectory.
        "TRAJECTORY_MODE": "around_center",  # "grid", "line", or "around_center".
        "TRAJECTORY_CENTER": (0.0, 0.0, 0.0),  # Reference center of the detector trajectory [cm].
        "LATERAL_SPAN": 15.0,  # grid/line: max left/right offset from TRAJECTORY_CENTER [cm].
        "VERTICAL_SPAN": 0.0,  # grid: max up/down offset; line: fixed Z offset from TRAJECTORY_CENTER [cm].
        "BASE_DISTANCE": 5.0,  # grid/line: nominal detector distance from TRAJECTORY_CENTER [cm].
        "DISTANCE_SPAN": 0.0,  # grid only: sinusoidal variation of detector distance [cm].
        "ORBIT_RADIUS": 20.0,  # around_center only: manual nominal orbit radius around TRAJECTORY_CENTER [cm].
        "ORBIT_RADIUS_SPAN": 0,  # around_center only: sinusoidal radius variation during orbit [cm].
        "ORBIT_ANGLE_SPAN_DEG": 180.0,  # around_center only: covered arc; 180 = half-turn, 360 = full orbit.
        "ORBIT_START_ANGLE_DEG": -60.0,  # around_center only: starting angle in the XY plane.
        "ORBIT_VERTICAL_SPAN": 0.0,  # around_center only: sinusoidal up/down motion during orbit [cm].

        "ORIENTATION_MODE": "towards_target",  # "fixed" or "towards_target"; use "towards_target" to keep looking at TRAJECTORY_CENTER.
        "FIXED_VIEW_DIRECTION": (0, 1, 0),  # fixed mode only: detector forward direction in world coordinates.
        "FIXED_SWEEP_Z_RANGE_DEG": 0.0,  # random left/right angle range around world Z axis, sampled as +/- this value.
        "FIXED_SWEEP_X_RANGE_DEG": 0.0,  # random up/down angle range around world X axis, sampled as +/- this value.
        "ANGULAR_JITTER_DEG": 0.0,  # Random orientation noise per measurement; simulates imperfect manual sensor motion.

        # Reconstruction algorithm.
        "NUM_ITERATIONS": 4000,  # ML-EM iteration count.
        "HISTORY_INTERVAL": 200,  # How often ML-EM stores history snapshots.
        "SPARSITY_LAMBDA": 1.0,  # L1-like activity penalty; increase to suppress diffuse weak voxels.
        "MAX_SOURCES_TO_REPORT": 1,  # Number of strongest separated peaks reported from the activity map.
        "MIN_SOURCE_DISTANCE": 8.0,  # Minimum separation between reported activity peaks [cm].

        # Localization metric.
        "ACTIVITY_MAX_RELATIVE_DROP": 0.5,  # Select voxels with activity >= (1 - drop) * max_activity.

        # Visualization output.
        "PRINT_GENERATED_COUNTS_TABLE": True,  # Print full generated measurement x channel table.
        "UPDATE_FRAME_MAX_VOXELS": 200000,  # Max strongest voxels plotted in one history frame.
        "GIF_FPS": 5.0,  # GIF frame rate for history animation.
        "AXIS_LIMITS_3D": (  # 3D plot axis limits [cm].
            (-30, 30),
            (-30, 30),
            (-30, 30),
        ),
    }


def build_config(overrides=None):
    """Merge a small override dict into the demo defaults."""
    config = make_default_config()
    overrides = overrides or {}
    if overrides:
        unknown_keys = sorted(set(overrides) - set(config))
        if unknown_keys:
            raise KeyError(f"Unknown config keys: {unknown_keys}")
        config.update(overrides)

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
        generation_distance_power=config["GENERATION_DISTANCE_POWER"],
        measurement_duration_seconds=config["MEASUREMENT_DURATION_SECONDS"],
        trajectory_center=config["TRAJECTORY_CENTER"],
        search_start_distance=config["SEARCH_START_DISTANCE"],
        search_stop_distance=config["SEARCH_STOP_DISTANCE"],
        search_start_radius=config["SEARCH_START_RADIUS"],
        search_stop_radius=config["SEARCH_STOP_RADIUS"],
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
        fixed_sweep_z_range_deg=config["FIXED_SWEEP_Z_RANGE_DEG"],
        fixed_sweep_x_range_deg=config["FIXED_SWEEP_X_RANGE_DEG"],
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
        "assumed_distance_power": config["ASSUMED_DISTANCE_POWER"],
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
        search_start_radius=dataset["search_start_radius"],
        search_stop_radius=dataset["search_stop_radius"],
        method=config["reconstruction_method"],
        grid_size=config["GRID_SIZE"],
        num_iterations=config["NUM_ITERATIONS"],
        history_interval=config["HISTORY_INTERVAL"],
        sparsity_lambda=config["SPARSITY_LAMBDA"],
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
    print(
        "Search distance window [cm]: "
        f"{dataset['search_start_distance']:.2f} -> {dataset['search_stop_distance']:.2f} "
        f"(length {dataset['search_distance_window']:.2f})"
    )
    print(
        "Search radius window [cm]:   "
        f"{dataset['search_start_radius']:.2f} -> {dataset['search_stop_radius']:.2f}"
    )
    print(f"Search min [cm]:            {np.round(dataset['search_min'], 4)}")
    print(f"Search max [cm]:            {np.round(dataset['search_max'], 4)}")


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
