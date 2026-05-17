"""
Entrypoint for the synthetic multi-point source demo.
"""

from pathlib import Path
import sys

import numpy as np

ON_MOCK_DATA_DIR = Path(__file__).resolve().parents[1]
if str(ON_MOCK_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(ON_MOCK_DATA_DIR))

from common.forward_model import compute_detector_response_diagnostics
from reconstruction_algorithms import pick_activity_peaks, reconstruct_sources
from metrics import (
    print_activity_localization_summary,
    print_estimated_source_distance_summary,
    summarize_activity_localization,
    summarize_estimated_source_distances,
)
from simulation import expected_counts_multi_source, generate_synthetic_dataset
from visualization import (
    create_gif_from_frames,
    visualize_detector_response_curves,
    visualize_expected_counts_breakdown,
    visualize_forward_response_components,
    visualize_forward_signal_summary,
    visualize_history_score,
    visualize_mlem_activity,
    visualize_multi_source_scene,
)


def make_default_config():
    """Return default settings for the synthetic multi-point demo."""
    return {
        # Reconstruction method.
        "reconstruction_method": "mlem",  # Available: "mlem".

        # Mock data generation.
        "SEED": 11,  # Random seed for mock data noise, channel scales, backgrounds, and trajectory jitter.
        "NUM_MEASUREMENTS": 42,  # Number of detector positions / generated measurements.
        "TRUE_SOURCE_SPECS": [  # Ground-truth source centers [cm], source strengths [/s], and Gaussian spread [cm].
            {"position": (-2.4, -1.7, -1.0), "strength_per_second": 5200.0, "spread_cm": 0.45},  # lateral, moderately shallow node.
            {"position": (0.7, -2.8, 0.8), "strength_per_second": 4200.0, "spread_cm": 0.65},  # deeper central node.
            {"position": (2.9, -1.2, 1.7), "strength_per_second": 6100.0, "spread_cm": 0.55},  # cranial/lateral shallow node.
        ],
        "NUM_CHANNELS": 16,  # Number of detector channels.
        "DIODE_PITCH": 0.15875,  # Channel width and center-to-center pitch [cm]; 16 channels span 2.54 cm.
        "MEASUREMENT_DURATION_SECONDS": 1.0,  # Duration of one measurement; rates below are multiplied by this.
        "BACKGROUND_MEAN_PER_SECOND": 1000.0,  # Mean background counts per channel per second.
        "BACKGROUND_CHANNEL_STD_PER_SECOND": 100.0,  # Channel-to-channel background std per second.
        "CHANNEL_SCALE_MEAN": 1.0,  # Mean multiplicative channel sensitivity.
        "CHANNEL_SCALE_STD": 0.08,  # Channel-to-channel sensitivity mismatch std.
        "GENERATION_K": 4.5,  # Legacy cos^k sharpness used only when ANGULAR_WIDTH_DEG is None; useful range: 1-12.
        "GENERATION_DISTANCE_POWER": 0.0,  # Extra empirical distance falloff; keep 0 with solid angle, test 0-1 if calibration needs more falloff.
        "GENERATION_ANGULAR_PEAK_DEG": 0.0,  # Peak of detector angular acceptance curve [deg]; typical test range: 0-45.
        "GENERATION_ANGULAR_WIDTH_DEG": 45.0,  # Width of angular acceptance curve [deg]; broad probe 30-80, collimated probe 5-30.
        "GENERATION_SKIN_THICKNESS_CM": 0.2,  # Skin slab thickness [cm]; typical adult skin: 0.1-0.3, stress-test up to 0.5.
        "GENERATION_SKIN_MU_PER_CM": 0.15,  # Gamma attenuation coefficient for skin/soft tissue [1/cm]; rough low-energy range: 0.05-0.30.
        "GENERATION_DETECTOR_INTRINSIC_EFFICIENCY": 1.0,  # Detection probability scale after entering active material; physical range: 0-1.
        "GENERATION_RESPONSE_SCALE": 10000,  # Calibration scale from source units to counts; increase to amplify signal, usually test 1-1e5.

        # Assumed reconstruction model.
        "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,  # Background rate used by reconstruction model.
        "ASSUMED_CHANNEL_SCALE": 1.0,  # Global channel sensitivity used by reconstruction model.
        "ASSUMED_K": 4.5,  # Legacy cos^k sharpness used only when ANGULAR_WIDTH_DEG is None; useful range: 1-12.
        "ASSUMED_DISTANCE_POWER": 0.0,  # Extra empirical distance falloff; keep 0 with solid angle, test 0-1 if calibration needs more falloff.
        "ASSUMED_ANGULAR_PEAK_DEG": 0.0,  # Peak of detector angular acceptance curve [deg]; typical test range: 0-45.
        "ASSUMED_ANGULAR_WIDTH_DEG": 45.0,  # Width of angular acceptance curve [deg]; broad probe 30-80, collimated probe 5-30.
        "ASSUMED_SKIN_THICKNESS_CM": 0.2,  # Skin thickness assumed by reconstruction [cm]; typical adult skin: 0.1-0.3.
        "ASSUMED_SKIN_MU_PER_CM": 0.15,  # Skin attenuation coefficient assumed by reconstruction [1/cm]; rough low-energy range: 0.05-0.30.
        "ASSUMED_DETECTOR_INTRINSIC_EFFICIENCY": 1.0,  # Detection probability scale assumed by reconstruction; physical range: 0-1.
        "ASSUMED_RESPONSE_SCALE": 10000,  # Calibration scale assumed by reconstruction; keep near GENERATION scale in matched simulations, usually test 1-1e5.

        # Reconstruction volume.
        "SEARCH_START_DISTANCE": 4.0,  # Start searching this far from each detector channel along detector forward axis [cm].
        "SEARCH_STOP_DISTANCE": 9.0,  # Stop searching this far from each detector channel; window length = stop - start [cm].
        "SEARCH_START_RADIUS": 1.5,  # Search tube/cone radius at SEARCH_START_DISTANCE [cm].
        "SEARCH_STOP_RADIUS": 5.0,  # Search tube/cone radius at SEARCH_STOP_DISTANCE [cm].
        # The dataset derives the axis-aligned reconstruction box from this detector-distance cone/tube.
        "GRID_SIZE": 10,  # ML-EM candidate grid size per axis.

        # Mock detector trajectory.
        "TRAJECTORY_MODE": "grid",  # "grid", "line", or "around_center"; here it approximates a raster ultrasound probe sweep over skin.
        "TRAJECTORY_CENTER": (0.0, -2.0, 0.3),  # Reference center of the detector trajectory [cm].
        "LATERAL_SPAN": 5.0,  # grid/line: max left/right offset from TRAJECTORY_CENTER [cm].
        "VERTICAL_SPAN": 3.0,  # grid: max up/down offset; line: fixed Z offset from TRAJECTORY_CENTER [cm].
        "BASE_DISTANCE": 5.2,  # grid/line: nominal detector distance from TRAJECTORY_CENTER [cm].
        "DISTANCE_SPAN": 0.2,  # grid only: sinusoidal variation of detector distance [cm].
        "ORBIT_RADIUS": 6.2,  # around_center only: nominal probe path radius; roughly skin radius + 1 cm detector standoff.
        "ORBIT_RADIUS_SPAN": 0.4,  # around_center only: small hand-pressure / neck-curvature distance variation [cm].
        "ORBIT_ANGLE_SPAN_DEG": 125.0,  # around_center only: partial sweep over one side of the neck.
        "ORBIT_START_ANGLE_DEG": -112.0,  # around_center only: starting angle in the XY plane.
        "ORBIT_VERTICAL_SPAN": 0.3,  # around_center only: mild cranio-caudal probe motion during the sweep [cm].

        "ORIENTATION_MODE": "fixed",  # "fixed" or "towards_target"; fixed mimics a probe swept over skin while pointing inward.
        "FIXED_VIEW_DIRECTION": (0, 1, 0),  # fixed mode only: detector forward direction in world coordinates.
        "FIXED_SWEEP_Z_RANGE_DEG": 0.0,  # random left/right angle range around world Z axis, sampled as +/- this value.
        "FIXED_SWEEP_X_RANGE_DEG": 0.0,  # random up/down angle range around world X axis, sampled as +/- this value.
        "ANGULAR_JITTER_DEG": 1.0,  # Random orientation noise per measurement; simulates imperfect manual sensor motion.

        # Reconstruction algorithm.
        "NUM_ITERATIONS": 120,  # ML-EM iteration count.
        "HISTORY_INTERVAL": 60,  # How often ML-EM stores history snapshots.
        "SPARSITY_LAMBDA": 0.2,  # L1-like activity penalty; increase to suppress diffuse weak voxels.
        "MAX_SOURCES_TO_REPORT": 3,  # Number of strongest separated peaks reported from the activity map.
        "MIN_SOURCE_DISTANCE": 1.5,  # Minimum separation between reported activity peaks [cm].
        "PEAK_CENTROID_RADIUS": 2.5,  # Weighted centroid radius for diffuse activity blobs; 0 reports the peak voxel itself.

        # Localization metric.
        "ACTIVITY_MAX_RELATIVE_DROP": 0.5,  # Select voxels with activity >= (1 - drop) * max_activity.

        # Visualization output.
        "PRINT_GENERATED_COUNTS_TABLE": True,  # Print full generated measurement x channel table.
        "UPDATE_FRAME_MAX_VOXELS": 200000,  # Max strongest voxels plotted in one history frame.
        "GIF_FPS": 1.0,  # GIF frame rate for history animation.
        "AXIS_LIMITS_3D": (  # 3D plot axis limits [cm].
            (-10, 10),
            (-10, 8),
            (-7, 7),
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


def clear_existing_frames(frames_dir, pattern):
    """Remove stale generated frames before writing a new animation."""
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        return
    for frame_path in frames_dir.glob(pattern):
        if frame_path.is_file():
            try:
                frame_path.unlink()
            except PermissionError:
                # The desktop preview can keep old PNGs open on Windows.
                # Leaving a locked frame in place is better than aborting the run.
                pass


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
        generation_angular_peak_angle_deg=config["GENERATION_ANGULAR_PEAK_DEG"],
        generation_angular_width_deg=config["GENERATION_ANGULAR_WIDTH_DEG"],
        generation_skin_thickness_cm=config["GENERATION_SKIN_THICKNESS_CM"],
        generation_skin_mu_per_cm=config["GENERATION_SKIN_MU_PER_CM"],
        generation_detector_intrinsic_efficiency=config["GENERATION_DETECTOR_INTRINSIC_EFFICIENCY"],
        generation_response_scale=config["GENERATION_RESPONSE_SCALE"],
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
        "assumed_angular_peak_angle_deg": config["ASSUMED_ANGULAR_PEAK_DEG"],
        "assumed_angular_width_deg": config["ASSUMED_ANGULAR_WIDTH_DEG"],
        "assumed_skin_thickness_cm": config["ASSUMED_SKIN_THICKNESS_CM"],
        "assumed_skin_mu_per_cm": config["ASSUMED_SKIN_MU_PER_CM"],
        "assumed_detector_intrinsic_efficiency": config["ASSUMED_DETECTOR_INTRINSIC_EFFICIENCY"],
        "assumed_response_scale": config["ASSUMED_RESPONSE_SCALE"],
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
        peak_centroid_radius=config["PEAK_CENTROID_RADIUS"],
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
            f"spread_cm={source.get('spread_cm', 0.0):.2f}, "
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
    if "mean_counts" in dataset:
        expected_source_signal = (
            np.asarray(dataset["mean_counts"], dtype=float)
            - np.asarray(dataset["true_backgrounds"], dtype=float)[np.newaxis, :]
        )
        background_mean = max(float(dataset["background_mean"]), 1e-12)
        background_std = max(float(dataset["background_channel_std"]), 1e-12)
        print(
            "Expected source signal:    "
            f"{expected_source_signal.min():.4f} / "
            f"{expected_source_signal.mean():.4f} / "
            f"{expected_source_signal.max():.4f} "
            "(min/mean/max counts)"
        )
        print(
            "Signal/background ratio:   "
            f"mean={expected_source_signal.mean() / background_mean:.6f}, "
            f"max={expected_source_signal.max() / background_mean:.6f}"
        )
        print(
            "Signal/channel-std ratio:  "
            f"mean={expected_source_signal.mean() / background_std:.6f}, "
            f"max={expected_source_signal.max() / background_std:.6f}"
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
    forward_model_dir = output_dir / "forward_model"
    history_3d_dir = output_dir / "mlem_activity_history_3d"
    forward_model_dir.mkdir(parents=True, exist_ok=True)
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
    visualize_detector_response_curves(
        k=config["GENERATION_K"],
        angular_peak_angle_deg=config["GENERATION_ANGULAR_PEAK_DEG"],
        angular_width_deg=config["GENERATION_ANGULAR_WIDTH_DEG"],
        skin_thickness_cm=config["GENERATION_SKIN_THICKNESS_CM"],
        skin_mu_per_cm=config["GENERATION_SKIN_MU_PER_CM"],
        save_path=forward_model_dir / "detector_response_curves.png",
        show=False,
    )
    visualize_expected_counts_breakdown(
        dataset,
        save_path=forward_model_dir / "expected_counts_breakdown.png",
        show=False,
    )

    source_signal_maps = []
    source_labels = []
    for source_index, source in enumerate(dataset["true_sources"], start=1):
        diagnostics = compute_detector_response_diagnostics(
            source_position=source["position"],
            geometry=dataset["geometry"],
            rotations=dataset["rotations"],
            translations=dataset["translations"],
            channel_scales=dataset["true_channel_scales"],
            k=config["GENERATION_K"],
            distance_power=config["GENERATION_DISTANCE_POWER"],
            angular_peak_angle_deg=config["GENERATION_ANGULAR_PEAK_DEG"],
            angular_width_deg=config["GENERATION_ANGULAR_WIDTH_DEG"],
            skin_thickness_cm=config["GENERATION_SKIN_THICKNESS_CM"],
            skin_mu_per_cm=config["GENERATION_SKIN_MU_PER_CM"],
            detector_intrinsic_efficiency=config["GENERATION_DETECTOR_INTRINSIC_EFFICIENCY"],
            response_scale=config["GENERATION_RESPONSE_SCALE"],
        )
        source_label = f"S{source_index}"
        source_signal = expected_counts_multi_source(
            sources=[source],
            geometry=dataset["geometry"],
            rotations=dataset["rotations"],
            translations=dataset["translations"],
            channel_scales=dataset["true_channel_scales"],
            backgrounds=np.zeros_like(dataset["true_backgrounds"]),
            k=config["GENERATION_K"],
            distance_power=config["GENERATION_DISTANCE_POWER"],
            angular_peak_angle_deg=config["GENERATION_ANGULAR_PEAK_DEG"],
            angular_width_deg=config["GENERATION_ANGULAR_WIDTH_DEG"],
            skin_thickness_cm=config["GENERATION_SKIN_THICKNESS_CM"],
            skin_mu_per_cm=config["GENERATION_SKIN_MU_PER_CM"],
            detector_intrinsic_efficiency=config["GENERATION_DETECTOR_INTRINSIC_EFFICIENCY"],
            response_scale=config["GENERATION_RESPONSE_SCALE"],
        )
        source_signal_maps.append(source_signal)
        source_labels.append(source_label)
        visualize_forward_response_components(
            diagnostics=diagnostics,
            source_label=source_label,
            save_path=forward_model_dir / f"response_components_{source_label}.png",
            show=False,
        )

    visualize_forward_signal_summary(
        source_signal_maps=source_signal_maps,
        source_labels=source_labels,
        save_path=forward_model_dir / "source_signal_summary.png",
        show=False,
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
    clear_existing_frames(history_3d_dir, "iteration_*.png")
    for frame_index, (activity_snapshot, history_item) in enumerate(
        zip(result["activity_history"], result["history"]),
        start=1,
    ):
        iteration = int(history_item["iteration"])
        estimated_sources_step = pick_activity_peaks(
            candidate_positions=result["candidate_positions"],
            activity=activity_snapshot,
            max_sources=config["MAX_SOURCES_TO_REPORT"],
            min_distance=config["MIN_SOURCE_DISTANCE"],
            centroid_radius=config["PEAK_CENTROID_RADIUS"],
        )
        frame_label = f"frame {frame_index}/{num_history_frames} | iteration {iteration}"

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
            frame_label=frame_label,
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
    print(" ", forward_model_dir / "detector_response_curves.png")
    print(" ", forward_model_dir / "expected_counts_breakdown.png")
    print(" ", forward_model_dir / "response_components_S1.png ...")
    print(" ", forward_model_dir / "source_signal_summary.png")
    print(" ", output_dir / "mlem_activity_reconstruction.png")
    print(" ", output_dir / "mlem_score_history.png")
    print(" ", history_3d_dir / "iteration_000001.png ...")
    print(" ", history_gif_path)


if __name__ == "__main__":
    main()
