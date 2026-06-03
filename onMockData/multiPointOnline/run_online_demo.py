"""
Entrypoint for the online BSREM synthetic multi-point source demo.

This script intentionally skips threshold refinement. It feeds measurements to
online BSREM in small batches and updates the reconstruction after each batch.
"""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ON_MOCK_DATA_DIR = Path(__file__).resolve().parents[1]
if str(ON_MOCK_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(ON_MOCK_DATA_DIR))

from common.forward_model import compute_detector_response_diagnostics
from metrics import (
    print_activity_component_matching_summary,
    print_activity_map_similarity_summary,
    summarize_activity_component_matching,
    summarize_activity_map_similarity,
)
from online_reconstruction_algorithms import (
    iter_measurement_batches,
    reconstruct_sources_online,
)
from simulation import expected_counts_multi_source, generate_synthetic_dataset
from visualization import (
    create_gif_from_frames,
    render_online_refinement_diagnostic_frames,
    render_system_matrix_measurement_3d_frames,
    visualize_activity_map_comparison,
    visualize_detector_response_curves,
    visualize_expected_counts_breakdown,
    visualize_forward_response_components,
    visualize_forward_signal_summary,
    visualize_history_score,
    visualize_mlem_activity,
    visualize_multi_source_scene,
    visualize_system_matrix,
    visualize_system_matrix_sensitivity_3d,
    visualize_true_voxel_activity,
    voxelize_true_source_activity,
)


def make_default_config():
    """Return default settings for the synthetic online multi-point demo."""
    return {
        # Reconstruction method.
        "reconstruction_method": "online_bsrem",  # Available: "online_bsrem".

        # Mock data generation.
        "SEED": 11,  # Random seed for mock data noise, channel scales, backgrounds, and trajectory jitter.
        "NUM_MEASUREMENTS": 100,  # Number of detector positions / generated measurements.
        # "TRUE_SOURCE_SPECS": [  # Ground-truth source centers [cm], source strengths [/s], and Gaussian spread [cm].
        #     {"position": (-6.5, 2.6, -0.8), "strength_per_second": 6000.0, "spread_cm": 1.8},  # lateral, moderately shallow node.
        #     {"position": (4.2, -1.2, 1.3), "strength_per_second": 7000.0, "spread_cm": 1.2},  # deeper central node.
        #     # {"position": (1.5, -5.2, 0.7), "strength_per_second": 6100.0, "spread_cm": 0.7},  # cranial/lateral shallow node.
        # ],
        # "TRUE_SOURCE_SPECS": [
        #     # Lewy płat: głębiej, rozmyty
        #     {"position": (-2.5, 1.5, -1.0), "strength_per_second": 5000.0, "spread_cm": 1.2},
        #     # Prawy płat: odrobinę wyżej, nieco mniejszy
        #     {"position": (2.5, 1.0, 1.0), "strength_per_second": 4500.0, "spread_cm": 1.0},
        #     # "Gorący guzek": bardzo blisko skóry (płytko), mały, wysoka aktywność
        #     {"position": (1.5, 5.0, -0.5), "strength_per_second": 8000.0, "spread_cm": 0.5},
        # ],
        "TRUE_SOURCE_SPECS": [
            # Główne źródło ukryte głęboko w tkance
            {"position": (0.0, -2.0, 0.0), "strength_per_second": 12000.0, "spread_cm": 1.0},
            # Słabe źródło leżące bezpośrednio nad tym głębokim, blisko detektora
            {"position": (0.0, 6.0, 0.0), "strength_per_second": 5000.0, "spread_cm": 0.5},
            # Punkt referencyjny z boku dla oceny ogólnej geometrii
            {"position": (-4.0, 2.0, 2.0), "strength_per_second": 6000.0, "spread_cm": 0.8},
        ],
        # "TRUE_SOURCE_SPECS": [
        #     # Masywne, rozmyte źródło centralne
        #     {"position": (-1.0, 0.0, 0.0), "strength_per_second": 20000.0, "spread_cm": 2.0},
        #     # Bardzo małe, słabe, odseparowane geometrycznie źródło
        #     {"position": (4.0, 4.0, 3.0), "strength_per_second": 5500.0, "spread_cm": 0.5},
        #     # Średnie źródło na "dolnej" krawędzi trajektorii Z
        #     {"position": (2.0, -3.0, -4.0), "strength_per_second": 5000.0, "spread_cm": 0.7},
        # ],
        "NUM_CHANNELS": 16,  # Number of detector channels.
        "DIODE_PITCH": 0.15875,  # Channel width and center-to-center pitch [cm]; 16 channels span 2.54 cm.
        "MEASUREMENT_DURATION_SECONDS": 1.0,  # Duration of one measurement; rates below are multiplied by this.
        "BACKGROUND_MEAN_PER_SECOND": 1000.0,  # Mean background counts per channel per second.
        "BACKGROUND_CHANNEL_STD_PER_SECOND": 10.0,  # Channel-to-channel background std per second.
        "CHANNEL_SCALE_MEAN": 1.0,  # Mean multiplicative channel sensitivity.
        "CHANNEL_SCALE_STD": 0.08,  # Channel-to-channel sensitivity mismatch std.
        "GENERATION_DISTANCE_POWER": 0.0,  # Extra empirical distance falloff; keep 0 with solid angle, test 0-1 if calibration needs more falloff.
        "GENERATION_ANGULAR_PEAK_DEG": 0.0,  # Peak of detector angular acceptance curve [deg]; typical test range: 0-45.
        "GENERATION_ANGULAR_WIDTH_DEG": 20.0,  # Intrinsic angular width before collimator [deg]; broad detector 30-80, narrow crystal response 15-35.
        "GENERATION_COLLIMATOR_HALF_ANGLE_DEG": 15.0,  # Collimator cone half-angle [deg]; narrow probe 8-20, broad/weak collimation 20-35.
        "GENERATION_COLLIMATOR_EDGE_WIDTH_DEG": 3.0,  # Softness of collimator cutoff [deg]; sharp lead collimator 1-4, soft/uncertain 4-8.
        "GENERATION_COLLIMATOR_LEAKAGE": 0.002,  # Side-response floor from imperfect shielding/scatter; ideal 0, realistic test 0.001-0.02.
        "GENERATION_SKIN_THICKNESS_CM": 0.2,  # Skin slab thickness [cm]; typical adult skin: 0.1-0.3, stress-test up to 0.5.
        "GENERATION_SKIN_MU_PER_CM": 0.15,  # Gamma attenuation coefficient for skin/soft tissue [1/cm]; rough low-energy range: 0.05-0.30.
        "GENERATION_DETECTOR_INTRINSIC_EFFICIENCY": 1.0,  # Detection probability scale after entering active material; physical range: 0-1.
        "GENERATION_RESPONSE_SCALE": 10000,  # Calibration scale from source units to counts; increase to amplify signal, usually test 1-1e5.

        # Assumed reconstruction model.
        "ASSUMED_BACKGROUND_MEAN_PER_SECOND": 1000.0,  # Background rate used by reconstruction model.
        "ASSUMED_CHANNEL_SCALE": 1.0,  # Global channel sensitivity used by reconstruction model.
        "ASSUMED_DISTANCE_POWER": 0.0,  # Extra empirical distance falloff; keep 0 with solid angle, test 0-1 if calibration needs more falloff.
        "ASSUMED_ANGULAR_PEAK_DEG": 0.0,  # Peak of detector angular acceptance curve [deg]; typical test range: 0-45.
        "ASSUMED_ANGULAR_WIDTH_DEG": 20.0,  # Intrinsic angular width before collimator [deg]; broad detector 30-80, narrow crystal response 15-35.
        "ASSUMED_COLLIMATOR_HALF_ANGLE_DEG": 15.0,  # Collimator cone half-angle assumed by reconstruction [deg]; try 10-25.
        "ASSUMED_COLLIMATOR_EDGE_WIDTH_DEG": 3.0,  # Assumed cutoff softness [deg]; smaller = stronger angular rejection.
        "ASSUMED_COLLIMATOR_LEAKAGE": 0.002,  # Assumed side-response floor; increase only if measured side lobes are visible.
        "ASSUMED_SKIN_THICKNESS_CM": 0.2,  # Skin thickness assumed by reconstruction [cm]; typical adult skin: 0.1-0.3.
        "ASSUMED_SKIN_MU_PER_CM": 0.15,  # Skin attenuation coefficient assumed by reconstruction [1/cm]; rough low-energy range: 0.05-0.30.
        "ASSUMED_DETECTOR_INTRINSIC_EFFICIENCY": 1.0,  # Detection probability scale assumed by reconstruction; physical range: 0-1.
        "ASSUMED_RESPONSE_SCALE": 10000,  # Calibration scale assumed by reconstruction; keep near GENERATION scale in matched simulations, usually test 1-1e5.

        # Reconstruction volume.
        "SEARCH_RADIUS": 10.0,  # Reconstruction cylinder radius around TRAJECTORY_CENTER [cm].
        "CYLINDER_HEIGHT": 12.0,  # Reconstruction cylinder height along world Z [cm].
        "VOXEL_SIZE_CM": 0.5,  # Reconstruction voxel spacing [cm]; smaller = denser 3D activity grid.

        # Mock detector trajectory.
        "TRAJECTORY_MODE": "around_center_grid",  # "grid", "line", "around_center", or "around_center_grid".
        "TRAJECTORY_CENTER": (0.0, 0.0, 0.0),  # Reference center of the detector trajectory [cm].
        "LATERAL_SPAN": 15.0,  # grid/line: max left/right offset from TRAJECTORY_CENTER [cm].
        "VERTICAL_SPAN": 3.0,  # grid: max up/down offset; line: fixed Z offset from TRAJECTORY_CENTER [cm].
        "BASE_DISTANCE": 5.0,  # grid/line: nominal detector distance from TRAJECTORY_CENTER [cm].
        "DISTANCE_SPAN": 0.2,  # grid only: sinusoidal variation of detector distance [cm].
        "ORBIT_RADIUS": 13.0,  # around_center only: nominal probe path radius; roughly skin radius + 1 cm detector standoff.
        "ORBIT_RADIUS_SPAN": 0.3,  # around_center only: small hand-pressure / neck-curvature distance variation [cm].
        "ORBIT_ANGLE_SPAN_DEG": 180.0,  # around_center modes: arc angle; 180 gives a half-orbit.
        "ORBIT_START_ANGLE_DEG": -90.0,  # around_center only: starting angle in the XY plane.
        "ORBIT_VERTICAL_SPAN": 5.0,  # around_center_grid: half-span between stacked arcs along Z [cm].

        "ORIENTATION_MODE": "towards_target",  # "fixed" or "towards_target"; fixed mimics a probe swept over skin while pointing inward.
        "FIXED_VIEW_DIRECTION": (0, 1, 0),  # fixed mode only: detector forward direction in world coordinates.
        "FIXED_SWEEP_Z_RANGE_DEG": 5.0,  # random left/right angle range around world Z axis, sampled as +/- this value.
        "FIXED_SWEEP_X_RANGE_DEG": 15.0,  # random up/down angle range around world X axis, sampled as +/- this value.
        "ANGULAR_JITTER_DEG": 10.0,  # Random orientation noise per measurement; simulates imperfect manual sensor motion.

        # Online reconstruction algorithm.
        "ONLINE_MEASUREMENT_BATCH_SIZE": 5,  # New detector positions added to online BSREM at once.
        "ONLINE_UPDATES_PER_BATCH": 50,  # BSREM passes over all received measurements after each incoming batch.
        "ONLINE_HISTORY_INTERVAL": 50,  # Store one history snapshot every N online updates.
        "RECONSTRUCTION_PRIORS": [
            {"type": "tv", "beta": 0.01, "epsilon": 0.001},  # Smoothed total variation; encourages compact blobs with sharper boundaries than Gaussian sieve.
            {"type": "l1", "beta": 0.01},  # Adds sparsity pressure; discourages diffuse low activity spread across many voxels.
        ],  # Priors used by online BSREM
        "RECONSTRUCTION_SIEVE_SIGMA_CM": 0.0,  # Optional post-update Gaussian sieve filter; 0 disables it.
        "BSREM_NUM_SUBSETS": 8,  # Ordered subsets over measurements received so far.
        "BSREM_RELAXATION_INITIAL": 1.0,  # Initial BSREM relaxation step.
        "BSREM_RELAXATION_DECAY": 0.01,  # Larger values damp later online BSREM updates more strongly.
        "BSREM_RELAXATION_POWER": 1.0,  # Exponent used by the BSREM relaxation schedule.

        # Optional second reconstruction pass on voxels selected from the thresholded map.
        "ENABLE_THRESHOLD_REFINEMENT": False,  # Online demo keeps refinement disabled; it can be added later.
        "THRESHOLD_REFINEMENT_METHOD": "same",
        "THRESHOLD_REFINEMENT_NUM_ITERATIONS": 100,
        "THRESHOLD_REFINEMENT_HISTORY_INTERVAL": 10,
        "THRESHOLD_REFINEMENT_PRIORS": [],
        "THRESHOLD_REFINEMENT_SIEVE_SIGMA_CM": 0.0,
        "THRESHOLD_REFINEMENT_PADDING_CM": 0.75,

        # Activity-map metrics.
        "ACTIVITY_COMPONENT_THRESHOLD_FRACTION": 0.1,  # Blob threshold as fraction of each map maximum; higher splits weak tails away.
        "ACTIVITY_COMPONENT_MIN_VOXELS": 3,  # Minimum connected voxels needed to report a blob.

        # Visualization output.
        "PRINT_GENERATED_COUNTS_TABLE": False,  # Print full generated measurement x channel table.
        "UPDATE_FRAME_MAX_VOXELS": 1000000000,  # Max strongest voxels plotted in one history frame.
        "GIF_FPS": 1.0,  # GIF frame rate for history animation.
        "GENERATE_SYSTEM_MATRIX_MEASUREMENT_GIF": False,  # Render per-measurement H frames/GIF; slow, useful mainly for forward-model QA.
        "AXIS_LIMITS_3D": (  # 3D plot axis limits [cm].
            (-30, 30),
            (-30, 30),
            (-30, 30),
        ),
    }


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
                pass


def print_summary(dataset):
    """Print a short summary of the generated online demo dataset."""
    counts = dataset["counts"]

    print("=== Synthetic online multi-point source demo ===")
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
    print(f"Search cylinder radius [cm]: {dataset['search_radius']:.2f}")
    print(f"Search cylinder height [cm]: {dataset['cylinder_height']:.2f}")
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


def make_online_config(overrides=None):
    """Merge a small override dict into the online demo defaults."""
    config = make_default_config()
    overrides = overrides or {}
    if overrides:
        unknown_keys = sorted(set(overrides) - set(config))
        if unknown_keys:
            raise KeyError(f"Unknown config keys: {unknown_keys}")
        config.update(overrides)

    return config


def generate_dataset_from_config(config):
    """Generate the synthetic dataset described by a demo config dictionary."""
    return generate_synthetic_dataset(
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
        generation_distance_power=config["GENERATION_DISTANCE_POWER"],
        generation_angular_peak_angle_deg=config["GENERATION_ANGULAR_PEAK_DEG"],
        generation_angular_width_deg=config["GENERATION_ANGULAR_WIDTH_DEG"],
        generation_collimator_half_angle_deg=config["GENERATION_COLLIMATOR_HALF_ANGLE_DEG"],
        generation_collimator_edge_width_deg=config["GENERATION_COLLIMATOR_EDGE_WIDTH_DEG"],
        generation_collimator_leakage=config["GENERATION_COLLIMATOR_LEAKAGE"],
        generation_skin_thickness_cm=config["GENERATION_SKIN_THICKNESS_CM"],
        generation_skin_mu_per_cm=config["GENERATION_SKIN_MU_PER_CM"],
        generation_detector_intrinsic_efficiency=config["GENERATION_DETECTOR_INTRINSIC_EFFICIENCY"],
        generation_response_scale=config["GENERATION_RESPONSE_SCALE"],
        measurement_duration_seconds=config["MEASUREMENT_DURATION_SECONDS"],
        trajectory_center=config["TRAJECTORY_CENTER"],
        search_radius=config["SEARCH_RADIUS"],
        cylinder_height=config["CYLINDER_HEIGHT"],
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


def assumed_model_from_config(config, num_channels):
    """Build the reconstruction-side detector model from config values."""
    return {
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
        "assumed_distance_power": config["ASSUMED_DISTANCE_POWER"],
        "assumed_angular_peak_angle_deg": config["ASSUMED_ANGULAR_PEAK_DEG"],
        "assumed_angular_width_deg": config["ASSUMED_ANGULAR_WIDTH_DEG"],
        "assumed_collimator_half_angle_deg": config["ASSUMED_COLLIMATOR_HALF_ANGLE_DEG"],
        "assumed_collimator_edge_width_deg": config["ASSUMED_COLLIMATOR_EDGE_WIDTH_DEG"],
        "assumed_collimator_leakage": config["ASSUMED_COLLIMATOR_LEAKAGE"],
        "assumed_skin_thickness_cm": config["ASSUMED_SKIN_THICKNESS_CM"],
        "assumed_skin_mu_per_cm": config["ASSUMED_SKIN_MU_PER_CM"],
        "assumed_detector_intrinsic_efficiency": config["ASSUMED_DETECTOR_INTRINSIC_EFFICIENCY"],
        "assumed_response_scale": config["ASSUMED_RESPONSE_SCALE"],
    }


def run_online_experiment(overrides=None):
    """Generate one dataset, run online BSREM, and return metrics."""
    config = make_online_config(overrides)
    dataset = generate_dataset_from_config(config)
    num_channels = dataset["counts"].shape[1]
    assumed_model = assumed_model_from_config(config, num_channels)
    measurement_batches = iter_measurement_batches(
        counts=dataset["counts"],
        rotations=dataset["rotations"],
        translations=dataset["translations"],
        batch_size=config["ONLINE_MEASUREMENT_BATCH_SIZE"],
    )

    reconstruction_start_time = perf_counter()
    result = reconstruct_sources_online(
        method=config["reconstruction_method"],
        measurement_batches=measurement_batches,
        geometry=dataset["geometry"],
        **assumed_model,
        trajectory_center=dataset["trajectory_center"],
        search_radius=dataset["search_radius"],
        cylinder_height=dataset["cylinder_height"],
        voxel_size_cm=config["VOXEL_SIZE_CM"],
        updates_per_batch=config["ONLINE_UPDATES_PER_BATCH"],
        history_interval=config["ONLINE_HISTORY_INTERVAL"],
        reconstruction_priors=config["RECONSTRUCTION_PRIORS"],
        reconstruction_sieve_sigma_cm=config["RECONSTRUCTION_SIEVE_SIGMA_CM"],
        num_subsets=config["BSREM_NUM_SUBSETS"],
        relaxation_initial=config["BSREM_RELAXATION_INITIAL"],
        relaxation_decay=config["BSREM_RELAXATION_DECAY"],
        relaxation_power=config["BSREM_RELAXATION_POWER"],
    )
    reconstruction_elapsed_seconds = perf_counter() - reconstruction_start_time

    true_voxel_activity = voxelize_true_source_activity(
        candidate_positions=result["candidate_positions"],
        true_sources=dataset["true_sources"],
    )
    activity_map_similarity = summarize_activity_map_similarity(
        true_activity=true_voxel_activity,
        reconstructed_activity=result["activity"],
    )
    activity_component_matching = summarize_activity_component_matching(
        true_activity=true_voxel_activity,
        reconstructed_activity=result["activity"],
        candidate_positions=result["candidate_positions"],
        candidate_flat_indices=result["candidate_flat_indices"],
        grid_shape=result["grid_shape"],
        true_sources=dataset["true_sources"],
        threshold_fraction=config["ACTIVITY_COMPONENT_THRESHOLD_FRACTION"],
        min_component_voxels=config["ACTIVITY_COMPONENT_MIN_VOXELS"],
    )

    return {
        "config": config,
        "dataset": dataset,
        "result": result,
        "reconstruction_elapsed_seconds": reconstruction_elapsed_seconds,
        "true_voxel_activity": true_voxel_activity,
        "activity_map_similarity": activity_map_similarity,
        "activity_component_matching": activity_component_matching,
    }


def main():
    experiment = run_online_experiment()
    config = experiment["config"]
    dataset = experiment["dataset"]
    result = experiment["result"]

    print_summary(dataset)
    if config["PRINT_GENERATED_COUNTS_TABLE"]:
        print_generated_counts_table(dataset)

    print()
    print("=== Online BSREM reconstruction ===")
    print(f"Batch size [measurements]: {config['ONLINE_MEASUREMENT_BATCH_SIZE']}")
    print(f"Updates per batch:         {config['ONLINE_UPDATES_PER_BATCH']}")
    print(f"History interval:          {config['ONLINE_HISTORY_INTERVAL']}")
    print(f"Reconstruction time [s]:   {experiment['reconstruction_elapsed_seconds']:.3f}")
    print(f"Reconstruction device:     {result['torch_device']}")

    online = result.get("online", {})
    print(
        "Online data received:      "
        f"{online.get('received_measurements', 0)} measurements in "
        f"{online.get('measurement_batches', 0)} batches"
    )

    timings = result.get("timings", {})
    if timings:
        print(
            "Reconstruction timing [s]: "
            f"H={timings['system_matrix_seconds']:.3f}, "
            f"transfer={timings['transfer_seconds']:.3f}, "
            f"updates={timings['online_bsrem_iterations_seconds']:.3f}"
        )

    print_activity_map_similarity_summary(experiment["activity_map_similarity"])
    print_activity_component_matching_summary(experiment["activity_component_matching"])

    output_dir = ON_MOCK_DATA_DIR / "multiPointOnline" / "visualizations"
    forward_model_dir = output_dir / "forward_model"
    thresholded_history_dir = output_dir / "online_thresholded_activity_history_3d"
    accepted_components_history_dir = output_dir / "online_accepted_components_history_3d"
    h_frames_3d_dir = forward_model_dir / "system_matrix_H_measurements_3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    forward_model_dir.mkdir(parents=True, exist_ok=True)
    thresholded_history_dir.mkdir(parents=True, exist_ok=True)
    accepted_components_history_dir.mkdir(parents=True, exist_ok=True)
    h_frames_3d_dir.mkdir(parents=True, exist_ok=True)

    max_activity_for_visualization = max(
        float(np.max(activity_snapshot))
        for activity_snapshot in result["activity_history"]
    )
    activity_color_limits = (0.0, float(np.log1p(max(max_activity_for_visualization, 0.0))))
    true_voxel_activity = experiment["true_voxel_activity"]

    visualize_multi_source_scene(
        dataset,
        save_path=output_dir / "multi_source_scene.png",
        show=False,
        axis_limits=config["AXIS_LIMITS_3D"],
    )
    visualize_true_voxel_activity(
        dataset,
        candidate_positions=result["candidate_positions"],
        true_activity=true_voxel_activity,
        save_path=output_dir / "true_voxel_activity.png",
        show=False,
        max_voxels=config["UPDATE_FRAME_MAX_VOXELS"],
        axis_limits=config["AXIS_LIMITS_3D"],
    )
    visualize_activity_map_comparison(
        dataset,
        candidate_positions=result["candidate_positions"],
        true_activity=true_voxel_activity,
        reconstructed_activity=result["activity"],
        component_matching=experiment["activity_component_matching"],
        voxel_size_cm=result["voxel_size_cm"],
        save_path=output_dir / "activity_map_comparison_online.png",
        show=False,
        max_voxels=config["UPDATE_FRAME_MAX_VOXELS"],
        axis_limits=config["AXIS_LIMITS_3D"],
    )
    visualize_detector_response_curves(
        angular_peak_angle_deg=config["GENERATION_ANGULAR_PEAK_DEG"],
        angular_width_deg=config["GENERATION_ANGULAR_WIDTH_DEG"],
        collimator_half_angle_deg=config["GENERATION_COLLIMATOR_HALF_ANGLE_DEG"],
        collimator_edge_width_deg=config["GENERATION_COLLIMATOR_EDGE_WIDTH_DEG"],
        collimator_leakage=config["GENERATION_COLLIMATOR_LEAKAGE"],
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
    visualize_system_matrix(
        result["system_matrix"],
        num_measurements=dataset["counts"].shape[0],
        num_channels=dataset["counts"].shape[1],
        save_path=forward_model_dir / "system_matrix_H.png",
        show=False,
    )
    visualize_system_matrix_sensitivity_3d(
        dataset,
        candidate_positions=result["candidate_positions"],
        system_matrix=result["system_matrix"],
        save_path=forward_model_dir / "system_matrix_H_sensitivity_3d.png",
        show=False,
        max_voxels=config["UPDATE_FRAME_MAX_VOXELS"],
        axis_limits=config["AXIS_LIMITS_3D"],
    )
    if config["GENERATE_SYSTEM_MATRIX_MEASUREMENT_GIF"]:
        clear_existing_frames(h_frames_3d_dir, "measurement_3d_*.png")
        render_system_matrix_measurement_3d_frames(
            dataset,
            candidate_positions=result["candidate_positions"],
            system_matrix=result["system_matrix"],
            num_measurements=dataset["counts"].shape[0],
            num_channels=dataset["counts"].shape[1],
            frames_dir=h_frames_3d_dir,
            max_voxels=config["UPDATE_FRAME_MAX_VOXELS"],
            axis_limits=config["AXIS_LIMITS_3D"],
        )
        create_gif_from_frames(
            frames_dir=h_frames_3d_dir,
            output_path=forward_model_dir / "system_matrix_H_3d_by_measurement.gif",
            pattern="measurement_3d_*.png",
            fps=config["GIF_FPS"],
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
            distance_power=config["GENERATION_DISTANCE_POWER"],
            angular_peak_angle_deg=config["GENERATION_ANGULAR_PEAK_DEG"],
            angular_width_deg=config["GENERATION_ANGULAR_WIDTH_DEG"],
            collimator_half_angle_deg=config["GENERATION_COLLIMATOR_HALF_ANGLE_DEG"],
            collimator_edge_width_deg=config["GENERATION_COLLIMATOR_EDGE_WIDTH_DEG"],
            collimator_leakage=config["GENERATION_COLLIMATOR_LEAKAGE"],
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
            distance_power=config["GENERATION_DISTANCE_POWER"],
            angular_peak_angle_deg=config["GENERATION_ANGULAR_PEAK_DEG"],
            angular_width_deg=config["GENERATION_ANGULAR_WIDTH_DEG"],
            collimator_half_angle_deg=config["GENERATION_COLLIMATOR_HALF_ANGLE_DEG"],
            collimator_edge_width_deg=config["GENERATION_COLLIMATOR_EDGE_WIDTH_DEG"],
            collimator_leakage=config["GENERATION_COLLIMATOR_LEAKAGE"],
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
        save_path=output_dir / "online_activity_reconstruction.png",
        show=False,
        max_voxels=config["UPDATE_FRAME_MAX_VOXELS"],
        axis_limits=config["AXIS_LIMITS_3D"],
        highlight_used_trajectory=False,
        activity_color_limits=activity_color_limits,
        activity_size_reference=max_activity_for_visualization,
    )
    visualize_history_score(
        result["history"],
        save_path=output_dir / "online_score_history.png",
        show=False,
    )

    clear_existing_frames(thresholded_history_dir, "online_refinement_diagnostic_*.png")
    render_online_refinement_diagnostic_frames(
        dataset=dataset,
        candidate_positions=result["candidate_positions"],
        true_activity=true_voxel_activity,
        candidate_flat_indices=result["candidate_flat_indices"],
        grid_shape=result["grid_shape"],
        online_activity_history=result["activity_history"],
        online_history=result["history"],
        threshold_fraction=config["ACTIVITY_COMPONENT_THRESHOLD_FRACTION"],
        min_component_voxels=1,
        frames_dir=thresholded_history_dir,
        refinement_events=[],
        axis_limits=config["AXIS_LIMITS_3D"],
        panel_mode="thresholded",
    )
    create_gif_from_frames(
        frames_dir=thresholded_history_dir,
        output_path=output_dir / "online_thresholded_activity_history_3d.gif",
        pattern="online_refinement_diagnostic_*.png",
        fps=config["GIF_FPS"],
    )
    clear_existing_frames(accepted_components_history_dir, "online_refinement_diagnostic_*.png")
    render_online_refinement_diagnostic_frames(
        dataset=dataset,
        candidate_positions=result["candidate_positions"],
        true_activity=true_voxel_activity,
        candidate_flat_indices=result["candidate_flat_indices"],
        grid_shape=result["grid_shape"],
        online_activity_history=result["activity_history"],
        online_history=result["history"],
        threshold_fraction=config["ACTIVITY_COMPONENT_THRESHOLD_FRACTION"],
        min_component_voxels=config["ACTIVITY_COMPONENT_MIN_VOXELS"],
        frames_dir=accepted_components_history_dir,
        refinement_events=[],
        axis_limits=config["AXIS_LIMITS_3D"],
        panel_mode="accepted_components",
    )
    create_gif_from_frames(
        frames_dir=accepted_components_history_dir,
        output_path=output_dir / "online_accepted_components_history_3d.gif",
        pattern="online_refinement_diagnostic_*.png",
        fps=config["GIF_FPS"],
    )

    print()
    print(f"Saved online visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
