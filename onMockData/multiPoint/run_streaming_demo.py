"""
Entrypoint for streaming ML-EM reconstruction on synthetic multi-point data.
"""

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
    visualize_streaming_best_position,
    visualize_streaming_localization_metrics,
)
from reconstruction_algorithms import pick_activity_peaks
from metrics import print_activity_localization_summary


def print_dataset_summary(dataset):
    """Print short summary for the generated multi-point dataset."""
    counts = dataset["counts"]
    print("=== Streaming multi-point source demo ===")
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
    print(f"Background mean:            {dataset['background_mean']:.1f}")


def print_streaming_summary(result):
    """Print prediction after each streaming update."""
    print()
    print("Streaming updates:")
    for item in result["stream_history"]:
        best_position = np.round(item["best_position"], 4)
        print(
            " "
            f"used_measurements={int(item['used_measurements'])}, "
            f"best_position={best_position}, "
            f"best_activity={item['best_activity']:.2f}, "
            f"score={item['score']:.2f}"
        )

    print()
    print("Final estimated activity peaks:")
    for index, source in enumerate(result["estimated_sources"], start=1):
        print(
            " "
            f"P{index}: position={np.round(source['position'], 4)}, "
            f"activity={source['activity']:.2f}"
        )


def print_streaming_localization_metrics(history):
    """Print localization metrics saved after each streaming update."""
    print()
    print("Localization metrics after each update:")
    for item in history:
        metrics = item["localization_metrics"]
        print(
            " "
            f"used_measurements={int(item['used_measurements'])}, "
            f"selected_voxels={metrics['selected_voxel_count']}, "
            f"voxel_source_mean={metrics['all_source_distance_mean']:.4f} m, "
            f"voxel_source_wmean={metrics['all_source_distance_weighted_mean']:.4f} m, "
            f"voxel_source_p90={metrics['all_source_distance_p90']:.4f} m"
        )


def main():
    # Mock data generation.
    SEED = 11
    NUM_MEASUREMENTS = 50
    TRUE_SOURCE_SPECS = [
        {"position": (0.18, -0.12, 0.26), "relative_strength": 1.0},
        {"position": (-0.10, -0.18, 0.16), "relative_strength": 0.65},
        {"position": (0.34, -0.07, 0.38), "relative_strength": 0.45},
    ]
    NUM_CHANNELS = 16
    DIODE_PITCH = 0.006
    BACKGROUND_MEAN = 1000.0
    BACKGROUND_CHANNEL_STD = 50.0
    CHANNEL_SCALE_MEAN = 1.0
    CHANNEL_SCALE_STD = 0.08
    GENERATION_K = 4.5
    PEAK_TOTAL_SOURCE_SIGNAL = 5000.0

    # Mock detector trajectory.
    ORIENTATION_MODE = "fixed"  # "fixed" or "towards_target"
    FIXED_VIEW_DIRECTION = (0, 1, 0)  # e.g. (1, 0, 0) to "look right" in world coords
    SCAN_CENTER = None  # None means centroid of TRUE_SOURCE_SPECS
    ANGULAR_JITTER_DEG = 3.0
    LATERAL_SPAN = 0.60
    VERTICAL_SPAN = 0.36
    BASE_DISTANCE = 0.65
    DISTANCE_SPAN = 0.18

    # Reconstruction volume.
    SEARCH_MIN = (-0.55, -0.65, -0.15)
    SEARCH_MAX = (0.75, 0.35, 0.75)
    GRID_SIZE = 13

    # Assumed reconstruction model.
    ASSUMED_BACKGROUND_MEAN = 1200.0
    ASSUMED_CHANNEL_SCALE = 1.0
    ASSUMED_K = 4.0

    # Streaming reconstruction algorithm.
    STREAMING_METHOD = "incremental_mlem"  # "prefix_mlem" or "incremental_mlem"
    RELAXATION = 1.0  # used by incremental_mlem; valid range: (0, 1]
    INNER_ITERATIONS_PER_UPDATE = 20
    MEASUREMENTS_PER_UPDATE = 2
    MAX_SOURCES_TO_REPORT = 3
    MIN_SOURCE_DISTANCE = 0.08
    AVOID_DETECTOR_DISTANCE = 0.65  # [m] penalty radius around trajectory; try 0.25..0.50
    AVOID_DETECTOR_POWER = 6.0  # higher => stronger push away from detector
    ACTIVITY_DECAY = 0.98  # 1.0 disables; try 0.95..0.995 to "forget" old spikes

    # Localization metric.
    ACTIVITY_MAX_RELATIVE_DROP = 0.5

    # Visualization output.
    UPDATE_FRAME_MAX_VOXELS = 100000
    GIF_FPS = 1.0

    dataset = generate_synthetic_dataset(
        num_measurements=NUM_MEASUREMENTS,
        seed=SEED,
        orientation_mode=ORIENTATION_MODE,
        fixed_view_direction=FIXED_VIEW_DIRECTION,
        source_specs=TRUE_SOURCE_SPECS,
        num_channels=NUM_CHANNELS,
        diode_pitch=DIODE_PITCH,
        background_mean=BACKGROUND_MEAN,
        background_channel_std=BACKGROUND_CHANNEL_STD,
        channel_scale_mean=CHANNEL_SCALE_MEAN,
        channel_scale_std=CHANNEL_SCALE_STD,
        generation_k=GENERATION_K,
        peak_total_source_signal=PEAK_TOTAL_SOURCE_SIGNAL,
        scan_center=SCAN_CENTER,
        angular_jitter_deg=ANGULAR_JITTER_DEG,
        lateral_span=LATERAL_SPAN,
        vertical_span=VERTICAL_SPAN,
        base_distance=BASE_DISTANCE,
        distance_span=DISTANCE_SPAN,
    )

    print_dataset_summary(dataset)

    num_channels = dataset["counts"].shape[1]

    assumed_model = {
        "assumed_backgrounds": np.full(num_channels, ASSUMED_BACKGROUND_MEAN, dtype=float),
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
        true_sources=dataset["true_sources"],
        method=STREAMING_METHOD,
        grid_size=GRID_SIZE,
        inner_iterations_per_update=INNER_ITERATIONS_PER_UPDATE,
        measurements_per_update=MEASUREMENTS_PER_UPDATE,
        max_sources=MAX_SOURCES_TO_REPORT,
        min_source_distance=MIN_SOURCE_DISTANCE,
        relaxation=RELAXATION,
        avoid_detector_distance=AVOID_DETECTOR_DISTANCE,
        avoid_detector_power=AVOID_DETECTOR_POWER,
        activity_decay=ACTIVITY_DECAY,
        localization_max_relative_drop=ACTIVITY_MAX_RELATIVE_DROP,
    )
    print_streaming_summary(result)
    print_streaming_localization_metrics(result["stream_history"])
    print_activity_localization_summary(result["stream_history"][-1]["localization_metrics"])

    output_dir = ON_MOCK_DATA_DIR / "multiPoint" / "visualizations"
    updates_3d_dir = output_dir / "streaming_activity_updates_3d"
    updates_3d_dir.mkdir(parents=True, exist_ok=True)

    visualize_multi_source_scene(
        dataset,
        save_path=output_dir / "multi_source_scene_streaming.png",
        show=False,
    )
    visualize_mlem_activity(
        dataset,
        candidate_positions=result["candidate_positions"],
        activity=result["activity"],
        estimated_sources=result["estimated_sources"],
        used_measurements=len(dataset["translations"]),
        save_path=output_dir / "streaming_mlem_activity_reconstruction.png",
        show=False,
    )
    visualize_history_score(
        result["stream_history"],
        save_path=output_dir / "streaming_mlem_score_history.png",
        show=False,
    )
    visualize_streaming_best_position(
        result["stream_history"],
        dataset=dataset,
        save_path=output_dir / "streaming_best_position_history.png",
        show=False,
    )
    visualize_streaming_localization_metrics(
        result["stream_history"],
        save_path=output_dir / "streaming_localization_metric_history.png",
        show=False,
    )

    previous_used_measurements = 0
    for update_index, (activity_snapshot, history_item) in enumerate(
        zip(result["activity_history"], result["stream_history"]),
        start=1,
    ):
        used_measurements = int(history_item.get("used_measurements", previous_used_measurements))
        estimated_sources_update = pick_activity_peaks(
            candidate_positions=result["candidate_positions"],
            activity=activity_snapshot,
            max_sources=MAX_SOURCES_TO_REPORT,
            min_distance=MIN_SOURCE_DISTANCE,
        )
        visualize_mlem_activity(
            dataset,
            candidate_positions=result["candidate_positions"],
            activity=activity_snapshot,
            estimated_sources=estimated_sources_update,
            used_measurements=used_measurements,
            update_start=previous_used_measurements,
            update_end=used_measurements,
            save_path=updates_3d_dir / f"update_{update_index:03d}.png",
            show=False,
            max_voxels=UPDATE_FRAME_MAX_VOXELS,
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
    print(" ", output_dir / "streaming_best_position_history.png")
    print(" ", output_dir / "streaming_localization_metric_history.png")
    print(" ", updates_3d_dir / "update_001.png ...")
    print(" ", updates_gif_path)


if __name__ == "__main__":
    main()
