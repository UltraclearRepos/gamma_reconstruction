"""
Visualization helpers for multi-point source experiments.
"""

from pathlib import Path

import numpy as np

from common.geometry import EPS
from common.visualization import finalize_figure, set_equal_3d_axes


def plot_detector_forward_directions(
    ax,
    translations,
    rotations,
    forward_local,
    stride,
    length,
    color,
    alpha,
    linewidth,
):
    """Draw sparse arrows that show detector forward direction along a trajectory."""
    if translations is None or rotations is None:
        return

    translations = np.asarray(translations, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    forward_local = np.asarray(forward_local, dtype=float)

    num_measurements = len(translations)
    if num_measurements == 0:
        return

    stride = max(1, int(stride))
    indices = np.arange(0, num_measurements, stride, dtype=int)

    positions = translations[indices]
    forward_world = rotations[indices] @ forward_local

    ax.quiver(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        forward_world[:, 0],
        forward_world[:, 1],
        forward_world[:, 2],
        length=float(length),
        color=color,
        alpha=float(alpha),
        linewidth=float(linewidth),
        normalize=True,
    )


def visualize_multi_source_scene(dataset, save_path=None, show=False):
    """Show detector trajectory and all true point sources in world coordinates."""
    import matplotlib.pyplot as plt

    translations = dataset["translations"]
    rotations = dataset.get("rotations")
    forward_local = dataset.get("geometry", {}).get("detector_forward_local", np.array([0.0, 0.0, 1.0]))
    source_positions = np.array([source["position"] for source in dataset["true_sources"]])
    source_strengths = np.array([source["strength"] for source in dataset["true_sources"]])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        color="tab:blue",
        linewidth=2.0,
        label="detector trajectory",
    )
    ax.scatter(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        color="tab:blue",
        s=18,
    )

    stride = max(1, len(translations) // 20)
    plot_detector_forward_directions(
        ax=ax,
        translations=translations,
        rotations=rotations,
        forward_local=forward_local,
        stride=stride,
        length=0.12,
        color="tab:green",
        alpha=0.70,
        linewidth=1.0,
    )

    max_strength = max(float(np.max(source_strengths)), EPS)
    source_sizes = 120.0 + 180.0 * source_strengths / max_strength
    ax.scatter(
        source_positions[:, 0],
        source_positions[:, 1],
        source_positions[:, 2],
        color="tab:red",
        s=source_sizes,
        marker="*",
        edgecolor="black",
        linewidth=0.8,
        depthshade=False,
        label="true sources",
    )

    for index, source in enumerate(dataset["true_sources"], start=1):
        position = source["position"]
        ax.text(
            position[0],
            position[1],
            position[2] + 0.035,
            f"S{index}",
            fontsize=9,
        )

    points_for_scale = np.vstack([translations, source_positions])
    set_equal_3d_axes(ax, points_for_scale)

    ax.set_title("Multi-point source scene")
    ax.set_xlabel("x world [m]")
    ax.set_ylabel("y world [m]")
    ax.set_zlabel("z world [m]")
    ax.legend(loc="upper right")

    finalize_figure(fig, save_path=save_path, show=show)


def visualize_mlem_activity(
    dataset,
    candidate_positions,
    activity,
    estimated_sources=None,
    used_measurements=None,
    update_start=None,
    update_end=None,
    save_path=None,
    show=False,
    max_voxels=300,
):
    """Show the strongest voxels from an ML-EM activity reconstruction."""
    import matplotlib.pyplot as plt

    translations = dataset["translations"]
    rotations = dataset.get("rotations")
    forward_local = dataset.get("geometry", {}).get("detector_forward_local", np.array([0.0, 0.0, 1.0]))
    true_source_positions = np.array([source["position"] for source in dataset["true_sources"]])
    activity = np.asarray(activity, dtype=float)
    candidate_positions = np.asarray(candidate_positions, dtype=float)
    estimated_sources = estimated_sources or []

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    points_for_scale = [translations, true_source_positions]

    if len(activity) > 0 and np.max(activity) > 0.0:
        num_voxels = min(max_voxels, len(activity))
        strongest_indices = np.argsort(activity)[-num_voxels:]
        voxel_positions = candidate_positions[strongest_indices]
        voxel_activity = activity[strongest_indices]

        normalized_activity = voxel_activity / max(float(np.max(voxel_activity)), EPS)
        voxel_sizes = 10.0 + 50.0 * normalized_activity

        scatter = ax.scatter(
            voxel_positions[:, 0],
            voxel_positions[:, 1],
            voxel_positions[:, 2],
            c=np.log1p(voxel_activity),
            s=voxel_sizes,
            cmap="inferno",
            alpha=0.30,
            depthshade=False,
            label="reconstructed activity",
        )
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.75)
        colorbar.set_label("log(1 + activity)")
        points_for_scale.append(voxel_positions)

    # Full detector trajectory in the background.
    ax.plot(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        color="lightgray",
        linewidth=1.2,
        alpha=0.9,
        label="full detector trajectory",
    )

    # Visualize detector pointing direction along the full path.
    plot_detector_forward_directions(
        ax=ax,
        translations=translations,
        rotations=rotations,
        forward_local=forward_local,
        stride=max(1, len(translations) // 18),
        length=0.10,
        color="lightgray",
        alpha=0.25,
        linewidth=0.9,
    )

    # Part of trajectory already used by streaming reconstruction.
    if used_measurements is not None:
        used_end = int(np.clip(used_measurements, 1, len(translations)))
        used_slice = slice(0, used_end)
        ax.plot(
            translations[used_slice, 0],
            translations[used_slice, 1],
            translations[used_slice, 2],
            color="tab:blue",
            linewidth=2.4,
            alpha=0.95,
            label="trajectory used so far",
        )
        plot_detector_forward_directions(
            ax=ax,
            translations=translations[used_slice],
            rotations=rotations[used_slice] if rotations is not None else None,
            forward_local=forward_local,
            stride=max(1, used_end // 18),
            length=0.11,
            color="tab:blue",
            alpha=0.70,
            linewidth=1.1,
        )

    # Newly added chunk in this update.
    if update_start is not None and update_end is not None:
        start = int(np.clip(update_start, 0, len(translations)))
        end = int(np.clip(update_end, 0, len(translations)))
        if end > start:
            ax.plot(
                translations[start:end, 0],
                translations[start:end, 1],
                translations[start:end, 2],
                color="tab:orange",
                linewidth=3.6,
                alpha=0.95,
                label="new trajectory chunk",
            )
            plot_detector_forward_directions(
                ax=ax,
                translations=translations[start:end],
                rotations=rotations[start:end] if rotations is not None else None,
                forward_local=forward_local,
                stride=max(1, (end - start) // 8),
                length=0.12,
                color="tab:orange",
                alpha=0.90,
                linewidth=1.2,
            )

    ax.scatter(
        true_source_positions[:, 0],
        true_source_positions[:, 1],
        true_source_positions[:, 2],
        color="lime",
        s=230,
        marker="*",
        edgecolor="black",
        linewidth=0.8,
        depthshade=False,
        label="true sources",
    )

    if estimated_sources:
        estimated_positions = np.array([source["position"] for source in estimated_sources])
        ax.scatter(
            estimated_positions[:, 0],
            estimated_positions[:, 1],
            estimated_positions[:, 2],
            color="cyan",
            s=130,
            marker="X",
            edgecolor="black",
            linewidth=0.8,
            depthshade=False,
            label="estimated peaks",
        )
        points_for_scale.append(estimated_positions)

    set_equal_3d_axes(ax, np.vstack(points_for_scale))

    ax.set_title("ML-EM activity reconstruction")
    ax.set_xlabel("x world [m]")
    ax.set_ylabel("y world [m]")
    ax.set_zlabel("z world [m]")
    ax.legend(loc="upper right")

    finalize_figure(fig, save_path=save_path, show=show)


def visualize_history_score(history, save_path=None, show=False):
    """Plot reconstruction score from history entries."""
    import matplotlib.pyplot as plt

    steps = []
    scores = []
    for index, item in enumerate(history, start=1):
        if "score" in item:
            steps.append(item.get("iteration", item.get("used_measurements", index)))
            scores.append(item["score"])

    if not scores:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, scores, marker="o", linewidth=1.8, color="tab:purple")
    ax.set_title("Reconstruction score history")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Poisson negative log-likelihood")
    ax.grid(True, alpha=0.3)

    finalize_figure(fig, save_path=save_path, show=show)


def visualize_streaming_best_position(history, dataset, save_path=None, show=False):
    """Plot best estimated source position after each streaming update."""
    import matplotlib.pyplot as plt

    used_measurements = []
    x_values = []
    y_values = []
    z_values = []

    for item in history:
        if "best_position" not in item:
            continue
        position = np.asarray(item["best_position"], dtype=float)
        if position.shape != (3,):
            continue

        used_measurements.append(float(item.get("used_measurements", len(used_measurements) + 1)))
        x_values.append(float(position[0]))
        y_values.append(float(position[1]))
        z_values.append(float(position[2]))

    if not used_measurements:
        return

    true_positions = np.array([source["position"] for source in dataset["true_sources"]])
    true_centroid = np.mean(true_positions, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    coordinate_series = [
        ("x", x_values, true_centroid[0], "tab:red"),
        ("y", y_values, true_centroid[1], "tab:green"),
        ("z", z_values, true_centroid[2], "tab:blue"),
    ]

    for axis, (label, values, true_value, color) in zip(axes, coordinate_series):
        axis.plot(used_measurements, values, marker="o", linewidth=1.7, color=color, label=f"estimated {label}")
        axis.axhline(true_value, linestyle="--", color="black", linewidth=1.0, label=f"true centroid {label}")
        axis.set_ylabel(f"{label} [m]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel("used measurements")
    fig.suptitle("Streaming estimate vs number of measurements")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    finalize_figure(fig, save_path=save_path, show=show)


def visualize_streaming_localization_metrics(history, save_path=None, show=False):
    """Plot selected voxel to true source distance metrics after each update."""
    import matplotlib.pyplot as plt

    used_measurements = []
    mean_distances = []
    weighted_mean_distances = []
    p90_distances = []
    selected_voxel_counts = []

    for item in history:
        metrics = item["localization_metrics"]
        used_measurements.append(float(item["used_measurements"]))
        mean_distances.append(float(metrics["all_source_distance_mean"]))
        weighted_mean_distances.append(float(metrics["all_source_distance_weighted_mean"]))
        p90_distances.append(float(metrics["all_source_distance_p90"]))
        selected_voxel_counts.append(int(metrics["selected_voxel_count"]))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[3, 1])

    axes[0].plot(
        used_measurements,
        mean_distances,
        marker="o",
        linewidth=1.8,
        color="tab:blue",
        label="mean",
    )
    axes[0].plot(
        used_measurements,
        weighted_mean_distances,
        marker="o",
        linewidth=1.8,
        color="tab:orange",
        label="weighted mean",
    )
    axes[0].plot(
        used_measurements,
        p90_distances,
        marker="o",
        linewidth=1.8,
        color="tab:green",
        label="p90",
    )
    axes[0].set_title("Streaming localization metric")
    axes[0].set_ylabel("voxel-source distance [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].bar(
        used_measurements,
        selected_voxel_counts,
        width=1.4,
        color="tab:gray",
        alpha=0.75,
    )
    axes[1].set_xlabel("used measurements")
    axes[1].set_ylabel("voxels")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    finalize_figure(fig, save_path=save_path, show=show)


def create_gif_from_frames(
    frames_dir,
    output_path=None,
    pattern="update_*.png",
    fps=5.0,
    loop=0,
):
    """
    Build an animated GIF from saved frame images.

    Frames are sorted lexicographically, so names like update_001.png,
    update_002.png, ... are handled in the expected order.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to create GIF files. Install it with 'pip install pillow'."
        ) from exc

    frames_dir = Path(frames_dir)
    if output_path is None:
        output_path = frames_dir / f"{frames_dir.name}_{int(round(float(fps)))}fps.gif"
    output_path = Path(output_path)

    frame_paths = sorted(frames_dir.glob(pattern))
    if not frame_paths:
        raise ValueError(f"No frames matching '{pattern}' in {frames_dir}.")

    if fps <= 0:
        raise ValueError("fps must be > 0.")
    duration_ms = int(round(1000.0 / float(fps)))

    images = []
    for path in frame_paths:
        with Image.open(path) as frame:
            images.append(frame.convert("RGB"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=int(loop),
        optimize=False,
    )
    return output_path
