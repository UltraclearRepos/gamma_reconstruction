"""
Visualization helpers for multi-point source experiments.
"""

from pathlib import Path

import numpy as np

from common.geometry import EPS
from common.visualization import finalize_figure, set_3d_axis_limits, set_equal_3d_axes
from common.detector_response_components import angular_response, skin_attenuation


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


def visualize_multi_source_scene(dataset, save_path=None, show=False, axis_limits=None):
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
        length=12.0,
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
            position[2] + 3.5,
            f"S{index}",
            fontsize=9,
        )

    points_for_scale = np.vstack([translations, source_positions])
    if axis_limits is None:
        set_equal_3d_axes(ax, points_for_scale)
    else:
        set_3d_axis_limits(ax, axis_limits)

    ax.set_title("Multi-point source scene")
    ax.set_xlabel("x world [cm]")
    ax.set_ylabel("y world [cm]")
    ax.set_zlabel("z world [cm]")
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
    axis_limits=None,
    activity_color_limits=None,
    activity_size_reference=None,
    frame_label=None,
    highlight_used_trajectory=True,
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

        if activity_size_reference is None:
            size_reference = float(np.max(voxel_activity))
        else:
            size_reference = float(activity_size_reference)
        normalized_activity = np.clip(voxel_activity / max(size_reference, EPS), 0.0, 1.0)
        voxel_sizes = 10.0 + 50.0 * normalized_activity
        color_values = np.log1p(voxel_activity)
        scatter_kwargs = {}
        if activity_color_limits is not None:
            scatter_kwargs["vmin"] = float(activity_color_limits[0])
            scatter_kwargs["vmax"] = float(activity_color_limits[1])

        scatter = ax.scatter(
            voxel_positions[:, 0],
            voxel_positions[:, 1],
            voxel_positions[:, 2],
            c=color_values,
            s=voxel_sizes,
            cmap="inferno",
            alpha=0.30,
            depthshade=False,
            label="reconstructed activity",
            **scatter_kwargs,
        )
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.75)
        colorbar.set_label("log(1 + activity)")
        points_for_scale.append(voxel_positions)

    # Full detector trajectory.
    full_trajectory_color = "lightgray" if highlight_used_trajectory else "tab:blue"
    full_trajectory_alpha = 0.9 if highlight_used_trajectory else 0.95
    full_trajectory_width = 1.2 if highlight_used_trajectory else 2.4
    full_trajectory_label = "full detector trajectory" if highlight_used_trajectory else "detector trajectory"
    ax.plot(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        color=full_trajectory_color,
        linewidth=full_trajectory_width,
        alpha=full_trajectory_alpha,
        label=full_trajectory_label,
    )

    # Visualize detector pointing direction along the full path.
    plot_detector_forward_directions(
        ax=ax,
        translations=translations,
        rotations=rotations,
        forward_local=forward_local,
        stride=max(1, len(translations) // 18),
            length=11.0 if highlight_used_trajectory else 12.0,
        color=full_trajectory_color,
        alpha=0.25 if highlight_used_trajectory else 0.70,
        linewidth=0.9 if highlight_used_trajectory else 1.1,
    )

    # Part of trajectory already used by streaming reconstruction.
    if highlight_used_trajectory and used_measurements is not None:
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
            length=11.0,
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
                length=12.0,
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

    if axis_limits is None:
        set_equal_3d_axes(ax, np.vstack(points_for_scale))
    else:
        set_3d_axis_limits(ax, axis_limits)

    title = "ML-EM activity reconstruction"
    if frame_label is not None:
        title = f"{title}\n{frame_label}"
    ax.set_title(title)
    ax.set_xlabel("x world [cm]")
    ax.set_ylabel("y world [cm]")
    ax.set_zlabel("z world [cm]")
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


def _add_heatmap(fig, ax, values, title, colorbar_label, cmap="viridis", log_scale=False):
    values = np.asarray(values, dtype=float)
    plot_values = np.log10(np.maximum(values, EPS)) if log_scale else values
    image = ax.imshow(plot_values, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("channel")
    ax.set_ylabel("measurement")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label(colorbar_label)


def visualize_forward_response_components(
    diagnostics,
    source_label,
    save_path=None,
    show=False,
):
    """Show detector-response component maps before reconstruction."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    axes = axes.ravel()

    _add_heatmap(
        fig,
        axes[0],
        diagnostics["response"],
        "final response",
        "log10(response)",
        cmap="magma",
        log_scale=True,
    )
    _add_heatmap(
        fig,
        axes[1],
        diagnostics["solid_angle"],
        "solid angle fraction",
        "log10(fraction)",
        cmap="viridis",
        log_scale=True,
    )
    _add_heatmap(
        fig,
        axes[2],
        diagnostics["theta_deg"],
        "effective incidence angle",
        "degrees",
        cmap="cividis",
    )
    _add_heatmap(
        fig,
        axes[3],
        diagnostics["distance_cm"],
        "effective distance",
        "cm",
        cmap="plasma",
    )
    _add_heatmap(
        fig,
        axes[4],
        diagnostics["angular_response"],
        "angular response",
        "gain",
        cmap="viridis",
    )
    _add_heatmap(
        fig,
        axes[5],
        diagnostics["skin_attenuation"],
        "skin attenuation",
        "transmission",
        cmap="viridis",
    )
    _add_heatmap(
        fig,
        axes[6],
        diagnostics["distance_factor"],
        "extra distance factor",
        "factor",
        cmap="viridis",
    )
    _add_heatmap(
        fig,
        axes[7],
        diagnostics["cos_theta"],
        "cos(theta)",
        "cos",
        cmap="cividis",
    )

    fig.suptitle(f"Forward model response components - {source_label}")
    finalize_figure(fig, save_path=save_path, show=show)


def visualize_forward_signal_summary(source_signal_maps, source_labels, save_path=None, show=False):
    """Plot per-source predicted signal before reconstruction."""
    import matplotlib.pyplot as plt

    if not source_signal_maps:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    measurement_axis = np.arange(1, source_signal_maps[0].shape[0] + 1)
    channel_axis = np.arange(1, source_signal_maps[0].shape[1] + 1)

    for signal_map, label in zip(source_signal_maps, source_labels):
        axes[0].plot(
            measurement_axis,
            np.sum(signal_map, axis=1),
            marker="o",
            linewidth=1.6,
            label=label,
        )
        axes[1].plot(
            channel_axis,
            np.mean(signal_map, axis=0),
            marker="o",
            linewidth=1.6,
            label=label,
        )

    axes[0].set_title("predicted source signal by measurement")
    axes[0].set_xlabel("measurement")
    axes[0].set_ylabel("sum over channels [counts]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].set_title("mean predicted source signal by channel")
    axes[1].set_xlabel("channel")
    axes[1].set_ylabel("mean over measurements [counts]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.suptitle("Forward model predicted source signal")
    finalize_figure(fig, save_path=save_path, show=show)


def visualize_expected_counts_breakdown(dataset, save_path=None, show=False):
    """Compare generated counts, noiseless expected counts, and source signal."""
    import matplotlib.pyplot as plt

    counts = np.asarray(dataset["counts"], dtype=float)
    mean_counts = np.asarray(dataset["mean_counts"], dtype=float)
    backgrounds = np.asarray(dataset["true_backgrounds"], dtype=float)
    source_signal = mean_counts - backgrounds[np.newaxis, :]
    residual = counts - mean_counts

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axes = axes.ravel()

    _add_heatmap(fig, axes[0], counts, "sampled counts", "counts", cmap="magma")
    _add_heatmap(fig, axes[1], mean_counts, "expected counts", "counts", cmap="magma")
    _add_heatmap(fig, axes[2], source_signal, "expected source signal", "counts", cmap="inferno")
    _add_heatmap(fig, axes[3], residual, "sampled - expected", "counts", cmap="coolwarm")

    measurement_axis = np.arange(1, counts.shape[0] + 1)
    axes[4].plot(measurement_axis, np.mean(counts, axis=1), label="sampled", linewidth=1.7)
    axes[4].plot(measurement_axis, np.mean(mean_counts, axis=1), label="expected", linewidth=1.7)
    axes[4].plot(measurement_axis, np.mean(source_signal, axis=1), label="source signal", linewidth=1.7)
    axes[4].set_title("mean by measurement")
    axes[4].set_xlabel("measurement")
    axes[4].set_ylabel("mean counts/channel")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc="best")

    channel_axis = np.arange(1, counts.shape[1] + 1)
    axes[5].plot(channel_axis, np.mean(counts, axis=0), label="sampled", linewidth=1.7)
    axes[5].plot(channel_axis, np.mean(mean_counts, axis=0), label="expected", linewidth=1.7)
    axes[5].plot(channel_axis, np.mean(source_signal, axis=0), label="source signal", linewidth=1.7)
    axes[5].set_title("mean by channel")
    axes[5].set_xlabel("channel")
    axes[5].set_ylabel("mean counts/measurement")
    axes[5].grid(True, alpha=0.3)
    axes[5].legend(loc="best")

    fig.suptitle("Counts before reconstruction")
    finalize_figure(fig, save_path=save_path, show=show)


def visualize_detector_response_curves(
    k,
    angular_peak_angle_deg,
    angular_width_deg,
    skin_thickness_cm,
    skin_mu_per_cm,
    save_path=None,
    show=False,
):
    """Show angular response and skin attenuation curves used by the model."""
    import matplotlib.pyplot as plt

    theta_values = np.linspace(0.0, 89.0, 300)
    cos_values = np.cos(np.deg2rad(theta_values))
    angular_values = np.array(
        [
            angular_response(
                cos_theta=cos_theta,
                k=k,
                angular_peak_angle_deg=angular_peak_angle_deg,
                angular_width_deg=angular_width_deg,
            )
            for cos_theta in cos_values
        ],
        dtype=float,
    )
    skin_values = np.array(
        [
            skin_attenuation(
                cos_theta=cos_theta,
                skin_thickness_cm=skin_thickness_cm,
                skin_mu_per_cm=skin_mu_per_cm,
            )
            for cos_theta in cos_values
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(theta_values, angular_values, linewidth=2.0, label="angular response")
    ax.plot(theta_values, skin_values, linewidth=2.0, label="skin attenuation")
    ax.plot(theta_values, angular_values * skin_values, linewidth=2.0, label="product")
    ax.set_title("Detector response curves")
    ax.set_xlabel("incidence angle [deg]")
    ax.set_ylabel("factor")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

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
        axis.set_ylabel(f"{label} [cm]")
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
    axes[0].set_ylabel("voxel-source distance [cm]")
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
        try:
            with Image.open(path) as frame:
                images.append(frame.convert("RGB"))
        except FileNotFoundError:
            # A previous Windows preview/cleanup race can leave a stale glob result.
            continue

    if not images:
        raise ValueError(f"No readable frames matching '{pattern}' in {frames_dir}.")

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
