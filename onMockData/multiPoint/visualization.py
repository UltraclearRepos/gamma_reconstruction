"""
Visualization helpers for multi-point source experiments.
"""

from pathlib import Path

import numpy as np

from common.geometry import EPS
from common.visualization import finalize_figure, set_3d_axis_limits, set_equal_3d_axes
from common.detector_response_components import (
    angular_response,
    collimator_response,
    skin_attenuation,
)


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


def plot_search_cylinder_outline(
    ax,
    dataset,
    color="black",
    alpha=0.35,
    linewidth=0.9,
    num_angles=96,
):
    """Draw the cylindrical reconstruction volume."""
    required_keys = (
        "trajectory_center",
        "search_radius",
        "cylinder_height",
    )
    if any(key not in dataset for key in required_keys):
        return None

    center = np.asarray(dataset["trajectory_center"], dtype=float)
    radius = float(dataset["search_radius"])
    half_height = 0.5 * float(dataset["cylinder_height"])
    angles = np.linspace(0.0, 2.0 * np.pi, int(num_angles), endpoint=True)
    circle_xy = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
    lower = np.column_stack([
        center[0] + circle_xy[:, 0],
        center[1] + circle_xy[:, 1],
        np.full(len(angles), center[2] - half_height),
    ])
    upper = lower.copy()
    upper[:, 2] = center[2] + half_height

    ax.plot(lower[:, 0], lower[:, 1], lower[:, 2], color=color, linestyle="--", linewidth=linewidth, alpha=alpha, label="search cylinder")
    ax.plot(upper[:, 0], upper[:, 1], upper[:, 2], color=color, linestyle="--", linewidth=linewidth, alpha=alpha)
    for index in np.linspace(0, len(angles) - 2, 8, dtype=int):
        ax.plot(
            [lower[index, 0], upper[index, 0]],
            [lower[index, 1], upper[index, 1]],
            [lower[index, 2], upper[index, 2]],
            color=color,
            linestyle="--",
            linewidth=linewidth,
            alpha=alpha,
        )

    return np.vstack([lower, upper])


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
        length=6.0,
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


def voxelize_true_source_activity(candidate_positions, true_sources):
    """
    Project generated source activity onto reconstruction voxels for plotting.

    This uses the actual source components stored in the generated dataset.
    Each component is assigned to its nearest reconstruction voxel.
    """
    candidate_positions = np.asarray(candidate_positions, dtype=float)
    true_activity = np.zeros(len(candidate_positions), dtype=float)
    if len(candidate_positions) == 0:
        return true_activity

    for source in true_sources:
        components = source.get("components")
        if components is None:
            components = [
                {
                    "position": source["position"],
                    "strength": source.get("strength", source.get("strength_per_second", 0.0)),
                }
            ]

        component_positions = np.asarray(
            [component["position"] for component in components],
            dtype=float,
        )
        component_strengths = np.asarray(
            [component["strength"] for component in components],
            dtype=float,
        )
        if len(component_positions) == 0:
            continue

        squared_distances = np.sum(
            (component_positions[:, np.newaxis, :] - candidate_positions[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        nearest_indices = np.argmin(squared_distances, axis=1)
        np.add.at(true_activity, nearest_indices, component_strengths)

    return true_activity


def visualize_true_voxel_activity(
    dataset,
    candidate_positions,
    true_activity,
    save_path=None,
    show=False,
    max_voxels=300,
    axis_limits=None,
):
    """Show generated ground-truth activity projected onto reconstruction voxels."""
    import matplotlib.pyplot as plt

    translations = dataset["translations"]
    rotations = dataset.get("rotations")
    forward_local = dataset.get("geometry", {}).get("detector_forward_local", np.array([0.0, 0.0, 1.0]))
    true_source_positions = np.array([source["position"] for source in dataset["true_sources"]])
    candidate_positions = np.asarray(candidate_positions, dtype=float)
    true_activity = np.asarray(true_activity, dtype=float)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    points_for_scale = [translations, true_source_positions]

    if len(true_activity) > 0 and np.max(true_activity) > 0.0:
        active_indices = np.flatnonzero(true_activity > 0.0)
        if len(active_indices) > max_voxels:
            active_indices = active_indices[np.argsort(true_activity[active_indices])[-max_voxels:]]

        voxel_positions = candidate_positions[active_indices]
        voxel_activity = true_activity[active_indices]
        normalized_activity = np.clip(
            voxel_activity / max(float(np.max(voxel_activity)), EPS),
            0.0,
            1.0,
        )
        voxel_sizes = 12.0 + 70.0 * normalized_activity

        scatter = ax.scatter(
            voxel_positions[:, 0],
            voxel_positions[:, 1],
            voxel_positions[:, 2],
            c=np.log1p(voxel_activity),
            s=voxel_sizes,
            cmap="viridis",
            alpha=0.42,
            depthshade=False,
            label="true voxel activity",
        )
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.75)
        colorbar.set_label("log(1 + true activity)")
        points_for_scale.append(voxel_positions)

    ax.plot(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        color="tab:blue",
        linewidth=2.4,
        alpha=0.95,
        label="detector trajectory",
    )
    plot_detector_forward_directions(
        ax=ax,
        translations=translations,
        rotations=rotations,
        forward_local=forward_local,
        stride=max(1, len(translations) // 18),
        length=6.0,
        color="tab:blue",
        alpha=0.65,
        linewidth=1.0,
    )
    search_outline_points = plot_search_cylinder_outline(ax, dataset)
    if search_outline_points is not None:
        points_for_scale.append(search_outline_points)

    if axis_limits is None:
        set_equal_3d_axes(ax, np.vstack(points_for_scale))
    else:
        set_3d_axis_limits(ax, axis_limits)

    ax.set_title("Generated true activity on reconstruction voxels")
    ax.set_xlabel("x world [cm]")
    ax.set_ylabel("y world [cm]")
    ax.set_zlabel("z world [cm]")
    ax.legend(loc="upper right")

    finalize_figure(fig, save_path=save_path, show=show)


def visualize_activity_map_comparison(
    dataset,
    candidate_positions,
    true_activity,
    reconstructed_activity,
    component_matching=None,
    voxel_size_cm=1.0,
    save_path=None,
    show=False,
    max_voxels=300,
    axis_limits=None,
):
    """Show true/reconstructed activity maps and optional blob matching."""
    import matplotlib.pyplot as plt

    translations = dataset["translations"]
    true_source_positions = np.array([source["position"] for source in dataset["true_sources"]])
    candidate_positions = np.asarray(candidate_positions, dtype=float)
    true_activity = np.asarray(true_activity, dtype=float)
    reconstructed_activity = np.asarray(reconstructed_activity, dtype=float)

    has_component_panel = component_matching is not None
    fig = plt.figure(figsize=(15, 13) if has_component_panel else (15, 7))
    axes = [
        fig.add_subplot(221 if has_component_panel else 121, projection="3d"),
        fig.add_subplot(222 if has_component_panel else 122, projection="3d"),
    ]
    if has_component_panel:
        axes.append(fig.add_subplot(223, projection="3d"))
        axes.append(fig.add_subplot(224, projection="3d"))

    def plot_activity_panel(ax, activity, title, cmap, label, show_search_bounds=False):
        points_for_scale = [translations, true_source_positions]
        if len(activity) > 0 and np.max(activity) > 0.0:
            active_indices = np.flatnonzero(activity > 0.0)
            if len(active_indices) > max_voxels:
                active_indices = active_indices[np.argsort(activity[active_indices])[-max_voxels:]]
            voxel_positions = candidate_positions[active_indices]
            voxel_activity = activity[active_indices]
            normalized_activity = np.clip(
                voxel_activity / max(float(np.max(voxel_activity)), EPS),
                0.0,
                1.0,
            )
            voxel_sizes = 10.0 + 65.0 * normalized_activity
            scatter = ax.scatter(
                voxel_positions[:, 0],
                voxel_positions[:, 1],
                voxel_positions[:, 2],
                c=np.log1p(voxel_activity),
                s=voxel_sizes,
                cmap=cmap,
                alpha=0.42,
                depthshade=False,
                label=label,
            )
            colorbar = fig.colorbar(scatter, ax=ax, shrink=0.70)
            colorbar.set_label("log(1 + activity)")
            points_for_scale.append(voxel_positions)

        ax.plot(
            translations[:, 0],
            translations[:, 1],
            translations[:, 2],
            color="tab:blue",
            linewidth=1.8,
            alpha=0.85,
            label="detector trajectory",
        )
        if show_search_bounds:
            search_outline_points = plot_search_cylinder_outline(ax, dataset)
            if search_outline_points is not None:
                points_for_scale.append(search_outline_points)
        if axis_limits is None:
            set_equal_3d_axes(ax, np.vstack(points_for_scale))
        else:
            set_3d_axis_limits(ax, axis_limits)
        ax.set_title(title)
        ax.set_xlabel("x world [cm]")
        ax.set_ylabel("y world [cm]")
        ax.set_zlabel("z world [cm]")
        ax.legend(loc="upper right")

    plot_activity_panel(
        axes[0],
        true_activity,
        "True activity on reconstruction voxels",
        "viridis",
        "true activity",
        show_search_bounds=True,
    )
    plot_activity_panel(
        axes[1],
        reconstructed_activity,
        "Reconstructed activity",
        "inferno",
        "reconstructed activity",
    )
    if has_component_panel:
        plot_thresholded_activity_panel(
            ax=axes[2],
            dataset=dataset,
            candidate_positions=candidate_positions,
            true_activity=true_activity,
            reconstructed_activity=reconstructed_activity,
            threshold_fraction=component_matching["threshold_fraction"],
            axis_limits=axis_limits,
            title="Voxels above blob threshold",
        )
        plot_component_matching_panel(
            ax=axes[3],
            dataset=dataset,
            candidate_positions=candidate_positions,
            component_matching=component_matching,
            voxel_size_cm=voxel_size_cm,
            axis_limits=axis_limits,
            title="Activity blob matching",
        )

    fig.suptitle("True vs reconstructed activity maps")
    finalize_figure(fig, save_path=save_path, show=show)


def plot_thresholded_activity_panel(
    ax,
    dataset,
    candidate_positions,
    true_activity,
    reconstructed_activity,
    threshold_fraction,
    axis_limits=None,
    title="Voxels above blob threshold",
):
    """Draw full true-source voxels and reconstructed voxels above the blob threshold."""
    translations = np.asarray(dataset["translations"], dtype=float)
    true_source_positions = np.array([source["position"] for source in dataset["true_sources"]])
    candidate_positions = np.asarray(candidate_positions, dtype=float)
    true_activity = np.maximum(np.asarray(true_activity, dtype=float), 0.0)
    reconstructed_activity = np.maximum(np.asarray(reconstructed_activity, dtype=float), 0.0)
    threshold_fraction = float(threshold_fraction)
    points_for_scale = [translations, true_source_positions]

    ax.plot(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        color="tab:blue",
        linewidth=1.8,
        alpha=0.85,
        label="detector trajectory",
    )

    true_mask = true_activity > EPS
    reconstructed_mask = np.zeros_like(reconstructed_activity, dtype=bool)
    if len(reconstructed_activity) > 0 and float(np.max(reconstructed_activity)) > EPS:
        reconstructed_mask = (
            reconstructed_activity
            >= threshold_fraction * float(np.max(reconstructed_activity))
        )

    if np.any(true_mask):
        true_positions = candidate_positions[true_mask]
        ax.scatter(
            true_positions[:, 0],
            true_positions[:, 1],
            true_positions[:, 2],
            color="tab:green",
            s=34,
            marker="o",
            alpha=0.55,
            depthshade=False,
            label="true source voxels",
        )
        points_for_scale.append(true_positions)

    if np.any(reconstructed_mask):
        reconstructed_positions = candidate_positions[reconstructed_mask]
        ax.scatter(
            reconstructed_positions[:, 0],
            reconstructed_positions[:, 1],
            reconstructed_positions[:, 2],
            color="tab:red",
            s=34,
            marker="^",
            alpha=0.55,
            depthshade=False,
            label="reconstructed active voxels",
        )
        points_for_scale.append(reconstructed_positions)

    if axis_limits is None:
        set_equal_3d_axes(ax, np.vstack(points_for_scale))
    else:
        set_3d_axis_limits(ax, axis_limits)

    ax.set_title(f"{title}\nrecon threshold {100.0 * threshold_fraction:.1f}% of max")
    ax.set_xlabel("x world [cm]")
    ax.set_ylabel("y world [cm]")
    ax.set_zlabel("z world [cm]")
    ax.legend(loc="upper right")


def render_thresholded_activity_history_frames(
    dataset,
    candidate_positions,
    true_activity,
    activity_history,
    history,
    threshold_fraction,
    frames_dir,
    axis_limits=None,
):
    """Save one thresholded true/reconstructed voxel panel per reconstruction snapshot."""
    import matplotlib.pyplot as plt

    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    num_frames = len(activity_history)

    for frame_index, (activity_snapshot, history_item) in enumerate(
        zip(activity_history, history),
        start=1,
    ):
        iteration = int(history_item.get("iteration", frame_index))
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        plot_thresholded_activity_panel(
            ax=ax,
            dataset=dataset,
            candidate_positions=candidate_positions,
            true_activity=true_activity,
            reconstructed_activity=activity_snapshot,
            threshold_fraction=threshold_fraction,
            axis_limits=axis_limits,
            title=f"Voxels above blob threshold\nframe {frame_index}/{num_frames} | iteration {iteration}",
        )
        frame_path = frames_dir / f"thresholded_iteration_{iteration:06d}.png"
        finalize_figure(fig, save_path=frame_path, show=False)
        frame_paths.append(frame_path)

    return frame_paths


def plot_component_box(
    ax,
    component,
    candidate_positions,
    voxel_size_cm,
    color,
    linewidth,
    alpha,
    label=None,
):
    """Draw a wireframe box enclosing one activity component."""
    positions = candidate_positions[component["indices"]]
    half_voxel = 0.5 * float(voxel_size_cm)
    minimum = np.min(positions, axis=0) - half_voxel
    maximum = np.max(positions, axis=0) + half_voxel

    corners = np.array(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=float,
    )
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for edge_index, (start, end) in enumerate(edges):
        ax.plot(
            corners[[start, end], 0],
            corners[[start, end], 1],
            corners[[start, end], 2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label if edge_index == 0 else None,
        )

    return corners


def plot_component_matching_panel(
    ax,
    dataset,
    candidate_positions,
    component_matching,
    voxel_size_cm,
    axis_limits=None,
    title="Activity blob matching",
):
    """Draw blob envelopes and match arrows on an existing 3D axis."""
    translations = np.asarray(dataset["translations"], dtype=float)
    candidate_positions = np.asarray(candidate_positions, dtype=float)
    true_components = component_matching.get("true_components", [])
    reconstructed_components = component_matching.get("reconstructed_components", [])
    points_for_scale = [translations]

    ax.plot(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        color="tab:blue",
        linewidth=2.0,
        alpha=0.85,
        label="detector trajectory",
    )

    for component_index, component in enumerate(true_components):
        corners = plot_component_box(
            ax=ax,
            component=component,
            candidate_positions=candidate_positions,
            voxel_size_cm=voxel_size_cm,
            color="tab:green",
            linewidth=2.2,
            alpha=0.85,
            label="true blob envelope" if component_index == 0 else None,
        )
        points_for_scale.append(corners)
        center = component["centroid"]
        ax.scatter(
            center[0],
            center[1],
            center[2],
            color="tab:green",
            s=35,
            marker="o",
            depthshade=False,
        )
        ax.text(center[0], center[1], center[2], f"T{component['label']}", fontsize=9)

    for component_index, component in enumerate(reconstructed_components):
        corners = plot_component_box(
            ax=ax,
            component=component,
            candidate_positions=candidate_positions,
            voxel_size_cm=voxel_size_cm,
            color="tab:red",
            linewidth=2.0,
            alpha=0.85,
            label="reconstructed blob envelope" if component_index == 0 else None,
        )
        points_for_scale.append(corners)
        center = component["centroid"]
        ax.scatter(
            center[0],
            center[1],
            center[2],
            color="tab:red",
            s=35,
            marker="^",
            depthshade=False,
        )
        ax.text(center[0], center[1], center[2], f"R{component['label']}", fontsize=9)

    true_by_label = {
        int(component["label"]): component
        for component in true_components
    }
    reconstructed_by_label = {
        int(component["label"]): component
        for component in reconstructed_components
    }
    for match_index, match in enumerate(component_matching.get("matches", [])):
        reconstructed_component = reconstructed_by_label.get(int(match["reconstructed_label"]))
        true_component = true_by_label.get(int(match["true_label"]))
        if reconstructed_component is None or true_component is None:
            continue

        start = np.asarray(reconstructed_component["centroid"], dtype=float)
        end = np.asarray(true_component["centroid"], dtype=float)
        vector = end - start
        ax.quiver(
            start[0],
            start[1],
            start[2],
            vector[0],
            vector[1],
            vector[2],
            color="black",
            linewidth=1.6,
            arrow_length_ratio=0.18,
            alpha=0.85,
            label="matched centroid vector" if match_index == 0 else None,
        )
        points_for_scale.append(np.vstack([start, end]))

    if axis_limits is None:
        set_equal_3d_axes(ax, np.vstack(points_for_scale))
    else:
        set_3d_axis_limits(ax, axis_limits)

    ax.set_title(title)
    ax.set_xlabel("x world [cm]")
    ax.set_ylabel("y world [cm]")
    ax.set_zlabel("z world [cm]")
    ax.legend(loc="upper right")


def visualize_activity_component_matching(
    dataset,
    candidate_positions,
    component_matching,
    voxel_size_cm,
    save_path=None,
    show=False,
    axis_limits=None,
):
    """Show only blob envelopes and match arrows between reconstructed and true blobs."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    plot_component_matching_panel(
        ax=ax,
        dataset=dataset,
        candidate_positions=candidate_positions,
        component_matching=component_matching,
        voxel_size_cm=voxel_size_cm,
        axis_limits=axis_limits,
    )

    finalize_figure(fig, save_path=save_path, show=show)


def visualize_mlem_activity(
    dataset,
    candidate_positions,
    activity,
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
            length=5.5 if highlight_used_trajectory else 6.0,
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
            length=5.5,
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
                length=6.0,
                color="tab:orange",
                alpha=0.90,
                linewidth=1.2,
            )

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


def visualize_system_matrix(
    system_matrix,
    num_measurements,
    num_channels,
    save_path=None,
    show=False,
    max_voxels_to_draw=4000,
):
    """Show the reconstruction matrix H: rows are measurement-channel pairs, columns are voxels."""
    import matplotlib.pyplot as plt

    system_matrix = np.asarray(system_matrix, dtype=float)
    if system_matrix.size == 0:
        return

    num_rows, num_voxels = system_matrix.shape
    stride = max(1, int(np.ceil(num_voxels / float(max_voxels_to_draw))))
    drawn_matrix = system_matrix[:, ::stride]
    drawn_voxel_count = drawn_matrix.shape[1]

    row_sum = np.sum(system_matrix, axis=1).reshape(num_measurements, num_channels)
    voxel_sensitivity = np.sum(system_matrix, axis=0)
    nonzero_values = system_matrix[system_matrix > 0.0]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()

    image = axes[0].imshow(
        np.log10(np.maximum(drawn_matrix, EPS)),
        aspect="auto",
        origin="lower",
        cmap="magma",
    )
    axes[0].set_title(f"H matrix log10 response ({num_rows} x {num_voxels})")
    axes[0].set_xlabel(
        f"voxel index sample, every {stride}" if stride > 1 else "voxel index"
    )
    axes[0].set_ylabel("measurement-channel row")
    colorbar = fig.colorbar(image, ax=axes[0], shrink=0.82)
    colorbar.set_label("log10(H)")

    _add_heatmap(
        fig,
        axes[1],
        row_sum,
        "row response sum",
        "sum over voxels",
        cmap="viridis",
        log_scale=True,
    )

    axes[2].plot(np.arange(num_voxels), voxel_sensitivity, linewidth=1.2)
    axes[2].set_title("voxel sensitivity")
    axes[2].set_xlabel("voxel index")
    axes[2].set_ylabel("sum over rows")
    axes[2].grid(True, alpha=0.3)

    if len(nonzero_values) > 0:
        axes[3].hist(np.log10(np.maximum(nonzero_values, EPS)), bins=80, color="tab:blue")
    axes[3].set_title("non-zero H values")
    axes[3].set_xlabel("log10(H)")
    axes[3].set_ylabel("count")
    axes[3].grid(True, alpha=0.3)

    if drawn_voxel_count == 0:
        axes[0].text(0.5, 0.5, "empty", transform=axes[0].transAxes, ha="center")

    finalize_figure(fig, save_path=save_path, show=show)


def visualize_system_matrix_sensitivity_3d(
    dataset,
    candidate_positions,
    system_matrix,
    save_path=None,
    show=False,
    max_voxels=3000,
    axis_limits=None,
):
    """Show voxel sensitivity from H in 3D: sensitivity[v] = sum_rows H[row, v]."""
    import matplotlib.pyplot as plt

    candidate_positions = np.asarray(candidate_positions, dtype=float)
    system_matrix = np.asarray(system_matrix, dtype=float)
    if system_matrix.size == 0 or len(candidate_positions) == 0:
        return

    sensitivity = np.sum(system_matrix, axis=0)
    if len(sensitivity) != len(candidate_positions):
        raise ValueError("system_matrix columns must match candidate_positions length.")

    translations = np.asarray(dataset["translations"], dtype=float)
    rotations = dataset.get("rotations")
    forward_local = dataset.get("geometry", {}).get("detector_forward_local", np.array([0.0, 0.0, 1.0]))
    true_source_positions = np.array([source["position"] for source in dataset["true_sources"]], dtype=float)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    points_for_scale = [translations, true_source_positions]

    if np.max(sensitivity) > 0.0:
        num_voxels = min(int(max_voxels), len(sensitivity))
        strongest_indices = np.argsort(sensitivity)[-num_voxels:]
        voxel_positions = candidate_positions[strongest_indices]
        voxel_sensitivity = sensitivity[strongest_indices]
        normalized = np.clip(voxel_sensitivity / max(float(np.max(voxel_sensitivity)), EPS), 0.0, 1.0)
        scatter = ax.scatter(
            voxel_positions[:, 0],
            voxel_positions[:, 1],
            voxel_positions[:, 2],
            c=np.log10(np.maximum(voxel_sensitivity, EPS)),
            s=8.0 + 48.0 * normalized,
            cmap="plasma",
            alpha=0.32,
            depthshade=False,
            label="H voxel sensitivity",
        )
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.75)
        colorbar.set_label("log10(sum H[:, voxel])")
        points_for_scale.append(voxel_positions)

    ax.plot(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        color="tab:blue",
        linewidth=2.2,
        alpha=0.9,
        label="detector trajectory",
    )
    plot_detector_forward_directions(
        ax=ax,
        translations=translations,
        rotations=rotations,
        forward_local=forward_local,
        stride=max(1, len(translations) // 18),
        length=6.0,
        color="tab:blue",
        alpha=0.55,
        linewidth=1.0,
    )
    search_outline_points = plot_search_cylinder_outline(ax, dataset)
    if search_outline_points is not None:
        points_for_scale.append(search_outline_points)

    ax.scatter(
        true_source_positions[:, 0],
        true_source_positions[:, 1],
        true_source_positions[:, 2],
        marker="*",
        s=130,
        color="black",
        label="true sources",
    )

    if axis_limits is None:
        set_equal_3d_axes(ax, np.vstack(points_for_scale))
    else:
        set_3d_axis_limits(ax, axis_limits)

    ax.set_title("3D H sensitivity map")
    ax.set_xlabel("x world [cm]")
    ax.set_ylabel("y world [cm]")
    ax.set_zlabel("z world [cm]")
    ax.legend(loc="upper right")
    finalize_figure(fig, save_path=save_path, show=show)


def render_system_matrix_measurement_3d_frames(
    dataset,
    candidate_positions,
    system_matrix,
    num_measurements,
    num_channels,
    frames_dir,
    max_voxels=3000,
    axis_limits=None,
):
    """Save one 3D voxel-sensitivity frame per detector measurement."""
    import matplotlib.pyplot as plt

    candidate_positions = np.asarray(candidate_positions, dtype=float)
    system_matrix = np.asarray(system_matrix, dtype=float)
    if system_matrix.size == 0 or len(candidate_positions) == 0:
        return []

    num_rows, num_voxels = system_matrix.shape
    expected_rows = int(num_measurements) * int(num_channels)
    if num_rows != expected_rows:
        raise ValueError("system_matrix row count must equal num_measurements * num_channels.")
    if num_voxels != len(candidate_positions):
        raise ValueError("system_matrix columns must match candidate_positions length.")

    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    measurement_maps = system_matrix.reshape(int(num_measurements), int(num_channels), num_voxels).sum(axis=1)
    positive_values = measurement_maps[measurement_maps > 0.0]
    if len(positive_values) == 0:
        return []
    color_min = float(np.log10(max(float(np.min(positive_values)), EPS)))
    color_max = float(np.log10(max(float(np.max(positive_values)), EPS)))

    translations = np.asarray(dataset["translations"], dtype=float)
    rotations = dataset.get("rotations")
    forward_local = dataset.get("geometry", {}).get("detector_forward_local", np.array([0.0, 0.0, 1.0]))
    true_source_positions = np.array([source["position"] for source in dataset["true_sources"]], dtype=float)
    frame_paths = []

    for measurement_index in range(int(num_measurements)):
        sensitivity = measurement_maps[measurement_index]
        if np.max(sensitivity) <= 0.0:
            continue

        num_to_draw = min(int(max_voxels), len(sensitivity))
        strongest_indices = np.argsort(sensitivity)[-num_to_draw:]
        voxel_positions = candidate_positions[strongest_indices]
        voxel_sensitivity = sensitivity[strongest_indices]
        normalized = np.clip(voxel_sensitivity / max(float(np.max(voxel_sensitivity)), EPS), 0.0, 1.0)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            voxel_positions[:, 0],
            voxel_positions[:, 1],
            voxel_positions[:, 2],
            c=np.log10(np.maximum(voxel_sensitivity, EPS)),
            s=8.0 + 48.0 * normalized,
            cmap="plasma",
            alpha=0.32,
            depthshade=False,
            vmin=color_min,
            vmax=color_max,
            label="measurement H sensitivity",
        )
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.75)
        colorbar.set_label("log10(sum channels H)")

        ax.plot(
            translations[:, 0],
            translations[:, 1],
            translations[:, 2],
            color="lightgray",
            linewidth=1.2,
            alpha=0.85,
            label="full trajectory",
        )
        ax.scatter(
            translations[measurement_index, 0],
            translations[measurement_index, 1],
            translations[measurement_index, 2],
            color="tab:blue",
            s=80,
            label="current detector position",
        )
        plot_detector_forward_directions(
            ax=ax,
            translations=translations[measurement_index:measurement_index + 1],
            rotations=rotations[measurement_index:measurement_index + 1] if rotations is not None else None,
            forward_local=forward_local,
            stride=1,
            length=6.0,
            color="tab:blue",
            alpha=0.9,
            linewidth=1.4,
        )
        ax.scatter(
            true_source_positions[:, 0],
            true_source_positions[:, 1],
            true_source_positions[:, 2],
            marker="*",
            s=130,
            color="black",
            label="true sources",
        )

        if axis_limits is None:
            points_for_scale = [translations, true_source_positions, voxel_positions]
            set_equal_3d_axes(ax, np.vstack(points_for_scale))
        else:
            set_3d_axis_limits(ax, axis_limits)

        ax.set_title(f"3D H map for measurement {measurement_index + 1}/{int(num_measurements)}")
        ax.set_xlabel("x world [cm]")
        ax.set_ylabel("y world [cm]")
        ax.set_zlabel("z world [cm]")
        ax.legend(loc="upper right")

        frame_path = frames_dir / f"measurement_3d_{measurement_index + 1:06d}.png"
        finalize_figure(fig, save_path=frame_path, show=False)
        frame_paths.append(frame_path)

    return frame_paths


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

    fig, axes = plt.subplots(3, 3, figsize=(18, 11), constrained_layout=True)
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
        diagnostics["collimator_response"],
        "collimator response",
        "gain",
        cmap="viridis",
    )
    _add_heatmap(
        fig,
        axes[6],
        diagnostics["skin_attenuation"],
        "skin attenuation",
        "transmission",
        cmap="viridis",
    )
    _add_heatmap(
        fig,
        axes[7],
        diagnostics["distance_factor"],
        "extra distance factor",
        "factor",
        cmap="viridis",
    )
    _add_heatmap(
        fig,
        axes[8],
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
    angular_peak_angle_deg,
    angular_width_deg,
    collimator_half_angle_deg,
    collimator_edge_width_deg,
    collimator_leakage,
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
                angular_peak_angle_deg=angular_peak_angle_deg,
                angular_width_deg=angular_width_deg,
            )
            for cos_theta in cos_values
        ],
        dtype=float,
    )
    collimator_values = np.array(
        [
            collimator_response(
                cos_theta=cos_theta,
                collimator_half_angle_deg=collimator_half_angle_deg,
                collimator_edge_width_deg=collimator_edge_width_deg,
                collimator_leakage=collimator_leakage,
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
    ax.plot(theta_values, collimator_values, linewidth=2.0, label="collimator response")
    ax.plot(theta_values, skin_values, linewidth=2.0, label="skin attenuation")
    ax.plot(
        theta_values,
        angular_values * collimator_values * skin_values,
        linewidth=2.0,
        label="product",
    )
    ax.set_title("Detector response curves")
    ax.set_xlabel("incidence angle [deg]")
    ax.set_ylabel("factor")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

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
