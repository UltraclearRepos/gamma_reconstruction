"""Create a four-panel overview of article source cases and detector trajectory."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


ON_MOCK_DATA_DIR = Path(__file__).resolve().parent
if str(ON_MOCK_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(ON_MOCK_DATA_DIR))

from article_pearson_settings import OUTPUT_DIR, TRUE_SOURCE_CASES  # noqa: E402
from common.geometry import make_detector_geometry, make_detector_trajectory  # noqa: E402


TRAJECTORY_SETTINGS = {
    "seed": 11,
    "num_measurements": 100,
    "num_channels": 16,
    "diode_pitch": 0.15875,
    "trajectory_center": (0.0, 0.0, 0.0),
    "search_radius": 10.0,
    "cylinder_height": 12.0,
    "trajectory_mode": "around_center_grid",
    "angular_jitter_deg": 10.0,
    "lateral_span": 15.0,
    "vertical_span": 3.0,
    "base_distance": 5.0,
    "distance_span": 0.2,
    "orbit_radius": 13.0,
    "orbit_radius_span": 0.3,
    "orbit_angle_span_deg": 180.0,
    "orbit_start_angle_deg": -90.0,
    "orbit_vertical_span": 5.0,
    "orientation_mode": "towards_target",
    "fixed_view_direction": (0.0, 1.0, 0.0),
    "fixed_sweep_z_range_deg": 5.0,
    "fixed_sweep_x_range_deg": 15.0,
}


def plot_search_cylinder(ax, radius, height, center=(0.0, 0.0, 0.0)):
    center = np.asarray(center, dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, 120)
    z_values = center[2] + np.array([-0.5 * height, 0.5 * height])
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)

    for z in z_values:
        ax.plot(x, y, np.full_like(x, z), color="0.25", linewidth=0.8, alpha=0.45)

    for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        x_edge = center[0] + radius * np.cos(angle)
        y_edge = center[1] + radius * np.sin(angle)
        ax.plot(
            [x_edge, x_edge],
            [y_edge, y_edge],
            z_values,
            color="0.25",
            linewidth=0.55,
            alpha=0.25,
        )


def plot_source_cloud(ax, source, color):
    position = np.asarray(source["position"], dtype=float)
    radius = max(float(source.get("spread_cm", 0.0)), 0.2)

    u, v = np.mgrid[0 : 2 * np.pi : 18j, 0 : np.pi : 10j]
    x = radius * np.cos(u) * np.sin(v) + position[0]
    y = radius * np.sin(u) * np.sin(v) + position[1]
    z = radius * np.cos(v) + position[2]

    ax.plot_wireframe(x, y, z, color=color, linewidth=0.55, alpha=0.70)
    ax.scatter(
        [position[0]],
        [position[1]],
        [position[2]],
        color=color,
        s=28,
        depthshade=False,
    )


def set_axes(ax):
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-8, 8)
    ax.set_box_aspect((30, 30, 16))
    ax.set_xlabel("x [cm]", labelpad=0)
    ax.set_ylabel("y [cm]", labelpad=0)
    ax.set_zlabel("z [cm]", labelpad=0)
    ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=24, azim=-58)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(TRAJECTORY_SETTINGS["seed"])
    geometry = make_detector_geometry(
        num_channels=TRAJECTORY_SETTINGS["num_channels"],
        diode_pitch=TRAJECTORY_SETTINGS["diode_pitch"],
    )
    rotations, translations = make_detector_trajectory(
        geometry=geometry,
        trajectory_center=TRAJECTORY_SETTINGS["trajectory_center"],
        num_measurements=TRAJECTORY_SETTINGS["num_measurements"],
        rng=rng,
        trajectory_mode=TRAJECTORY_SETTINGS["trajectory_mode"],
        angular_jitter_deg=TRAJECTORY_SETTINGS["angular_jitter_deg"],
        lateral_span=TRAJECTORY_SETTINGS["lateral_span"],
        vertical_span=TRAJECTORY_SETTINGS["vertical_span"],
        base_distance=TRAJECTORY_SETTINGS["base_distance"],
        distance_span=TRAJECTORY_SETTINGS["distance_span"],
        orbit_radius=TRAJECTORY_SETTINGS["orbit_radius"],
        orbit_radius_span=TRAJECTORY_SETTINGS["orbit_radius_span"],
        orbit_angle_span_deg=TRAJECTORY_SETTINGS["orbit_angle_span_deg"],
        orbit_start_angle_deg=TRAJECTORY_SETTINGS["orbit_start_angle_deg"],
        orbit_vertical_span=TRAJECTORY_SETTINGS["orbit_vertical_span"],
        orientation_mode=TRAJECTORY_SETTINGS["orientation_mode"],
        fixed_view_direction=TRAJECTORY_SETTINGS["fixed_view_direction"],
        fixed_sweep_z_range_deg=TRAJECTORY_SETTINGS["fixed_sweep_z_range_deg"],
        fixed_sweep_x_range_deg=TRAJECTORY_SETTINGS["fixed_sweep_x_range_deg"],
    )

    forward_local = geometry["detector_forward_local"]
    forward_world = rotations @ forward_local
    arrow_stride = 6
    arrow_indices = np.arange(0, len(translations), arrow_stride)

    all_strengths = [
        float(source["strength_per_second"])
        for case in TRUE_SOURCE_CASES
        for source in case["true_sources"]
    ]
    strength_norm = Normalize(vmin=min(all_strengths), vmax=max(all_strengths))
    strength_cmap = plt.get_cmap("plasma")

    fig = plt.figure(figsize=(10, 8.2))
    for plot_index, case in enumerate(TRUE_SOURCE_CASES, start=1):
        ax = fig.add_subplot(2, 2, plot_index, projection="3d")

        plot_search_cylinder(
            ax,
            radius=TRAJECTORY_SETTINGS["search_radius"],
            height=TRAJECTORY_SETTINGS["cylinder_height"],
            center=TRAJECTORY_SETTINGS["trajectory_center"],
        )

        ax.plot(
            translations[:, 0],
            translations[:, 1],
            translations[:, 2],
            color="tab:blue",
            linewidth=1.8,
            alpha=0.90,
            label="detector trajectory" if plot_index == 1 else None,
        )
        ax.quiver(
            translations[arrow_indices, 0],
            translations[arrow_indices, 1],
            translations[arrow_indices, 2],
            forward_world[arrow_indices, 0],
            forward_world[arrow_indices, 1],
            forward_world[arrow_indices, 2],
            length=2.5,
            normalize=True,
            color="tab:blue",
            alpha=0.58,
            linewidth=0.75,
            label="detector orientation" if plot_index == 1 else None,
        )

        for source in case["true_sources"]:
            source_strength = float(source["strength_per_second"])
            plot_source_cloud(ax, source, strength_cmap(strength_norm(source_strength)))

        ax.set_title(
            f"Source case {plot_index} ({len(case['true_sources'])} sources)",
            fontsize=10,
            pad=8,
        )
        set_axes(ax)
        if plot_index == 1:
            ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(
        "Synthetic source scenarios and detector trajectory",
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(left=0.035, right=0.84, bottom=0.035, top=0.91, wspace=0.12, hspace=0.18)
    scalar_mappable = plt.cm.ScalarMappable(norm=strength_norm, cmap=strength_cmap)
    scalar_mappable.set_array([])
    colorbar_axis = fig.add_axes([0.875, 0.20, 0.026, 0.58])
    colorbar = fig.colorbar(scalar_mappable, cax=colorbar_axis)
    colorbar.set_label("source activity [counts/s]")

    output_path = OUTPUT_DIR / "source_cases_trajectory_overview.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
