"""
Reconstruction helpers for estimating the source position from measurements.
"""

import numpy as np

from forward_model import compute_detector_response
from geometry import EPS


def estimate_strength_least_squares(
    counts,
    backgrounds,
    response,
):
    """
    Estimate source strength A for a fixed candidate source position.

    To keep the code simple, A is estimated by non-negative least squares
    in closed form:
        y - b ~= A * response
    """
    signal = counts.ravel() - np.broadcast_to(backgrounds, counts.shape).ravel()
    response_flat = response.ravel()

    denominator = float(np.dot(response_flat, response_flat))
    if denominator < EPS:
        return 0.0

    numerator = float(np.dot(response_flat, signal))
    return max(0.0, numerator / denominator)


def poisson_negative_log_likelihood(counts, mean_counts):
    """
    Poisson negative log-likelihood without the constant log(y!) term.

    The omitted term does not depend on the candidate source position,
    so it does not affect the argmin.
    """
    mean_counts = np.maximum(mean_counts, EPS)
    return float(np.sum(mean_counts - counts * np.log(mean_counts)))


def score_candidate_position(
    candidate_source_position,
    counts,
    geometry,
    rotations,
    translations,
    assumed_backgrounds,
    assumed_channel_scales,
    assumed_k,
):
    """Return (score, estimated_source_strength) for one candidate position."""
    response = compute_detector_response(
        source_position=candidate_source_position,
        geometry=geometry,
        rotations=rotations,
        translations=translations,
        channel_scales=assumed_channel_scales,
        k=assumed_k,
    )

    estimated_strength = estimate_strength_least_squares(
        counts=counts,
        backgrounds=assumed_backgrounds,
        response=response,
    )

    predicted_counts = assumed_backgrounds[None, :] + estimated_strength * response
    score = poisson_negative_log_likelihood(counts, predicted_counts)
    return score, estimated_strength


def compute_score_map_xyz(
    counts,
    geometry,
    rotations,
    translations,
    assumed_backgrounds,
    assumed_channel_scales,
    assumed_k,
    search_min,
    search_max,
    grid_size=21,
):
    """
    Evaluate reconstruction score on a full XYZ grid.

    Lower score means the candidate position explains the measured counts better.
    """
    xs = np.linspace(search_min[0], search_max[0], grid_size)
    ys = np.linspace(search_min[1], search_max[1], grid_size)
    zs = np.linspace(search_min[2], search_max[2], grid_size)
    score_map = np.zeros((grid_size, grid_size, grid_size), dtype=float)
    strength_map = np.zeros((grid_size, grid_size, grid_size), dtype=float)

    for y_index, y in enumerate(ys):
        for x_index, x in enumerate(xs):
            for z_index, z in enumerate(zs):
                candidate = np.array([x, y, z], dtype=float)

                score, strength = score_candidate_position(
                    candidate_source_position=candidate,
                    counts=counts,
                    geometry=geometry,
                    rotations=rotations,
                    translations=translations,
                    assumed_backgrounds=assumed_backgrounds,
                    assumed_channel_scales=assumed_channel_scales,
                    assumed_k=assumed_k,
                )

                score_map[y_index, x_index, z_index] = score
                strength_map[y_index, x_index, z_index] = strength

    return xs, ys, zs, score_map, strength_map


def coarse_to_fine_grid_search(
    counts,
    geometry,
    rotations,
    translations,
    assumed_backgrounds,
    assumed_channel_scales,
    assumed_k,
    search_min,
    search_max,
    grid_sizes=(13, 11, 9),
):
    """
    Reconstruct source position with several increasingly finer 3D grids.

    This is slower than gradient-based optimization, but it is easy to
    understand and robust as a first prototype.
    """
    search_min = np.array(search_min, dtype=float)
    search_max = np.array(search_max, dtype=float)

    center = 0.5 * (search_min + search_max)
    half_width = 0.5 * (search_max - search_min)

    best_position = center.copy()
    best_strength = 0.0
    best_score = float("inf")
    history = []

    for level_index, grid_size in enumerate(grid_sizes, start=1):
        xs = np.linspace(center[0] - half_width[0], center[0] + half_width[0], grid_size)
        ys = np.linspace(center[1] - half_width[1], center[1] + half_width[1], grid_size)
        zs = np.linspace(center[2] - half_width[2], center[2] + half_width[2], grid_size)

        for x in xs:
            for y in ys:
                for z in zs:
                    candidate = np.array([x, y, z], dtype=float)

                    score, strength = score_candidate_position(
                        candidate_source_position=candidate,
                        counts=counts,
                        geometry=geometry,
                        rotations=rotations,
                        translations=translations,
                        assumed_backgrounds=assumed_backgrounds,
                        assumed_channel_scales=assumed_channel_scales,
                        assumed_k=assumed_k,
                    )

                    if score < best_score:
                        best_score = score
                        best_position = candidate
                        best_strength = strength

        history.append(
            {
                "level": float(level_index),
                "grid_size": float(grid_size),
                "best_score": best_score,
                "best_x": best_position[0],
                "best_y": best_position[1],
                "best_z": best_position[2],
            }
        )

        # Shrink the search box around the best candidate found so far.
        center = best_position.copy()
        half_width = half_width / 3.0

    return best_position, best_strength, history
