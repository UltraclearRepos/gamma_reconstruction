"""
Reconstruction algorithms for estimating the source position from measurements.
"""

import numpy as np

from reconstruction_objective import score_candidate_position


def reconstruct_source(
    counts,
    geometry,
    rotations,
    translations,
    assumed_backgrounds,
    assumed_channel_scales,
    assumed_k,
    search_min,
    search_max,
    method="coarse_to_fine",
    grid_sizes=(13, 11, 9),
    grid_size=21,
    random_samples=4000,
    random_seed=0,
):
    """
    Run one reconstruction method using a plain if/elif dispatcher.

    Returns:
        estimated_position, estimated_strength, history
    """
    method = method.strip().lower().replace("-", "_")

    if method == "coarse_to_fine":
        return coarse_to_fine_grid_search(
            counts=counts,
            geometry=geometry,
            rotations=rotations,
            translations=translations,
            assumed_backgrounds=assumed_backgrounds,
            assumed_channel_scales=assumed_channel_scales,
            assumed_k=assumed_k,
            search_min=search_min,
            search_max=search_max,
            grid_sizes=grid_sizes,
        )

    if method == "grid_search":
        return grid_search(
            counts=counts,
            geometry=geometry,
            rotations=rotations,
            translations=translations,
            assumed_backgrounds=assumed_backgrounds,
            assumed_channel_scales=assumed_channel_scales,
            assumed_k=assumed_k,
            search_min=search_min,
            search_max=search_max,
            grid_size=grid_size,
        )

    if method == "random_search":
        return random_search(
            counts=counts,
            geometry=geometry,
            rotations=rotations,
            translations=translations,
            assumed_backgrounds=assumed_backgrounds,
            assumed_channel_scales=assumed_channel_scales,
            assumed_k=assumed_k,
            search_min=search_min,
            search_max=search_max,
            random_samples=random_samples,
            random_seed=random_seed,
        )

    raise ValueError(
        "Unknown reconstruction method "
        f"'{method}'. Use: coarse_to_fine, grid_search, random_search."
    )


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
        level_candidates = []
        level_scores = []

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

                    level_candidates.append(candidate)
                    level_scores.append(score)

                    if score < best_score:
                        best_score = score
                        best_position = candidate
                        best_strength = strength

        history.append(
            {
                "level": float(level_index),
                "grid_size": float(grid_size),
                "search_center": center.copy(),
                "search_half_width": half_width.copy(),
                "best_score": best_score,
                "best_position": best_position.copy(),
                "best_strength": float(best_strength),
                "candidate_positions": np.array(level_candidates, dtype=float),
                "candidate_scores": np.array(level_scores, dtype=float),
            }
        )

        # Shrink the search box around the best candidate found so far.
        center = best_position.copy()
        half_width = half_width / 3.0

    return best_position, best_strength, history


def grid_search(
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
    """Evaluate one full 3D grid and return the best grid point."""
    xs = np.linspace(search_min[0], search_max[0], grid_size)
    ys = np.linspace(search_min[1], search_max[1], grid_size)
    zs = np.linspace(search_min[2], search_max[2], grid_size)

    candidates = []
    scores = []
    best_position = None
    best_strength = 0.0
    best_score = float("inf")

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

                candidates.append(candidate)
                scores.append(score)

                if score < best_score:
                    best_score = score
                    best_position = candidate
                    best_strength = strength

    history = [
        {
            "grid_size": float(grid_size),
            "evaluations": float(len(scores)),
            "best_score": best_score,
            "best_position": best_position.copy(),
            "best_strength": float(best_strength),
            "candidate_positions": np.array(candidates, dtype=float),
            "candidate_scores": np.array(scores, dtype=float),
        }
    ]

    return best_position, best_strength, history


def random_search(
    counts,
    geometry,
    rotations,
    translations,
    assumed_backgrounds,
    assumed_channel_scales,
    assumed_k,
    search_min,
    search_max,
    random_samples=4000,
    random_seed=0,
):
    """Sample random candidate positions and keep the best one."""
    search_min = np.array(search_min, dtype=float)
    search_max = np.array(search_max, dtype=float)
    rng = np.random.default_rng(random_seed)

    best_position = None
    best_strength = 0.0
    best_score = float("inf")
    history = []
    report_every = max(1, int(random_samples) // 10)
    chunk_candidates = []
    chunk_scores = []

    candidates = rng.uniform(search_min, search_max, size=(int(random_samples), 3))
    candidates = np.vstack([0.5 * (search_min + search_max), candidates])

    for index, candidate in enumerate(candidates, start=1):
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

        chunk_candidates.append(candidate)
        chunk_scores.append(score)

        if score < best_score:
            best_score = score
            best_position = candidate.copy()
            best_strength = strength

        if index == 1 or index % report_every == 0 or index == len(candidates):
            history.append(
                {
                    "checked_samples": float(index),
                    "best_score": best_score,
                    "best_position": best_position.copy(),
                    "best_strength": float(best_strength),
                    "candidate_positions": np.array(chunk_candidates, dtype=float),
                    "candidate_scores": np.array(chunk_scores, dtype=float),
                }
            )
            chunk_candidates = []
            chunk_scores = []

    return best_position, float(best_strength), history
