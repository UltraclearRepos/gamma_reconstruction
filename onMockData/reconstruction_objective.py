"""
Objective functions used by reconstruction algorithms.
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
