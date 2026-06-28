"""
Objective/model helpers for OpenGATE-based multi-point reconstruction.

H is built directly as a Torch tensor by interpolating an OpenGATE relative
response table. There is no separate NumPy reference path here.
"""

import torch

from common.geometry import EPS
from relative_response_table import load_relative_response_table


def poisson_negative_log_likelihood(counts, mean_counts):
    """Poisson negative log-likelihood without the constant log(y!) term."""
    counts = torch.as_tensor(counts)
    mean_counts = torch.clamp(torch.as_tensor(mean_counts, device=counts.device), min=EPS)
    return float(torch.sum(mean_counts - counts * torch.log(mean_counts)).detach().cpu())


def _axis_brackets(axis, values):
    """Vectorized interpolation brackets for one sorted Torch axis."""
    if axis.numel() == 1:
        zeros = torch.zeros_like(values, dtype=torch.long)
        weights = torch.zeros_like(values)
        valid = torch.ones_like(values, dtype=torch.bool)
        return zeros, zeros, weights, valid

    valid = (values >= axis[0]) & (values <= axis[-1])
    clamped = torch.clamp(values, min=float(axis[0]), max=float(axis[-1]))
    high = torch.searchsorted(axis, clamped, right=False)
    high = torch.clamp(high, min=0, max=axis.numel() - 1)
    low = torch.clamp(high - 1, min=0, max=axis.numel() - 1)
    low = torch.where(high == 0, high, low)

    low_values = axis[low]
    high_values = axis[high]
    span = high_values - low_values
    weights = torch.where(
        torch.abs(span) > EPS,
        (clamped - low_values) / torch.clamp(span, min=EPS),
        torch.zeros_like(clamped),
    )
    return low, high, weights, valid


def _interpolate_channels_for_local_sources(source_local_cm, response_grid):
    """
    Interpolate channel responses for many source positions in detector-local cm.

    response_grid keys: x_axis, y_axis, z_axis, values. values has shape
    x, y, z, channels. Points outside the sampled grid raise ValueError.
    """
    x0, x1, tx, valid_x = _axis_brackets(response_grid["x_axis"], source_local_cm[:, 0])
    y0, y1, ty, valid_y = _axis_brackets(response_grid["y_axis"], source_local_cm[:, 1])
    z0, z1, tz, valid_z = _axis_brackets(response_grid["z_axis"], source_local_cm[:, 2])
    valid = valid_x & valid_y & valid_z
    if not bool(torch.all(valid).detach().cpu()):
        invalid_indices = torch.nonzero(~valid, as_tuple=False).flatten()
        first_invalid = source_local_cm[invalid_indices[0]].detach().cpu().tolist()
        invalid_count = int(invalid_indices.numel())
        total_count = int(source_local_cm.shape[0])
        x_axis = response_grid["x_axis"].detach().cpu()
        y_axis = response_grid["y_axis"].detach().cpu()
        z_axis = response_grid["z_axis"].detach().cpu()
        raise ValueError(
            "OpenGATE relative response table does not cover all requested "
            "source_local positions. "
            f"Outside grid: {invalid_count}/{total_count}. "
            f"First outside source_local_cm={first_invalid}. "
            f"Covered ranges: "
            f"x=[{float(x_axis[0])}, {float(x_axis[-1])}], "
            f"y=[{float(y_axis[0])}, {float(y_axis[-1])}], "
            f"z=[{float(z_axis[0])}, {float(z_axis[-1])}]."
        )

    values = response_grid["values"]
    num_sources = int(source_local_cm.shape[0])
    num_channels = int(values.shape[-1])
    response = torch.zeros(
        (num_sources, num_channels),
        dtype=values.dtype,
        device=values.device,
    )

    for xi, wx in ((x0, 1.0 - tx), (x1, tx)):
        for yi, wy in ((y0, 1.0 - ty), (y1, ty)):
            for zi, wz in ((z0, 1.0 - tz), (z1, tz)):
                weight = (wx * wy * wz).unsqueeze(1)
                response += weight * values[xi, yi, zi, :]

    return response


def _torch_response_grid(response_table, device, dtype):
    """Move the OpenGATE response grid to the Torch device."""
    return {
        "x_axis": torch.as_tensor(response_table.x_axis, dtype=dtype, device=device),
        "y_axis": torch.as_tensor(response_table.y_axis, dtype=dtype, device=device),
        "z_axis": torch.as_tensor(response_table.z_axis, dtype=dtype, device=device),
        "values": torch.as_tensor(
            response_table.values,
            dtype=dtype,
            device=device,
        ),
    }


def build_system_matrix(
    candidate_positions,
    rotations,
    translations,
    relative_response_table_path,
    response_column="response_weighted_probability",
    activity_to_emission_scale=1.0,
    device=None,
    dtype=torch.float32,
):
    """
    Build H for reconstruction from the OpenGATE relative-response table.

    H[row, voxel] is the expected detector count contribution of activity=1 in
    that candidate voxel. If the table stores probabilities per emitted photon,
    activity_to_emission_scale says how many emitted photons activity=1 means
    during one measurement.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    response_table = load_relative_response_table(
        str(relative_response_table_path),
        response_column=response_column,
    )
    response_grid = _torch_response_grid(response_table, device=device, dtype=dtype)

    candidate_positions = torch.as_tensor(candidate_positions, dtype=dtype, device=device)
    rotations = torch.as_tensor(rotations, dtype=dtype, device=device)
    translations = torch.as_tensor(translations, dtype=dtype, device=device)

    num_measurements = int(translations.shape[0])
    num_channels = int(response_table.num_channels)
    num_voxels = int(candidate_positions.shape[0])
    system_matrix = torch.empty(
        (num_measurements * num_channels, num_voxels),
        dtype=dtype,
        device=device,
    )

    scale = float(activity_to_emission_scale)
    for measurement_index in range(num_measurements):
        row_start = measurement_index * num_channels
        source_local_cm = (
            candidate_positions - translations[measurement_index]
        ) @ rotations[measurement_index]
        response = _interpolate_channels_for_local_sources(
            source_local_cm=source_local_cm,
            response_grid=response_grid,
        )
        system_matrix[row_start : row_start + num_channels, :] = (scale * response).T

    return system_matrix


def predict_counts_from_activity(backgrounds_flat, system_matrix, activity):
    """Predict flattened detector counts from background and voxel activity."""
    return backgrounds_flat + system_matrix @ activity
