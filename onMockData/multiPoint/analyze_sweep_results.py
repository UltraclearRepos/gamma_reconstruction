"""
Analyze reconstruction sweep CSV results and create focused visualizations.

The script does not run reconstruction. It reads the CSV produced by
run_parameter_sweep.py and writes separate plots/tables for:
- shape_pearson, where higher is better
- centroid_distance_mean_cm, where lower is better
"""

from pathlib import Path

import numpy as np
import pandas as pd

from run_demo import ON_MOCK_DATA_DIR


INPUT_CSV = (
    ON_MOCK_DATA_DIR
    / "multiPoint"
    / "visualizations"
    / "reconstruction_method_prior_sources_sweep_v2.csv"
)
OUTPUT_DIR = ON_MOCK_DATA_DIR / "multiPoint" / "visualizations" / "sweep_analysis"
TOP_N = 8
EXPECTED_METHODS = {"mlem", "map_em", "bsrem"}


NUMERIC_COLUMNS = [
    "elapsed_seconds",
    "system_matrix_seconds",
    "transfer_seconds",
    "iteration_seconds",
    "true_total_activity",
    "reconstructed_total_activity",
    "scale_to_true_total",
    "shape_cosine",
    "shape_pearson",
    "true_component_count",
    "reconstructed_component_count",
    "matched_component_count",
    "extra_reconstructed_count",
    "missed_true_count",
    "centroid_distance_mean_cm",
    "centroid_distance_max_cm",
    "closest_voxel_distance_mean_cm",
    "activity_recovery_mean",
    "matched_reconstructed_voxels_mean",
    "matched_true_voxels_mean",
    "matched_voxel_count_ratio_mean",
    "matched_reconstructed_voxels_total",
    "matched_true_voxels_total",
    "matched_voxel_count_ratio_total",
    "num_measurements",
    "num_iterations",
    "history_interval",
    "activity_component_threshold_fraction",
]


METRICS = [
    {
        "key": "pearson",
        "column": "shape_pearson",
        "label": "shape_pearson",
        "higher_is_better": True,
        "cmap": "viridis",
        "value_format": "{:.3f}",
    },
    {
        "key": "centroid_mean",
        "column": "centroid_distance_mean_cm",
        "label": "centroid_distance_mean_cm",
        "higher_is_better": False,
        "cmap": "viridis_r",
        "value_format": "{:.2f}",
    },
]


def safe_numeric(series, default=np.nan):
    """Convert a series to numeric values while keeping missing values tame."""
    return pd.to_numeric(series, errors="coerce").fillna(default)


def prior_family_from_name(prior_case):
    """Return a compact family label from generated prior_case names."""
    text = str(prior_case)
    if text == "no_priors" or not text:
        return "none"
    if "__" in text:
        return "combo"
    return text.split("_", 1)[0]


def prepare_dataframe(path):
    """Load CSV, coerce metrics, and keep only expected reconstruction methods."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    for column in NUMERIC_COLUMNS:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan

    required = [
        "reconstruction_method",
        "prior_case",
        "source_case",
        "shape_pearson",
        "centroid_distance_mean_cm",
        "true_component_count",
        "reconstructed_component_count",
        "matched_component_count",
        "missed_true_count",
        "extra_reconstructed_count",
    ]
    missing = [column for column in required if column not in df]
    if missing:
        raise KeyError(f"Missing required CSV columns: {missing}")

    df["reconstruction_method"] = df["reconstruction_method"].astype(str)
    valid_method_mask = df["reconstruction_method"].isin(EXPECTED_METHODS)
    dropped_rows = int((~valid_method_mask).sum())
    if dropped_rows:
        print(
            f"Dropped {dropped_rows} rows with invalid reconstruction_method. "
            "This usually means the CSV was appended after the column schema changed."
        )
    df = df[valid_method_mask].copy()
    if df.empty:
        raise ValueError(
            "No valid sweep rows left after filtering reconstruction_method. "
            "Delete or rename the old CSV and rerun run_parameter_sweep.py."
        )

    df["prior_family"] = df["prior_case"].map(prior_family_from_name)
    df["blob_count_exact"] = (
        safe_numeric(df["reconstructed_component_count"])
        == safe_numeric(df["true_component_count"])
    )
    df["all_true_blobs_matched"] = (
        safe_numeric(df["matched_component_count"])
        == safe_numeric(df["true_component_count"])
    ) & (safe_numeric(df["missed_true_count"]) == 0)
    df["blob_detection_exact"] = (
        df["blob_count_exact"]
        & df["all_true_blobs_matched"]
        & (safe_numeric(df["extra_reconstructed_count"]) == 0)
    )
    return df


def save_table(df, path, columns=None):
    """Save a table to CSV with useful rounded numeric values."""
    output = df.copy()
    if columns is not None:
        output = output[[column for column in columns if column in output.columns]]
    for column in output.select_dtypes(include=[np.number]).columns:
        output[column] = output[column].round(6)
    output.to_csv(path, index=False)


def metric_sort_values(metric, prefix="metric"):
    """Return sort columns and order for a metric table."""
    if metric["higher_is_better"]:
        return [f"{prefix}_mean", f"{prefix}_best"], [False, False]
    return [f"{prefix}_mean", f"{prefix}_best"], [True, True]


def sort_by_metric(df, metric, column=None):
    """Sort rows by one metric where best can be high or low."""
    if column is None:
        column = metric["column"]
    return df.sort_values(column, ascending=not metric["higher_is_better"])


def best_row_for_metric(df, metric):
    """Return the best concrete run for a metric."""
    return sort_by_metric(df, metric).iloc[0]


def top_columns(metric):
    """Columns kept in per-metric top-run CSV files."""
    metric_column = metric["column"]
    columns = [
        metric_column,
        "source_case",
        "reconstruction_method",
        "prior_case",
        "num_measurements",
        "num_iterations",
        "shape_pearson",
        "centroid_distance_mean_cm",
        "centroid_distance_max_cm",
        "shape_cosine",
        "activity_recovery_mean",
        "matched_component_count",
        "missed_true_count",
        "extra_reconstructed_count",
        "elapsed_seconds",
        "iteration_seconds",
    ]
    return list(dict.fromkeys(columns))


def best_rows_by_group(df, group_columns, metric, top_n):
    """Return top run rows for each group according to one metric."""
    return (
        sort_by_metric(df, metric)
        .groupby(group_columns, group_keys=False, dropna=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def aggregation_mean_over(group_columns):
    """Describe what is averaged after best-budget selection."""
    columns = tuple(group_columns)
    if columns == ("source_case", "reconstruction_method", "prior_case"):
        return "single source_case and reconstruction_method"
    if columns == ("reconstruction_method", "prior_case"):
        return "source_case"
    if columns == ("source_case", "prior_case"):
        return "available reconstruction_method"
    return "best-budget rows"


def aggregate_metric_settings(df, group_columns, metric):
    """
    Aggregate one metric by prior setting after budget selection.

    First keep only the best iteration/measurement budget for each
    source_case x reconstruction_method x prior_case. Then aggregate those
    best-budget rows into the requested group. For example, method+prior means
    are averages over source cases, not averages over every budget.
    """
    rows = []
    metric_column = metric["column"]
    budget_selected = best_prior_per_method_source(df, metric)
    for group_values, group in budget_selected.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        best_row = best_row_for_metric(group, metric)
        row = {column: value for column, value in zip(group_columns, group_values)}
        row.update(
            {
                "metric": metric_column,
                "mean_over": aggregation_mean_over(group_columns),
                "budget_selection": (
                    "best num_measurements+num_iterations per "
                    "source_case+reconstruction_method+prior_case"
                ),
                "metric_mean": float(group[metric_column].mean()),
                "metric_best": float(best_row[metric_column]),
                "metric_worst": float(
                    group[metric_column].min()
                    if metric["higher_is_better"]
                    else group[metric_column].max()
                ),
                "shape_pearson_mean": float(group["shape_pearson"].mean()),
                "centroid_distance_mean_cm": float(group["centroid_distance_mean_cm"].mean()),
                "activity_recovery_mean": float(group["activity_recovery_mean"].mean()),
                "blob_count_exact_fraction": float(group["blob_count_exact"].mean()),
                "blob_detection_exact_fraction": float(group["blob_detection_exact"].mean()),
                "true_component_count_mean": float(group["true_component_count"].mean()),
                "reconstructed_component_count_mean": float(
                    group["reconstructed_component_count"].mean()
                ),
                "matched_component_count_mean": float(group["matched_component_count"].mean()),
                "best_num_measurements": int(best_row["num_measurements"]),
                "best_num_iterations": int(best_row["num_iterations"]),
                "best_shape_pearson": float(best_row["shape_pearson"]),
                "best_centroid_distance_mean_cm": float(
                    best_row["centroid_distance_mean_cm"]
                ),
                "source_case_count": int(group["source_case"].nunique()),
                "method_count": int(group["reconstruction_method"].nunique()),
                "cases": int(len(group)),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    sort_columns, ascending = metric_sort_values(metric)
    return result.sort_values(sort_columns, ascending=ascending)


def best_prior_per_method_source(df, metric):
    """Keep the best run for each prior in each algorithm and source scenario."""
    return (
        sort_by_metric(df, metric)
        .groupby(["prior_case", "reconstruction_method", "source_case"], dropna=False)
        .head(1)
        .reset_index(drop=True)
    )


def write_metric_tables(df, output_dir, metric, top_n):
    """Write top-run CSV files for one metric."""
    output_dir.mkdir(parents=True, exist_ok=True)
    key = metric["key"]
    sorted_df = sort_by_metric(df, metric)

    save_table(
        sorted_df.head(top_n),
        output_dir / f"top_overall_{key}.csv",
        top_columns(metric),
    )
    save_table(
        best_rows_by_group(df, ["reconstruction_method"], metric, top_n),
        output_dir / f"top_runs_by_method_{key}.csv",
        top_columns(metric),
    )
    save_table(
        best_rows_by_group(df, ["source_case", "reconstruction_method"], metric, top_n),
        output_dir / f"top_runs_by_source_and_method_{key}.csv",
        top_columns(metric),
    )
    return {"sorted": sorted_df}


def setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def save_current_figure(plt, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compact_label(text, max_length=58):
    """Shorten long prior names while keeping labels readable."""
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def axis_limits(values, metric):
    """Choose readable axis limits for one metric."""
    finite = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if finite.empty:
        return None
    min_value = float(finite.min())
    max_value = float(finite.max())
    if metric["column"] == "shape_pearson":
        return (min(-1.0, min_value), max(1.0, max_value))
    if min_value >= 0.0:
        return (0.0, max_value * 1.08 if max_value > 0.0 else 1.0)
    padding = 0.08 * max(abs(min_value), abs(max_value), 1.0)
    return (min_value - padding, max_value + padding)


def config_label(row):
    """Human-readable label for one source/method/prior configuration."""
    return compact_label(f"{row.source_case} | {row.reconstruction_method} | {row.prior_case}", 80)


def select_top_configurations(df, metric, top_n):
    """Select top source/method/prior configurations across measurements/iterations."""
    configs = aggregate_metric_settings(
        df,
        ["source_case", "reconstruction_method", "prior_case"],
        metric,
    )
    return configs.head(top_n)


def select_all_configurations(df, metric):
    """Select every source/method/prior configuration after budget ranking metadata."""
    return aggregate_metric_settings(
        df,
        ["source_case", "reconstruction_method", "prior_case"],
        metric,
    )


def plot_metric_vs_axis_top_configs(df, configs, axis_column, output_path, title, metric):
    """Plot one metric for selected configs while holding the other sweep axis fixed."""
    plt = setup_matplotlib()
    if configs.empty:
        return

    if axis_column == "num_measurements":
        fixed_axis = "num_iterations"
        fixed_value_column = "best_num_iterations"
    elif axis_column == "num_iterations":
        fixed_axis = "num_measurements"
        fixed_value_column = "best_num_measurements"
    else:
        raise ValueError(f"Unsupported sweep axis: {axis_column}")

    metric_column = metric["column"]
    best_agg = "max" if metric["higher_is_better"] else "min"
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for config in configs.itertuples():
        fixed_value = int(getattr(config, fixed_value_column))
        subset = df[
            (df["source_case"] == config.source_case)
            & (df["reconstruction_method"] == config.reconstruction_method)
            & (df["prior_case"] == config.prior_case)
            & (df[fixed_axis] == fixed_value)
        ]
        curve = (
            subset.groupby(axis_column, dropna=False)
            .agg(
                metric_mean=(metric_column, "mean"),
                metric_best=(metric_column, best_agg),
            )
            .reset_index()
            .sort_values(axis_column)
        )
        if curve.empty:
            continue
        ax.plot(
            curve[axis_column],
            curve["metric_mean"],
            marker="o",
            linewidth=1.8,
            label=compact_label(f"{config_label(config)} | fixed {fixed_axis}={fixed_value}", 95),
        )
        ax.scatter(
            curve[axis_column],
            curve["metric_best"],
            s=24,
            alpha=0.55,
        )

    ax.set_xlabel(axis_column)
    ax.set_ylabel(metric["label"])
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    save_current_figure(plt, output_path)


def plot_metric_vs_axis_all_configs(df, configs, axis_column, output_path, title, metric):
    """Plot one metric averaged over every available configuration for one sweep axis."""
    plt = setup_matplotlib()
    if df.empty:
        return

    if axis_column == "num_measurements":
        averaged_over = "source_case, reconstruction_method, prior_case, num_iterations"
    elif axis_column == "num_iterations":
        averaged_over = "source_case, reconstruction_method, prior_case, num_measurements"
    else:
        raise ValueError(f"Unsupported sweep axis: {axis_column}")

    metric_column = metric["column"]
    best_agg = "max" if metric["higher_is_better"] else "min"
    curve = (
        df.groupby(axis_column, dropna=False)
        .agg(
            metric_mean=(metric_column, "mean"),
            metric_best=(metric_column, best_agg),
            metric_worst=(
                metric_column,
                "min" if metric["higher_is_better"] else "max",
            ),
            cases=(metric_column, "size"),
        )
        .reset_index()
        .sort_values(axis_column)
    )
    if curve.empty:
        return

    save_table(curve, output_path.with_suffix(".csv"))

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(
        curve[axis_column],
        curve["metric_mean"],
        marker="o",
        linewidth=2.0,
        color="#386cb0",
        label="mean over all configs",
    )
    ax.scatter(
        curve[axis_column],
        curve["metric_best"],
        s=32,
        color="#f28e2b",
        label="best run at this value",
        zorder=3,
    )
    ax.scatter(
        curve[axis_column],
        curve["metric_worst"],
        s=26,
        color="#8c8c8c",
        label="worst run at this value",
        alpha=0.75,
        zorder=3,
    )
    for row in curve.itertuples():
        ax.annotate(
            f"n={int(row.cases)}",
            (getattr(row, axis_column), row.metric_mean),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7,
        )
    ax.set_xlabel(axis_column)
    ax.set_ylabel(metric["label"])
    ax.set_title(f"{title}\naveraged over {averaged_over}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.22)
    save_current_figure(plt, output_path)


def aggregate_elapsed_time(df):
    """Average total elapsed time over source cases for every method/prior/budget."""
    by_prior_budget = (
        df.groupby(
            [
                "reconstruction_method",
                "prior_case",
                "num_iterations",
                "num_measurements",
            ],
            dropna=False,
        )
        .agg(
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            elapsed_seconds_min=("elapsed_seconds", "min"),
            elapsed_seconds_max=("elapsed_seconds", "max"),
            elapsed_seconds_std=("elapsed_seconds", "std"),
            source_case_count=("source_case", "nunique"),
            cases=("elapsed_seconds", "size"),
        )
        .reset_index()
        .sort_values(
            [
                "reconstruction_method",
                "prior_case",
                "num_iterations",
                "num_measurements",
            ]
        )
    )
    by_prior_budget["mean_over"] = "source_case"

    by_method_budget = (
        df.groupby(["reconstruction_method", "num_iterations", "num_measurements"], dropna=False)
        .agg(
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            elapsed_seconds_min=("elapsed_seconds", "min"),
            elapsed_seconds_max=("elapsed_seconds", "max"),
            elapsed_seconds_std=("elapsed_seconds", "std"),
            prior_case_count=("prior_case", "nunique"),
            source_case_count=("source_case", "nunique"),
            cases=("elapsed_seconds", "size"),
        )
        .reset_index()
        .sort_values(["reconstruction_method", "num_iterations", "num_measurements"])
    )
    by_method_budget["mean_over"] = "source_case+prior_case"
    return by_prior_budget, by_method_budget


def plot_elapsed_time_vs_axis_by_method(table, axis_column, output_path, title):
    """
    Plot elapsed time as line charts.

    One figure contains multiple subplots for the fixed budget axis. Inside each
    subplot, every reconstruction method is a separate line.
    """
    plt = setup_matplotlib()
    if table.empty:
        return

    if axis_column == "num_measurements":
        fixed_column = "num_iterations"
        x_label = "NUM_MEASUREMENTS"
        fixed_label = "iterations"
    elif axis_column == "num_iterations":
        fixed_column = "num_measurements"
        x_label = "NUM_ITERATIONS"
        fixed_label = "measurements"
    else:
        raise ValueError(f"Unsupported elapsed-time axis: {axis_column}")

    fixed_values = sorted(table[fixed_column].dropna().unique())
    methods = sorted(table["reconstruction_method"].dropna().unique())
    if not fixed_values or not methods:
        return

    column_count = min(3, len(fixed_values))
    row_count = int(np.ceil(len(fixed_values) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, 4.0 * row_count),
        squeeze=False,
        sharey=True,
    )
    flat_axes = axes.ravel()

    for axis, fixed_value in zip(flat_axes, fixed_values):
        subset = table[table[fixed_column] == fixed_value]
        for method in methods:
            method_subset = (
                subset[subset["reconstruction_method"] == method]
                .sort_values(axis_column)
            )
            if method_subset.empty:
                continue
            axis.plot(
                method_subset[axis_column],
                method_subset["elapsed_seconds_mean"],
                marker="o",
                linewidth=1.8,
                label=method,
            )
        axis.set_title(f"{fixed_label}={int(fixed_value)}")
        axis.set_xlabel(x_label)
        axis.set_ylabel("total elapsed time [s]")
        axis.grid(True, alpha=0.22)

    for axis in flat_axes[len(fixed_values):]:
        axis.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    save_current_figure(plt, output_path)


def make_elapsed_time_outputs(df, output_dir):
    """Write elapsed-time tables and plots."""
    by_prior_budget, by_method_budget = aggregate_elapsed_time(df)
    save_table(
        by_prior_budget,
        output_dir / "elapsed_seconds_by_method_prior_iterations_measurements.csv",
    )
    save_table(
        by_method_budget,
        output_dir / "elapsed_seconds_by_method_iterations_measurements.csv",
    )
    plot_elapsed_time_vs_axis_by_method(
        by_method_budget,
        "num_iterations",
        output_dir / "elapsed_seconds_vs_iterations_by_method.png",
        "Total elapsed time vs iterations by method",
    )
    return {
        "by_prior_budget": by_prior_budget,
        "by_method_budget": by_method_budget,
    }


def filename_token(value):
    """Make a compact filename token."""
    text = str(value)
    safe = []
    for character in text:
        if character.isalnum() or character in ("-", "_"):
            safe.append(character)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def aggregate_priors_by_algorithm_source_valid_count(df, metric):
    """
    Aggregate priors for per-algorithm/source plots.

    Only rows with the correct number of reconstructed components are included.
    The metric is averaged over all iterations and measurements.
    """
    metric_column = metric["column"]
    filtered = df[df["blob_count_exact"]].copy()
    if filtered.empty:
        return pd.DataFrame()

    best_agg = "max" if metric["higher_is_better"] else "min"
    worst_agg = "min" if metric["higher_is_better"] else "max"
    base_group_columns = ["reconstruction_method", "source_case", "prior_case"]
    table = (
        filtered.groupby(
            base_group_columns,
            dropna=False,
        )
        .agg(
            metric_mean=(metric_column, "mean"),
            metric_best=(metric_column, best_agg),
            metric_worst=(metric_column, worst_agg),
            metric_std=(metric_column, "std"),
            cases=(metric_column, "size"),
            iteration_count=("num_iterations", "nunique"),
            measurement_count=("num_measurements", "nunique"),
            true_component_count_mean=("true_component_count", "mean"),
            reconstructed_component_count_mean=("reconstructed_component_count", "mean"),
            matched_component_count_mean=("matched_component_count", "mean"),
            matched_reconstructed_voxels_mean=(
                "matched_reconstructed_voxels_mean",
                "mean",
            ),
            matched_true_voxels_mean=("matched_true_voxels_mean", "mean"),
            matched_voxel_count_ratio_mean=("matched_voxel_count_ratio_mean", "mean"),
            matched_reconstructed_voxels_total_mean=(
                "matched_reconstructed_voxels_total",
                "mean",
            ),
            matched_true_voxels_total_mean=("matched_true_voxels_total", "mean"),
            matched_voxel_count_ratio_total_mean=(
                "matched_voxel_count_ratio_total",
                "mean",
            ),
        )
        .reset_index()
    )
    best_budget_rows = (
        sort_by_metric(filtered, metric)
        .groupby(base_group_columns, dropna=False)
        .head(1)
        .loc[
            :,
            [
                "reconstruction_method",
                "source_case",
                "prior_case",
                "num_iterations",
                "num_measurements",
            ],
        ]
        .rename(
            columns={
                "num_iterations": "best_num_iterations",
                "num_measurements": "best_num_measurements",
            }
        )
    )
    table = table.merge(best_budget_rows, on=base_group_columns, how="left")
    table["best_budget"] = table.apply(
        lambda row: f"I{int(row.best_num_iterations)} M{int(row.best_num_measurements)}",
        axis=1,
    )
    budget_counts = (
        filtered.groupby(
            [
                "reconstruction_method",
                "source_case",
                "prior_case",
                "num_iterations",
                "num_measurements",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="budget_case_count")
        .sort_values(
            [
                "reconstruction_method",
                "source_case",
                "prior_case",
                "budget_case_count",
            ],
            ascending=[True, True, True, False],
        )
        .groupby(base_group_columns, dropna=False)
        .head(1)
        .rename(
            columns={
                "num_iterations": "most_common_num_iterations",
                "num_measurements": "most_common_num_measurements",
            }
        )
    )
    table = table.merge(
        budget_counts[
            [
                "reconstruction_method",
                "source_case",
                "prior_case",
                "most_common_num_iterations",
                "most_common_num_measurements",
                "budget_case_count",
            ]
        ],
        on=base_group_columns,
        how="left",
    )
    table["most_common_budget"] = table.apply(
        lambda row: (
            f"I{int(row.most_common_num_iterations)} "
            f"M{int(row.most_common_num_measurements)}"
        ),
        axis=1,
    )
    table["mean_over"] = "iterations+measurements"
    table["row_filter"] = "reconstructed_component_count == true_component_count"
    sort_columns, ascending = metric_sort_values(metric)
    return table.sort_values(sort_columns, ascending=ascending)


def plot_priors_per_algorithm_by_source(table, output_dir, metric, top_n):
    """Write one prior-ranking PNG per reconstruction algorithm."""
    plt = setup_matplotlib()
    if table.empty:
        return

    sort_columns, ascending = metric_sort_values(metric)
    for method, method_table in table.groupby("reconstruction_method", dropna=False):
        source_groups = list(method_table.groupby("source_case", dropna=False))
        if not source_groups:
            continue

        column_count = min(2, len(source_groups))
        row_count = int(np.ceil(len(source_groups) / column_count))
        fig, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(13, max(4.4, 3.8 * row_count)),
            squeeze=False,
        )
        flat_axes = axes.ravel()

        for axis, (source_case, group) in zip(flat_axes, source_groups):
            top = group.sort_values(sort_columns, ascending=ascending).head(top_n)
            labels = [compact_label(row.prior_case, 62) for row in top.itertuples()]
            y = np.arange(len(top))
            axis.barh(y, top["metric_mean"], color="#386cb0", label="mean over I/M")
            axis.scatter(top["metric_best"], y, color="#f28e2b", s=24, label="best run")
            for row_index, row in enumerate(top.itertuples()):
                axis.annotate(
                    f"{row.best_budget}; n={int(row.cases)}",
                    (row.metric_mean, row_index),
                    textcoords="offset points",
                    xytext=(5, 0),
                    va="center",
                    fontsize=7,
                )
            axis.set_yticks(y)
            axis.set_yticklabels(labels, fontsize=8)
            axis.invert_yaxis()
            limits = axis_limits(pd.concat([top["metric_mean"], top["metric_best"]]), metric)
            if limits:
                axis.set_xlim(*limits)
            axis.set_title(f"source_case={source_case}")
            axis.set_xlabel(f"{metric['label']} mean over iterations+measurements")
            axis.legend(loc="best", fontsize=8)

        for axis in flat_axes[len(source_groups):]:
            axis.axis("off")

        fig.suptitle(
            f"Best priors for {method}: {metric['label']} "
            "(correct source count only)"
        )
        save_current_figure(
            plt,
            output_dir
            / f"best_priors_by_true_sources_{filename_token(method)}_{metric['key']}_valid_count.png",
        )


def make_prior_outputs(df, output_dir, metric, top_n):
    """Write simplified prior rankings per algorithm and true-source scenario."""
    table = aggregate_priors_by_algorithm_source_valid_count(df, metric)
    if table.empty:
        print(
            f"No rows with correct source count for prior plots: {metric['label']}"
        )
        return table

    save_table(
        table,
        output_dir / f"best_prior_settings_by_algorithm_true_sources_{metric['key']}_valid_count.csv",
    )
    plot_priors_per_algorithm_by_source(table, output_dir, metric, top_n)
    return table


def threshold_directory_name(threshold):
    """Return a stable directory name for one component threshold value."""
    return f"threshold_{filename_token(f'{float(threshold):g}')}"


def write_report(df, all_tables, output_dir):
    """Write a compact markdown report pointing to the generated artifacts."""
    report_path = output_dir / "summary.md"
    lines = [
        "# Sweep Analysis Summary",
        "",
        f"Rows analyzed: {len(df)}",
        f"Methods: {', '.join(map(str, sorted(df['reconstruction_method'].dropna().unique())))}",
        f"Source cases: {', '.join(map(str, sorted(df['source_case'].dropna().unique())))}",
        "",
    ]
    for metric in METRICS:
        key = metric["key"]
        best = all_tables[key]["sorted"].head(min(TOP_N, 8))
        lines.extend(
            [
                f"## Top Runs By {metric['label']}",
                "",
            ]
        )
        for index, row in enumerate(best.itertuples(), start=1):
            metric_value = getattr(row, metric["column"])
            lines.append(
                f"{index}. {metric['label']}={metric_value:.4f}, "
                f"method={row.reconstruction_method}, source={row.source_case}, "
                f"priors={row.prior_case}, measurements={int(row.num_measurements)}, "
                f"iterations={int(row.num_iterations)}, "
                f"pearson={row.shape_pearson:.4f}, "
                f"centroid_mean={row.centroid_distance_mean_cm:.3f} cm"
            )
        lines.append("")

    lines.extend(
        [
            "## Files",
            "",
            "Every ranking is generated separately for pearson and centroid_mean.",
            "Use files ending in `_pearson` for shape correlation and `_centroid_mean` for localization error.",
            "Threshold-specific plots are written into `threshold_*` subdirectories.",
            "Elapsed-time files use `elapsed_seconds` averaged over source cases for each method/prior/budget.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def make_metric_outputs(df, metric, output_dir):
    """Write all plots and tables for one metric."""
    key = metric["key"]
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = write_metric_tables(df, output_dir, metric, TOP_N)
    make_prior_outputs(df, output_dir, metric, TOP_N)
    top_configs = select_top_configurations(df, metric, TOP_N)
    all_configs = select_all_configurations(df, metric)
    plot_metric_vs_axis_top_configs(
        df,
        top_configs,
        "num_measurements",
        output_dir / f"{key}_vs_measurements_top_configs.png",
        f"{metric['label']} vs number of measurements for top source/method/prior configs",
        metric,
    )
    plot_metric_vs_axis_top_configs(
        df,
        top_configs,
        "num_iterations",
        output_dir / f"{key}_vs_iterations_top_configs.png",
        f"{metric['label']} vs iterations for top source/method/prior configs",
        metric,
    )
    plot_metric_vs_axis_all_configs(
        df,
        all_configs,
        "num_measurements",
        output_dir / f"{key}_vs_measurements_all_configs.png",
        f"{metric['label']} vs number of measurements for all source/method/prior configs",
        metric,
    )
    plot_metric_vs_axis_all_configs(
        df,
        all_configs,
        "num_iterations",
        output_dir / f"{key}_vs_iterations_all_configs.png",
        f"{metric['label']} vs iterations for all source/method/prior configs",
        metric,
    )
    return tables


def make_threshold_metric_outputs(df, metric, output_dir):
    """Write the same metric outputs separately for every component threshold."""
    threshold_column = "activity_component_threshold_fraction"
    thresholds = sorted(df[threshold_column].dropna().unique())
    for threshold in thresholds:
        threshold_df = df[df[threshold_column].eq(threshold)].copy()
        threshold_dir = output_dir / threshold_directory_name(threshold)
        make_metric_outputs(threshold_df, metric, threshold_dir)


def main():
    df = prepare_dataframe(INPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tables = {}
    for metric in METRICS:
        all_tables[metric["key"]] = make_metric_outputs(df, metric, OUTPUT_DIR)
        make_threshold_metric_outputs(df, metric, OUTPUT_DIR)
    make_elapsed_time_outputs(df, OUTPUT_DIR)
    write_report(df, all_tables, OUTPUT_DIR)

    print(f"Rows analyzed: {len(df)}")
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
