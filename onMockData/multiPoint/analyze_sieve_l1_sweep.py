"""
Analyze threshold/sieve/L1 sweep results.

The script reads reconstruction_sieve_l1_sweep.csv and writes plots/tables for
configurations that reconstructed the correct number of source components.
It does not run reconstruction.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from run_demo import ON_MOCK_DATA_DIR


INPUT_CSV = (
    ON_MOCK_DATA_DIR
    / "multiPoint"
    / "visualizations"
    / "reconstruction_sieve_l1_sweep.csv"
)
OUTPUT_DIR = ON_MOCK_DATA_DIR / "multiPoint" / "visualizations" / "sieve_l1_sweep_analysis"
TOP_N = 10

NUMERIC_COLUMNS = [
    "num_measurements",
    "num_iterations",
    "history_interval",
    "reconstruction_sieve_sigma_cm",
    "activity_component_threshold_fraction",
    "elapsed_seconds",
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
]

METRICS = [
    {
        "key": "pearson",
        "column": "shape_pearson",
        "label": "shape_pearson",
        "higher_is_better": True,
        "cmap": "viridis",
        "value_format": "{:.3f}",
        "require_exact_count": False,
    },
    {
        "key": "centroid_mean",
        "column": "centroid_distance_mean_cm",
        "label": "centroid_distance_mean_cm [cm]",
        "higher_is_better": False,
        "cmap": "viridis_r",
        "value_format": "{:.2f}",
        "require_exact_count": True,
    },
]

CONFIG_COLUMNS = [
    "reconstruction_method",
    "source_case",
    "prior_case",
    "l1_beta",
    "reconstruction_sieve_sigma_cm",
    "activity_component_threshold_fraction",
]


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )
    return plt


def filename_token(value):
    text = str(value)
    safe = []
    for character in text:
        if character.isalnum() or character in ("-", "_"):
            safe.append(character)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def source_sort_key(value):
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def l1_beta_from_row(row):
    priors_text = row.get("reconstruction_priors", "")
    if isinstance(priors_text, str) and priors_text.strip():
        try:
            priors = json.loads(priors_text)
            for prior in priors:
                if prior.get("type") == "l1":
                    return float(prior.get("beta", 0.0))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    prior_case = str(row.get("prior_case", ""))
    if prior_case == "no_priors":
        return 0.0
    marker = "l1_beta_"
    if marker not in prior_case:
        return np.nan
    token = prior_case.split(marker, 1)[1].split("__", 1)[0]
    try:
        return float(token.replace("_", "."))
    except ValueError:
        return np.nan


def l1_label(prior_case, beta):
    if str(prior_case) == "no_priors" or float(beta) == 0.0:
        return "none"
    return f"l1={float(beta):g}"


def config_label(row):
    return (
        f"thr={float(row.activity_component_threshold_fraction):g} | "
        f"sieve={float(row.reconstruction_sieve_sigma_cm):g} | "
        f"{l1_label(row.prior_case, row.l1_beta)}"
    )


def prepare_dataframe(path):
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
        "source_case",
        "prior_case",
        "reconstruction_priors",
        "reconstruction_sieve_sigma_cm",
        "activity_component_threshold_fraction",
        "shape_pearson",
        "centroid_distance_mean_cm",
        "true_component_count",
        "reconstructed_component_count",
    ]
    missing = [column for column in required if column not in df]
    if missing:
        raise KeyError(f"Missing required CSV columns: {missing}")

    df["source_case"] = df["source_case"].astype(str)
    df["l1_beta"] = df.apply(l1_beta_from_row, axis=1)
    df["blob_count_exact"] = df["reconstructed_component_count"].eq(
        df["true_component_count"]
    )
    df["configuration_label"] = df.apply(config_label, axis=1)
    return df


def metric_sort(metric):
    if metric["higher_is_better"]:
        return ["metric_mean", "metric_best", "centroid_mean"], [False, False, True]
    return ["metric_mean", "metric_best", "pearson_mean"], [True, True, False]


def metric_filter_label(metric):
    if metric.get("require_exact_count", True):
        return "reconstructed_component_count == true_component_count"
    return "all rows"


def metric_filter_token(metric):
    if metric.get("require_exact_count", True):
        return "valid_count"
    return "all_rows"


def sort_rows_by_metric(df, metric):
    return df.sort_values(
        metric["column"],
        ascending=not metric["higher_is_better"],
        na_position="last",
    )


def aggregate_configs(valid_df, metric):
    metric_column = metric["column"]
    best_agg = "max" if metric["higher_is_better"] else "min"
    worst_agg = "min" if metric["higher_is_better"] else "max"

    table = (
        valid_df.groupby(CONFIG_COLUMNS, dropna=False)
        .agg(
            metric_mean=(metric_column, "mean"),
            metric_best=(metric_column, best_agg),
            metric_worst=(metric_column, worst_agg),
            metric_std=(metric_column, "std"),
            cases=(metric_column, "size"),
            pearson_mean=("shape_pearson", "mean"),
            centroid_mean=("centroid_distance_mean_cm", "mean"),
            activity_recovery_mean=("activity_recovery_mean", "mean"),
            matched_reconstructed_voxels_mean=(
                "matched_reconstructed_voxels_mean",
                "mean",
            ),
            matched_true_voxels_mean=("matched_true_voxels_mean", "mean"),
            matched_voxel_count_ratio_mean=("matched_voxel_count_ratio_mean", "mean"),
            iteration_count=("num_iterations", "nunique"),
            measurement_count=("num_measurements", "nunique"),
        )
        .reset_index()
    )

    best_rows = (
        sort_rows_by_metric(valid_df, metric)
        .groupby(CONFIG_COLUMNS, dropna=False)
        .head(1)
        .loc[
            :,
            [
                *CONFIG_COLUMNS,
                "num_iterations",
                "num_measurements",
                "shape_pearson",
                "centroid_distance_mean_cm",
            ],
        ]
        .rename(
            columns={
                "num_iterations": "best_num_iterations",
                "num_measurements": "best_num_measurements",
                "shape_pearson": "best_run_shape_pearson",
                "centroid_distance_mean_cm": "best_run_centroid_distance_mean_cm",
            }
        )
    )
    table = table.merge(best_rows, on=CONFIG_COLUMNS, how="left")
    table["best_budget"] = table.apply(
        lambda row: f"I{int(row.best_num_iterations)} M{int(row.best_num_measurements)}",
        axis=1,
    )
    table["configuration_label"] = table.apply(config_label, axis=1)
    table["row_filter"] = metric_filter_label(metric)
    sort_columns, ascending = metric_sort(metric)
    return table.sort_values(sort_columns, ascending=ascending)


def save_table(table, path):
    rounded = table.copy()
    for column in rounded.select_dtypes(include=[np.number]).columns:
        rounded[column] = rounded[column].round(6)
    path.parent.mkdir(parents=True, exist_ok=True)
    rounded.to_csv(path, index=False)


def axis_limits(values, metric):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    low = float(values.min())
    high = float(values.max())
    if low == high:
        pad = abs(low) * 0.05 or 1.0
        return low - pad, high + pad
    pad = (high - low) * 0.08
    if metric["higher_is_better"]:
        return max(0.0, low - pad), high + pad
    return max(0.0, low - pad), high + pad


def plot_best_configs_by_algorithm_source(table, output_dir, metric):
    """Write one bar-chart PNG per algorithm, with source cases as subplots."""
    plt = setup_matplotlib()
    if table.empty:
        return

    sort_columns, ascending = metric_sort(metric)
    for method, method_table in table.groupby("reconstruction_method", dropna=False):
        source_cases = sorted(method_table["source_case"].unique(), key=source_sort_key)
        if not source_cases:
            continue

        fig, axes = plt.subplots(
            len(source_cases),
            1,
            figsize=(13, max(4.2, 3.2 * len(source_cases))),
            squeeze=False,
        )
        flat_axes = axes.ravel()

        for axis, source_case in zip(flat_axes, source_cases):
            group = method_table[method_table["source_case"] == source_case]
            top = group.sort_values(sort_columns, ascending=ascending).head(TOP_N)
            y = np.arange(len(top))
            axis.barh(y, top["metric_mean"], color="#386cb0", label="mean over I/M")
            axis.scatter(top["metric_best"], y, color="#f28e2b", s=24, label="best run")
            axis.set_yticks(y)
            axis.set_yticklabels(top["configuration_label"], fontsize=8)
            axis.invert_yaxis()
            axis.set_title(f"source_case={source_case}")
            axis.set_xlabel(metric["label"])
            limits = axis_limits(pd.concat([top["metric_mean"], top["metric_best"]]), metric)
            if limits:
                axis.set_xlim(*limits)
            for index, row in enumerate(top.itertuples()):
                axis.annotate(
                    f"{row.best_budget}; n={int(row.cases)}",
                    (row.metric_mean, index),
                    textcoords="offset points",
                    xytext=(5, 0),
                    va="center",
                    fontsize=7,
                )
            axis.legend(loc="best", fontsize=8)

        fig.suptitle(
            f"Best threshold/sieve/L1 configs for {method}: {metric['label']} "
            f"({metric_filter_label(metric)})"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(
            output_dir
            / f"best_threshold_sieve_l1_by_source_{filename_token(method)}_{metric['key']}.png",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_heatmaps_by_algorithm_source(valid_df, output_dir, metric):
    """Write one heatmap grid per algorithm: rows=sources, columns=L1 variants."""
    plt = setup_matplotlib()
    metric_column = metric["column"]
    for method, method_df in valid_df.groupby("reconstruction_method", dropna=False):
        source_cases = sorted(method_df["source_case"].unique(), key=source_sort_key)
        prior_rows = (
            method_df[["prior_case", "l1_beta"]]
            .drop_duplicates()
            .sort_values(["l1_beta", "prior_case"], na_position="last")
        )
        prior_cases = list(prior_rows.itertuples(index=False, name=None))
        if not source_cases or not prior_cases:
            continue

        vmin = method_df[metric_column].min()
        vmax = method_df[metric_column].max()
        fig, axes = plt.subplots(
            len(source_cases),
            len(prior_cases),
            figsize=(max(14, 3.4 * len(prior_cases)), max(8, 2.8 * len(source_cases))),
            squeeze=False,
        )
        image = None
        for row_index, source_case in enumerate(source_cases):
            for col_index, (prior_case, beta) in enumerate(prior_cases):
                axis = axes[row_index, col_index]
                group = method_df[
                    (method_df["source_case"] == source_case)
                    & (method_df["prior_case"] == prior_case)
                ]
                pivot = group.pivot_table(
                    index="activity_component_threshold_fraction",
                    columns="reconstruction_sieve_sigma_cm",
                    values=metric_column,
                    aggfunc="mean",
                ).sort_index().sort_index(axis=1)
                if pivot.empty:
                    axis.axis("off")
                    continue

                values = pivot.to_numpy(dtype=float)
                image = axis.imshow(
                    values,
                    aspect="auto",
                    cmap=metric["cmap"],
                    vmin=vmin,
                    vmax=vmax,
                )
                axis.set_xticks(np.arange(len(pivot.columns)))
                axis.set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=45)
                axis.set_yticks(np.arange(len(pivot.index)))
                axis.set_yticklabels([f"{value:g}" for value in pivot.index])
                if row_index == 0:
                    axis.set_title(l1_label(prior_case, beta))
                if col_index == 0:
                    axis.set_ylabel(f"source={source_case}\nthreshold")
                if row_index == len(source_cases) - 1:
                    axis.set_xlabel("sieve sigma [cm]")

                if np.isfinite(values).any():
                    best_flat_index = (
                        np.nanargmax(values)
                        if metric["higher_is_better"]
                        else np.nanargmin(values)
                    )
                    best_y, best_x = np.unravel_index(best_flat_index, values.shape)
                    axis.scatter(
                        [best_x],
                        [best_y],
                        marker="*",
                        s=90,
                        color="white",
                        edgecolor="black",
                        linewidth=0.6,
                    )

        if image is not None:
            fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, label=metric["label"])
        fig.suptitle(
            f"{metric['label']} by threshold/sieve/L1 for {method} "
            f"({metric_filter_label(metric)})"
        )
        fig.savefig(
            output_dir
            / f"threshold_sieve_l1_heatmaps_by_source_{filename_token(method)}_{metric['key']}.png",
            bbox_inches="tight",
        )
        plt.close(fig)


def write_summary(df, exact_count_df, metric_tables, metric_row_counts, output_dir):
    lines = [
        "# Sieve L1 Sweep Analysis",
        "",
        f"Input CSV: `{INPUT_CSV}`",
        f"Rows analyzed: {len(df)}",
        f"Rows with correct reconstructed source count: {len(exact_count_df)}",
        f"Valid-count rate: {len(exact_count_df) / max(len(df), 1):.3f}",
        "",
        "Pearson rankings/plots use all rows.",
        "Centroid rankings/plots use only rows where `reconstructed_component_count == true_component_count`.",
        "",
    ]
    for metric in METRICS:
        table = metric_tables[metric["key"]]
        if table.empty:
            continue
        lines.extend(
            [
                f"## Top {metric['label']}",
                "",
                f"Rows used: {metric_row_counts[metric['key']]} ({metric_filter_label(metric)})",
                "",
            ]
        )
        for index, row in enumerate(table.head(12).itertuples(), start=1):
            lines.append(
                f"{index}. {metric['label']} mean={row.metric_mean:.4f}, "
                f"best={row.metric_best:.4f}, method={row.reconstruction_method}, "
                f"source={row.source_case}, {row.configuration_label}, "
                f"{row.best_budget}, n={int(row.cases)}"
            )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    df = prepare_dataframe(INPUT_CSV)
    exact_count_df = df[df["blob_count_exact"]].copy()
    if exact_count_df.empty:
        raise ValueError("No rows reconstructed the correct number of sources.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_tables = {}
    metric_row_counts = {}
    for metric in METRICS:
        metric_df = exact_count_df if metric.get("require_exact_count", True) else df
        metric_row_counts[metric["key"]] = len(metric_df)
        table = aggregate_configs(metric_df, metric)
        metric_tables[metric["key"]] = table
        save_table(
            table,
            OUTPUT_DIR
            / f"threshold_sieve_l1_config_ranking_{metric['key']}_{metric_filter_token(metric)}.csv",
        )
        plot_best_configs_by_algorithm_source(table, OUTPUT_DIR, metric)
        plot_heatmaps_by_algorithm_source(metric_df, OUTPUT_DIR, metric)

    write_summary(df, exact_count_df, metric_tables, metric_row_counts, OUTPUT_DIR)

    print(f"Rows analyzed: {len(df)}")
    print(f"Rows with correct source count: {len(exact_count_df)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
