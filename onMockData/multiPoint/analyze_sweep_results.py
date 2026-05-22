"""
Analyze reconstruction sweep CSV results and create focused visualizations.

This script does not run reconstruction. It reads the CSV produced by
run_parameter_sweep.py and writes plots/tables into visualizations/sweep_analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from run_demo import ON_MOCK_DATA_DIR


INPUT_CSV = (
    ON_MOCK_DATA_DIR
    / "multiPoint"
    / "visualizations"
    / "sweep_results"
    / "reconstruction_method_prior_sources_sweep.csv"
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
    "num_measurements",
    "num_iterations",
    "history_interval",
]


def safe_numeric(series, default=np.nan):
    """Convert a series to numeric values while keeping missing values tame."""
    return pd.to_numeric(series, errors="coerce").fillna(default)


def normalize_pearson(series):
    """Map Pearson from [-1, 1] to [0, 1], clipping bad values."""
    return ((safe_numeric(series) + 1.0) / 2.0).clip(0.0, 1.0)


def geometry_score(centroid_distance_cm):
    """Reward small centroid distance; 5 cm is a soft scale for the penalty."""
    distance = safe_numeric(centroid_distance_cm)
    return np.exp(-distance.clip(lower=0.0) / 5.0).fillna(0.0)


def activity_recovery_score(activity_recovery):
    """Reward recovery close to 1.0, with symmetric penalty for under/over recovery."""
    recovery = safe_numeric(activity_recovery)
    recovery = recovery.where(recovery > 0.0, np.nan)
    score = np.exp(-np.abs(np.log(recovery)))
    return score.fillna(0.0).clip(0.0, 1.0)


def component_match_score(df):
    """Reward matching all true components and mildly penalize extras."""
    true_count = safe_numeric(df["true_component_count"]).clip(lower=1.0)
    matched = safe_numeric(df["matched_component_count"]).clip(lower=0.0)
    extras = safe_numeric(df["extra_reconstructed_count"]).clip(lower=0.0)
    score = matched / true_count - 0.1 * extras
    return score.clip(0.0, 1.0)


def prior_family_from_name(prior_case):
    """Return a compact family label from generated prior_case names."""
    text = str(prior_case)
    if text == "no_priors" or not text:
        return "none"
    if "__" in text:
        return "combo"
    return text.split("_", 1)[0]


def prepare_dataframe(path):
    """Load CSV, coerce metrics, and add derived ranking columns."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    for column in NUMERIC_COLUMNS:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    required = [
        "reconstruction_method",
        "prior_case",
        "source_case",
        "shape_cosine",
        "shape_pearson",
        "centroid_distance_mean_cm",
        "activity_recovery_mean",
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
    df["shape_score"] = (
        0.55 * safe_numeric(df["shape_cosine"]).clip(0.0, 1.0)
        + 0.45 * normalize_pearson(df["shape_pearson"])
    )
    df["geometry_score"] = geometry_score(df["centroid_distance_mean_cm"])
    df["activity_score"] = activity_recovery_score(df["activity_recovery_mean"])
    df["component_score"] = component_match_score(df)
    df["quality_score"] = (
        0.35 * df["shape_score"]
        + 0.30 * df["geometry_score"]
        + 0.20 * df["activity_score"]
        + 0.15 * df["component_score"]
    )
    df["activity_abs_error"] = np.abs(safe_numeric(df["activity_recovery_mean"]) - 1.0)
    return df


def save_table(df, path, columns=None):
    """Save a table to CSV with useful rounded numeric values."""
    output = df.copy()
    if columns is not None:
        output = output[[column for column in columns if column in output.columns]]
    for column in output.select_dtypes(include=[np.number]).columns:
        output[column] = output[column].round(6)
    output.to_csv(path, index=False)


def top_columns():
    return [
        "quality_score",
        "source_case",
        "reconstruction_method",
        "prior_case",
        "num_measurements",
        "num_iterations",
        "shape_cosine",
        "shape_pearson",
        "centroid_distance_mean_cm",
        "centroid_distance_max_cm",
        "activity_recovery_mean",
        "matched_component_count",
        "missed_true_count",
        "extra_reconstructed_count",
        "elapsed_seconds",
        "iteration_seconds",
    ]


def best_rows_by_group(df, group_columns, top_n):
    """Return top run rows for each group."""
    return (
        df.sort_values("quality_score", ascending=False)
        .groupby(group_columns, group_keys=False, dropna=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def aggregate_prior_settings(df, group_columns):
    """Aggregate quality by prior setting and keep the best concrete run metadata."""
    rows = []
    for group_values, group in df.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        best_row = group.sort_values("quality_score", ascending=False).iloc[0]
        row = {column: value for column, value in zip(group_columns, group_values)}
        row.update(
            {
                "quality_mean": float(group["quality_score"].mean()),
                "quality_max": float(group["quality_score"].max()),
                "quality_min": float(group["quality_score"].min()),
                "shape_cosine_mean": float(group["shape_cosine"].mean()),
                "centroid_distance_mean_cm": float(group["centroid_distance_mean_cm"].mean()),
                "activity_recovery_mean": float(group["activity_recovery_mean"].mean()),
                "best_num_measurements": int(best_row["num_measurements"]),
                "best_num_iterations": int(best_row["num_iterations"]),
                "best_quality_score": float(best_row["quality_score"]),
                "best_shape_cosine": float(best_row["shape_cosine"]),
                "best_centroid_distance_mean_cm": float(best_row["centroid_distance_mean_cm"]),
                "source_case_count": int(group["source_case"].nunique()),
                "method_count": int(group["reconstruction_method"].nunique()),
                "cases": int(len(group)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["quality_mean", "quality_max"], ascending=False)


def best_prior_per_method_source(df):
    """Keep the best run for each prior in each algorithm and source scenario."""
    return (
        df.sort_values("quality_score", ascending=False)
        .groupby(["prior_case", "reconstruction_method", "source_case"], dropna=False)
        .head(1)
        .reset_index(drop=True)
    )


def aggregate_common_prior_settings(df):
    """Rank priors by their best achievable quality across every method/source pair."""
    best_pairs = best_prior_per_method_source(df)
    common = (
        best_pairs.groupby("prior_case", dropna=False)
        .agg(
            quality_mean=("quality_score", "mean"),
            quality_min=("quality_score", "min"),
            quality_max=("quality_score", "max"),
            quality_std=("quality_score", "std"),
            shape_cosine_mean=("shape_cosine", "mean"),
            centroid_distance_mean_cm=("centroid_distance_mean_cm", "mean"),
            activity_recovery_mean=("activity_recovery_mean", "mean"),
            method_count=("reconstruction_method", "nunique"),
            source_case_count=("source_case", "nunique"),
            method_source_pair_count=("quality_score", "size"),
        )
        .reset_index()
        .sort_values(["quality_mean", "quality_min"], ascending=False)
    )
    return common, best_pairs


def write_summary_tables(df, output_dir, top_n):
    """Write ranking and grouped summary CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_df = df.sort_values("quality_score", ascending=False)

    save_table(sorted_df.head(top_n), output_dir / "top_overall.csv", top_columns())
    save_table(
        best_rows_by_group(df, ["reconstruction_method"], top_n),
        output_dir / "top_runs_by_method.csv",
        top_columns(),
    )
    save_table(
        best_rows_by_group(df, ["source_case", "reconstruction_method"], top_n),
        output_dir / "top_runs_by_source_and_method.csv",
        top_columns(),
    )

    prior_by_source_method = aggregate_prior_settings(
        df,
        ["source_case", "reconstruction_method", "prior_case"],
    )
    save_table(
        prior_by_source_method,
        output_dir / "best_prior_settings_by_source_and_method.csv",
    )

    prior_by_method = aggregate_prior_settings(df, ["reconstruction_method", "prior_case"])
    save_table(prior_by_method, output_dir / "best_prior_settings_by_method_all_sources.csv")

    prior_by_source = aggregate_prior_settings(df, ["source_case", "prior_case"])
    save_table(prior_by_source, output_dir / "best_prior_settings_by_source_all_methods.csv")

    prior_overall, common_prior_pairs = aggregate_common_prior_settings(df)
    save_table(prior_overall, output_dir / "common_prior_settings_all_algorithms_sources.csv")
    save_table(
        common_prior_pairs,
        output_dir / "best_prior_run_per_algorithm_source.csv",
        top_columns(),
    )

    return {
        "sorted": sorted_df,
        "prior_by_source_method": prior_by_source_method,
        "prior_by_method": prior_by_method,
        "prior_by_source": prior_by_source,
        "prior_overall": prior_overall,
        "common_prior_pairs": common_prior_pairs,
    }


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


def plot_top_by_method(df, output_dir, top_n):
    """Bar chart with best runs per algorithm."""
    plt = setup_matplotlib()
    methods = list(df["reconstruction_method"].dropna().unique())
    if not methods:
        return

    fig, axes = plt.subplots(
        len(methods),
        1,
        figsize=(13, max(4, 0.42 * top_n * len(methods))),
        squeeze=False,
    )
    for axis, method in zip(axes[:, 0], methods):
        subset = (
            df[df["reconstruction_method"] == method]
            .sort_values("quality_score", ascending=False)
            .head(top_n)
            .copy()
        )
        labels = [
            f"{row.source_case} | {row.prior_case} | M{int(row.num_measurements)} I{int(row.num_iterations)}"
            for row in subset.itertuples()
        ]
        y = np.arange(len(subset))
        axis.barh(y, subset["quality_score"], color="#3662a6")
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, max(1.0, float(subset["quality_score"].max()) * 1.08))
        axis.set_title(f"Top {len(subset)} settings for {method}")
        axis.set_xlabel("quality_score")
    save_current_figure(plt, output_dir / "top_settings_by_method.png")


def plot_quality_tradeoff(df, output_dir):
    """Quality vs runtime scatter, useful for finding cheap good settings."""
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(df["reconstruction_method"].dropna().unique())
    colors = dict(zip(methods, ["#2f6fbb", "#c44e52", "#55a868", "#8172b2"]))

    for method in methods:
        subset = df[df["reconstruction_method"] == method]
        ax.scatter(
            subset["elapsed_seconds"],
            subset["quality_score"],
            s=24 + 0.35 * safe_numeric(subset["num_measurements"], 0.0),
            alpha=0.72,
            label=method,
            color=colors.get(method),
            edgecolors="none",
        )
    ax.set_xlabel("elapsed_seconds")
    ax.set_ylabel("quality_score")
    ax.set_title("Quality vs runtime")
    ax.legend()
    save_current_figure(plt, output_dir / "quality_vs_runtime.png")


def plot_shape_vs_geometry(df, output_dir):
    """Show tradeoff between map shape similarity and localization error."""
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(df["reconstruction_method"].dropna().unique())
    markers = ["o", "s", "^", "D"]

    for marker, method in zip(markers, methods):
        subset = df[df["reconstruction_method"] == method]
        scatter = ax.scatter(
            subset["centroid_distance_mean_cm"],
            subset["shape_cosine"],
            c=subset["activity_recovery_mean"],
            cmap="viridis",
            marker=marker,
            s=55,
            alpha=0.75,
            label=method,
        )
    ax.set_xlabel("centroid_distance_mean_cm")
    ax.set_ylabel("shape_cosine")
    ax.set_title("Shape similarity vs localization error")
    ax.legend()
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("activity_recovery_mean")
    save_current_figure(plt, output_dir / "shape_vs_geometry.png")


def plot_iteration_measurement_heatmaps(df, output_dir):
    """Heatmaps of mean quality by measurement count and iteration count."""
    plt = setup_matplotlib()
    methods = list(df["reconstruction_method"].dropna().unique())
    if not methods:
        return

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5), squeeze=False)
    for axis, method in zip(axes[0], methods):
        subset = df[df["reconstruction_method"] == method]
        pivot = subset.pivot_table(
            index="num_iterations",
            columns="num_measurements",
            values="quality_score",
            aggfunc="mean",
        ).sort_index(ascending=True)
        image = axis.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        axis.set_title(f"Mean quality: {method}")
        axis.set_xlabel("NUM_MEASUREMENTS")
        axis.set_ylabel("NUM_ITERATIONS")
        axis.set_xticks(np.arange(len(pivot.columns)))
        axis.set_xticklabels([str(int(value)) for value in pivot.columns])
        axis.set_yticks(np.arange(len(pivot.index)))
        axis.set_yticklabels([str(int(value)) for value in pivot.index])
        for row_index in range(pivot.shape[0]):
            for column_index in range(pivot.shape[1]):
                value = pivot.values[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if value < np.nanmax(pivot.values) * 0.72 else "black",
                        fontsize=8,
                    )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    save_current_figure(plt, output_dir / "quality_heatmap_iterations_measurements.png")


def plot_prior_family_boxplots(df, output_dir):
    """Compare prior families per method."""
    plt = setup_matplotlib()
    methods = list(df["reconstruction_method"].dropna().unique())
    families = list(df["prior_family"].dropna().unique())
    if not methods or not families:
        return

    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 5), squeeze=False)
    for axis, method in zip(axes[0], methods):
        data = [
            df[
                (df["reconstruction_method"] == method)
                & (df["prior_family"] == family)
            ]["quality_score"].dropna().values
            for family in families
        ]
        axis.boxplot(data, labels=families, showmeans=True)
        axis.set_title(f"Prior families: {method}")
        axis.set_ylabel("quality_score")
        axis.tick_params(axis="x", rotation=35)
    save_current_figure(plt, output_dir / "prior_family_quality_boxplots.png")


def plot_robust_settings(robust, output_dir, top_n):
    """Show settings that work well across source cases."""
    plt = setup_matplotlib()
    top = robust.head(top_n).copy()
    if top.empty:
        return

    labels = [
        f"{row.reconstruction_method} | {row.prior_case} | M{int(row.num_measurements)} I{int(row.num_iterations)}"
        for row in top.itertuples()
    ]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(13, max(5, 0.45 * len(top))))
    ax.barh(y, top["quality_mean"], xerr=top["quality_std"].fillna(0.0), color="#4c8f6b")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean quality_score across source cases")
    ax.set_title("Most robust settings across source scenarios")
    save_current_figure(plt, output_dir / "robust_settings_across_sources.png")


def plot_activity_recovery(df, output_dir):
    """Activity recovery vs geometry error highlights amplitude/localization tradeoffs."""
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(df["reconstruction_method"].dropna().unique())
    colors = dict(zip(methods, ["#2f6fbb", "#c44e52", "#55a868", "#8172b2"]))

    for method in methods:
        subset = df[df["reconstruction_method"] == method]
        ax.scatter(
            subset["centroid_distance_mean_cm"],
            subset["activity_recovery_mean"],
            c=colors.get(method),
            s=45,
            alpha=0.72,
            label=method,
            edgecolors="none",
        )
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("centroid_distance_mean_cm")
    ax.set_ylabel("activity_recovery_mean")
    ax.set_title("Activity recovery vs localization error")
    ax.legend()
    save_current_figure(plt, output_dir / "activity_recovery_vs_geometry.png")


def compact_label(text, max_length=58):
    """Shorten long prior names while keeping labels readable."""
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def plot_prior_ranking(table, group_columns, output_path, title, top_n):
    """Plot best prior settings for each requested grouping."""
    plt = setup_matplotlib()
    if table.empty:
        return

    if not group_columns:
        groups = [((), table)]
    else:
        groups = list(table.groupby(group_columns, dropna=False))

    column_count = 2 if len(groups) > 1 else 1
    row_count = int(np.ceil(len(groups) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(15, max(4.5, 3.7 * row_count)),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for axis, (group_key, group) in zip(flat_axes, groups):
        top = group.sort_values(["quality_mean", "quality_max"], ascending=False).head(top_n)
        labels = []
        for row in top.itertuples():
            if hasattr(row, "best_num_measurements") and hasattr(row, "best_num_iterations"):
                label = (
                    f"{row.prior_case} | best M{int(row.best_num_measurements)} "
                    f"I{int(row.best_num_iterations)}"
                )
            else:
                label = str(row.prior_case)
            labels.append(compact_label(label))
        y = np.arange(len(top))
        axis.barh(y, top["quality_mean"], color="#386cb0")
        axis.scatter(top["quality_max"], y, color="#f28e2b", s=24, label="best run")
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, max(1.0, float(top["quality_max"].max()) * 1.08))
        if group_columns:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            subtitle = ", ".join(
                f"{column}={value}" for column, value in zip(group_columns, group_key)
            )
        else:
            subtitle = "all sources, all methods"
        axis.set_title(subtitle)
        axis.set_xlabel("mean quality_score")
        axis.legend(loc="lower right", fontsize=8)

    for axis in flat_axes[len(groups):]:
        axis.axis("off")

    fig.suptitle(title)
    save_current_figure(plt, output_path)


def config_label(row):
    """Human-readable label for one source/method/prior configuration."""
    return compact_label(f"{row.source_case} | {row.reconstruction_method} | {row.prior_case}", 80)


def select_top_configurations(df, top_n):
    """Select top source/method/prior configurations across measurements/iterations."""
    configs = aggregate_prior_settings(
        df,
        ["source_case", "reconstruction_method", "prior_case"],
    )
    return configs.head(top_n)


def plot_accuracy_vs_axis(df, configs, axis_column, output_path, title):
    """Plot quality vs one sweep axis while holding the other sweep axis fixed."""
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
        raise ValueError(f"Unsupported accuracy sweep axis: {axis_column}")

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
                quality_mean=("quality_score", "mean"),
                quality_max=("quality_score", "max"),
            )
            .reset_index()
            .sort_values(axis_column)
        )
        if curve.empty:
            continue
        ax.plot(
            curve[axis_column],
            curve["quality_mean"],
            marker="o",
            linewidth=1.8,
            label=compact_label(f"{config_label(config)} | fixed {fixed_axis}={fixed_value}", 95),
        )
        ax.scatter(
            curve[axis_column],
            curve["quality_max"],
            s=24,
            alpha=0.55,
        )

    ax.set_xlabel(axis_column)
    ax.set_ylabel("quality_score")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    save_current_figure(plt, output_path)


def plot_common_prior_consistency(common_pairs, common_priors, output_dir, top_n):
    """Heatmap of top common priors across algorithm/source pairs."""
    plt = setup_matplotlib()
    top_priors = list(common_priors.head(top_n)["prior_case"])
    if not top_priors:
        return

    subset = common_pairs[common_pairs["prior_case"].isin(top_priors)].copy()
    subset["algorithm_source"] = (
        subset["reconstruction_method"].astype(str)
        + " | "
        + subset["source_case"].astype(str)
    )
    pivot = subset.pivot_table(
        index="prior_case",
        columns="algorithm_source",
        values="quality_score",
        aggfunc="mean",
    ).reindex(top_priors)
    save_table(
        pivot.reset_index(),
        output_dir / "common_prior_quality_by_algorithm_source.csv",
    )

    fig, ax = plt.subplots(
        figsize=(max(9, 1.25 * len(pivot.columns)), max(4, 0.55 * len(pivot.index))),
    )
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([compact_label(value, 72) for value in pivot.index], fontsize=8)
    ax.set_title("Top common priors: best quality in each algorithm/source pair")
    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            value = pivot.values[row_index, column_index]
            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < 0.62 else "black",
                )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("quality_score")
    save_current_figure(plt, output_dir / "common_prior_consistency_heatmap.png")


def plot_measurement_iteration_heatmaps(df, prior_case, output_dir):
    """Heatmaps for one common prior across algorithm/source pairs."""
    plt = setup_matplotlib()
    subset = df[df["prior_case"] == prior_case].copy()
    groups = list(subset.groupby(["reconstruction_method", "source_case"], dropna=False))
    if not groups:
        return

    column_count = min(3, len(groups))
    row_count = int(np.ceil(len(groups) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, 4.3 * row_count),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    last_image = None
    for axis, ((method, source_case), group) in zip(flat_axes, groups):
        pivot = group.pivot_table(
            index="num_iterations",
            columns="num_measurements",
            values="quality_score",
            aggfunc="mean",
        ).sort_index()
        last_image = axis.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(f"{method} | source={source_case}", fontsize=9)
        axis.set_xlabel("NUM_MEASUREMENTS")
        axis.set_ylabel("NUM_ITERATIONS")
        axis.set_xticks(np.arange(len(pivot.columns)))
        axis.set_xticklabels([str(int(value)) for value in pivot.columns])
        axis.set_yticks(np.arange(len(pivot.index)))
        axis.set_yticklabels([str(int(value)) for value in pivot.index])
        for row_index in range(pivot.shape[0]):
            for column_index in range(pivot.shape[1]):
                value = pivot.values[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value < 0.62 else "black",
                    )
    for axis in flat_axes[len(groups):]:
        axis.axis("off")
    fig.suptitle(f"Accuracy vs iterations and measurements for common prior: {prior_case}")
    if last_image is not None:
        fig.colorbar(last_image, ax=flat_axes.tolist(), fraction=0.025, pad=0.02)
    save_current_figure(
        plt,
        output_dir / "accuracy_heatmaps_iterations_measurements_common_prior.png",
    )


def write_report(df, tables, output_dir, top_n):
    """Write a compact markdown report pointing to the generated artifacts."""
    report_path = output_dir / "summary.md"
    best = tables["sorted"].head(min(top_n, 8))
    lines = [
        "# Sweep Analysis Summary",
        "",
        f"Rows analyzed: {len(df)}",
        f"Methods: {', '.join(map(str, sorted(df['reconstruction_method'].dropna().unique())))}",
        f"Source cases: {', '.join(map(str, sorted(df['source_case'].dropna().unique())))}",
        "",
        "## Top Runs",
        "",
    ]
    for index, row in enumerate(best.itertuples(), start=1):
        lines.append(
            f"{index}. score={row.quality_score:.4f}, method={row.reconstruction_method}, "
            f"source={row.source_case}, priors={row.prior_case}, "
            f"measurements={int(row.num_measurements)}, iterations={int(row.num_iterations)}, "
            f"cosine={row.shape_cosine:.4f}, centroid_mean={row.centroid_distance_mean_cm:.3f} cm"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- top_overall.csv",
            "- top_runs_by_method.csv",
            "- top_runs_by_source_and_method.csv",
            "- best_prior_settings_by_source_and_method.csv",
            "- best_prior_settings_by_method_all_sources.csv",
            "- best_prior_settings_by_source_all_methods.csv",
            "- common_prior_settings_all_algorithms_sources.csv",
            "- best_prior_run_per_algorithm_source.csv",
            "- common_prior_quality_by_algorithm_source.csv",
            "- best_priors_by_source_and_method.png",
            "- best_priors_by_method_all_sources.png",
            "- best_priors_by_source_all_methods.png",
            "- best_priors_all_sources_all_methods.png",
            "- common_prior_consistency_heatmap.png",
            "- accuracy_vs_measurements_top_configs.png",
            "- accuracy_vs_iterations_top_configs.png",
            "- accuracy_heatmaps_iterations_measurements_common_prior.png",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    df = prepare_dataframe(INPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tables = write_summary_tables(df, OUTPUT_DIR, TOP_N)
    plot_prior_ranking(
        tables["prior_by_source_method"],
        ["source_case", "reconstruction_method"],
        OUTPUT_DIR / "best_priors_by_source_and_method.png",
        "Best prior settings by true source and algorithm",
        TOP_N,
    )
    plot_prior_ranking(
        tables["prior_by_method"],
        ["reconstruction_method"],
        OUTPUT_DIR / "best_priors_by_method_all_sources.png",
        "Best prior settings by algorithm, all true sources combined",
        TOP_N,
    )
    plot_prior_ranking(
        tables["prior_by_source"],
        ["source_case"],
        OUTPUT_DIR / "best_priors_by_source_all_methods.png",
        "Best prior settings by true source, algorithms combined",
        TOP_N,
    )
    plot_prior_ranking(
        tables["prior_overall"],
        [],
        OUTPUT_DIR / "best_priors_all_sources_all_methods.png",
        "Best prior settings, all true sources and algorithms combined",
        TOP_N,
    )
    plot_common_prior_consistency(
        tables["common_prior_pairs"],
        tables["prior_overall"],
        OUTPUT_DIR,
        TOP_N,
    )
    top_configs = select_top_configurations(df, TOP_N)
    plot_accuracy_vs_axis(
        df,
        top_configs,
        "num_measurements",
        OUTPUT_DIR / "accuracy_vs_measurements_top_configs.png",
        "Accuracy vs number of measurements for top source/method/prior configs",
    )
    plot_accuracy_vs_axis(
        df,
        top_configs,
        "num_iterations",
        OUTPUT_DIR / "accuracy_vs_iterations_top_configs.png",
        "Accuracy vs iterations for top source/method/prior configs",
    )
    if not tables["prior_overall"].empty:
        plot_measurement_iteration_heatmaps(
            df,
            tables["prior_overall"].iloc[0]["prior_case"],
            OUTPUT_DIR,
        )
    write_report(df, tables, OUTPUT_DIR, TOP_N)

    print(f"Rows analyzed: {len(df)}")
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
