"""
Analyze TV/L1 reconstruction sweep results using shape_pearson only.

The script reads reconstruction_TV_l1_sweep.csv and writes per-algorithm
rankings/plots. It does not run reconstruction and it does not filter by
reconstructed blob count.
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
    / "reconstruction_TV_l1_sweep.csv"
)
OUTPUT_DIR = ON_MOCK_DATA_DIR / "multiPoint" / "visualizations" / "TV_l1_sweep_analysis"
TOP_N = 15
MEASUREMENT_FILTER = 100

NUMERIC_COLUMNS = [
    "num_measurements",
    "num_iterations",
    "history_interval",
    "activity_component_threshold_fraction",
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
]

CONFIG_COLUMNS = [
    "reconstruction_method",
    "prior_case",
    "l1_beta",
    "tv_beta",
    "tv_epsilon",
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


def load_priors(row):
    priors_text = row.get("reconstruction_priors", "")
    if not isinstance(priors_text, str) or not priors_text.strip():
        return []
    try:
        priors = json.loads(priors_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(priors, list):
        return []
    return priors


def prior_value_from_json(row, prior_type, key, default=np.nan):
    for prior in load_priors(row):
        if str(prior.get("type", "")).strip().lower() == prior_type:
            try:
                return float(prior.get(key, default))
            except (TypeError, ValueError):
                return default
    return default


def prior_value_from_name(prior_case, marker, default=np.nan):
    text = str(prior_case)
    if text == "no_priors":
        return 0.0
    if marker not in text:
        return default
    token = text.split(marker, 1)[1].split("_epsilon_", 1)[0].split("__", 1)[0]
    try:
        return float(token.replace("_", "."))
    except ValueError:
        return default


def epsilon_from_name(prior_case, default=np.nan):
    text = str(prior_case)
    if text == "no_priors":
        return 0.0
    marker = "_epsilon_"
    if marker not in text:
        return default
    token = text.split(marker, 1)[1].split("__", 1)[0]
    try:
        return float(token.replace("_", "."))
    except ValueError:
        return default


def l1_beta_from_row(row):
    value = prior_value_from_json(row, "l1", "beta")
    if np.isfinite(value):
        return value
    return prior_value_from_name(row.get("prior_case", ""), "l1_beta_")


def tv_beta_from_row(row):
    value = prior_value_from_json(row, "tv", "beta")
    if np.isfinite(value):
        return value
    return prior_value_from_name(row.get("prior_case", ""), "tv_beta_")


def tv_epsilon_from_row(row):
    value = prior_value_from_json(row, "tv", "epsilon")
    if np.isfinite(value):
        return value
    return epsilon_from_name(row.get("prior_case", ""))


def compact_config_label(row):
    if str(row.prior_case) == "no_priors":
        prior = "no priors"
    else:
        prior = (
            f"l1={float(row.l1_beta):g}, "
            f"tv={float(row.tv_beta):g}, "
            f"eps={float(row.tv_epsilon):g}"
        )
    return (
        f"{prior}, "
        f"thr={float(row.activity_component_threshold_fraction):g}"
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
        "activity_component_threshold_fraction",
        "shape_pearson",
        "num_iterations",
        "num_measurements",
    ]
    missing = [column for column in required if column not in df]
    if missing:
        raise KeyError(f"Missing required CSV columns: {missing}")

    df = df[np.isfinite(df["shape_pearson"])].copy()
    if df.empty:
        raise ValueError("No rows with finite shape_pearson.")

    df["source_case"] = df["source_case"].astype(str)
    df["reconstruction_method"] = df["reconstruction_method"].astype(str)
    df["l1_beta"] = df.apply(l1_beta_from_row, axis=1)
    df["tv_beta"] = df.apply(tv_beta_from_row, axis=1)
    df["tv_epsilon"] = df.apply(tv_epsilon_from_row, axis=1)
    df["configuration_label"] = df.apply(compact_config_label, axis=1)
    return df


def save_table(df, path, columns=None):
    output = df.copy()
    if columns is not None:
        output = output[[column for column in columns if column in output.columns]]
    for column in output.select_dtypes(include=[np.number]).columns:
        output[column] = output[column].round(6)
    output.to_csv(path, index=False)


def best_budget_rows(df):
    """
    Keep the best iteration/measurement budget for every method/source/config.

    No blob-count condition is applied. The only ranking metric is shape_pearson.
    """
    group_columns = [
        "reconstruction_method",
        "source_case",
        "prior_case",
        "activity_component_threshold_fraction",
    ]
    return (
        df.sort_values("shape_pearson", ascending=False, na_position="last")
        .groupby(group_columns, group_keys=False, dropna=False)
        .head(1)
        .reset_index(drop=True)
    )


def rank_configs(best_df):
    group_columns = CONFIG_COLUMNS
    rows = []
    for keys, group in best_df.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "pearson_mean": float(group["shape_pearson"].mean()),
                "pearson_best": float(group["shape_pearson"].max()),
                "pearson_worst": float(group["shape_pearson"].min()),
                "pearson_std": float(group["shape_pearson"].std(ddof=0)),
                "source_case_count": int(group["source_case"].nunique()),
                "rows": int(len(group)),
                "best_source_case": str(
                    group.sort_values("shape_pearson", ascending=False)
                    .iloc[0]["source_case"]
                ),
                "best_num_iterations": int(
                    group.sort_values("shape_pearson", ascending=False)
                    .iloc[0]["num_iterations"]
                ),
                "best_num_measurements": int(
                    group.sort_values("shape_pearson", ascending=False)
                    .iloc[0]["num_measurements"]
                ),
                "configuration_label": str(group.iloc[0]["configuration_label"]),
            }
        )
        rows.append(row)
    ranking = pd.DataFrame(rows)
    return ranking.sort_values(
        ["reconstruction_method", "pearson_mean", "pearson_best"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def top_concrete_runs(df):
    columns = [
        "reconstruction_method",
        "source_case",
        "prior_case",
        "l1_beta",
        "tv_beta",
        "tv_epsilon",
        "activity_component_threshold_fraction",
        "shape_pearson",
        "num_iterations",
        "num_measurements",
        "shape_cosine",
        "true_component_count",
        "reconstructed_component_count",
        "matched_component_count",
        "extra_reconstructed_count",
        "missed_true_count",
        "centroid_distance_mean_cm",
        "activity_recovery_mean",
        "elapsed_seconds",
        "iteration_seconds",
    ]
    return df.sort_values(
        ["reconstruction_method", "shape_pearson"],
        ascending=[True, False],
    )[[column for column in columns if column in df.columns]]


def write_summary(df, best_df, ranking, output_dir, num_measurements, num_iterations):
    lines = [
        "# TV/L1 Pearson Sweep Analysis",
        "",
        f"Input CSV: `{INPUT_CSV}`",
        f"Rows filter: `num_measurements == {int(num_measurements)}` and "
        f"`num_iterations == {int(num_iterations)}`",
        f"Rows analyzed: {len(df)}",
        "Metric: `shape_pearson` only, higher is better.",
        "Blob-count filter: none.",
        "",
    ]
    for method in sorted(df["reconstruction_method"].unique()):
        method_rank = ranking[ranking["reconstruction_method"] == method]
        method_best = best_df[best_df["reconstruction_method"] == method]
        lines.extend(
            [
                f"## {method}",
                "",
                f"Rows: {int((df['reconstruction_method'] == method).sum())}",
                f"Best-budget rows: {len(method_best)}",
                "",
                "Top configurations after selecting best budget per source/config:",
                "",
            ]
        )
        for index, row in enumerate(method_rank.head(TOP_N).itertuples(index=False), start=1):
            lines.append(
                f"{index}. mean={row.pearson_mean:.4f}, "
                f"best={row.pearson_best:.4f}, "
                f"{row.configuration_label}, "
                f"best source={row.best_source_case}, "
                f"I{row.best_num_iterations} M{row.best_num_measurements}"
            )
        lines.append("")

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_top_config_bars(ranking, output_dir, plt):
    for method in sorted(ranking["reconstruction_method"].unique()):
        method_rank = ranking[ranking["reconstruction_method"] == method].head(TOP_N)
        if method_rank.empty:
            continue
        labels = list(method_rank["configuration_label"])[::-1]
        values = list(method_rank["pearson_mean"])[::-1]

        fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(labels))))
        ax.barh(np.arange(len(labels)), values, color="tab:blue", alpha=0.82)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("mean best-budget shape_pearson over source cases")
        ax.set_title(f"{method}: top TV/L1 configurations by Pearson")
        ax.set_xlim(0.0, max(1.0, float(np.nanmax(values)) * 1.08))
        fig.tight_layout()
        fig.savefig(
            output_dir / f"top_tv_l1_configs_{filename_token(method)}_pearson.png"
        )
        plt.close(fig)


def plot_heatmaps(ranking, output_dir, plt):
    heatmap_df = ranking[
        ranking["prior_case"].astype(str).ne("no_priors")
        & np.isfinite(ranking["l1_beta"])
        & np.isfinite(ranking["tv_beta"])
    ].copy()
    if heatmap_df.empty:
        return

    for method in sorted(heatmap_df["reconstruction_method"].unique()):
        method_df = heatmap_df[heatmap_df["reconstruction_method"] == method]
        thresholds = sorted(method_df["activity_component_threshold_fraction"].unique())
        if not thresholds:
            continue

        fig, axes = plt.subplots(
            1,
            len(thresholds),
            figsize=(max(5, 4.2 * len(thresholds)), 4.8),
            squeeze=False,
        )
        axes = axes.ravel()
        color_min = float(method_df["pearson_mean"].min())
        color_max = float(method_df["pearson_mean"].max())
        if color_max <= color_min:
            color_max = color_min + 1.0

        last_image = None
        for axis, threshold in zip(axes, thresholds):
            table = method_df[
                method_df["activity_component_threshold_fraction"] == threshold
            ]
            pivot = table.pivot_table(
                index="l1_beta",
                columns="tv_beta",
                values="pearson_mean",
                aggfunc="mean",
            ).sort_index(ascending=True).sort_index(axis=1, ascending=True)
            image = axis.imshow(
                pivot.values,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                vmin=color_min,
                vmax=color_max,
            )
            last_image = image
            axis.set_title(f"threshold={float(threshold):g}")
            axis.set_xlabel("tv beta")
            axis.set_ylabel("l1 beta")
            axis.set_xticks(np.arange(len(pivot.columns)))
            axis.set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=45)
            axis.set_yticks(np.arange(len(pivot.index)))
            axis.set_yticklabels([f"{value:g}" for value in pivot.index])

            for row_index, l1_beta in enumerate(pivot.index):
                for column_index, tv_beta in enumerate(pivot.columns):
                    value = pivot.loc[l1_beta, tv_beta]
                    if np.isfinite(value):
                        axis.text(
                            column_index,
                            row_index,
                            f"{value:.3f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if value < (color_min + color_max) / 2 else "black",
                        )

        if last_image is not None:
            colorbar = fig.colorbar(last_image, ax=axes.tolist(), shrink=0.82)
            colorbar.set_label("mean best-budget shape_pearson")
        fig.suptitle(f"{method}: TV/L1 Pearson heatmaps")
        fig.subplots_adjust(left=0.08, right=0.90, bottom=0.16, top=0.82, wspace=0.35)
        fig.savefig(output_dir / f"tv_l1_heatmaps_{filename_token(method)}_pearson.png")
        plt.close(fig)


def plot_source_heatmaps(best_df, output_dir, plt):
    heatmap_df = best_df[
        best_df["prior_case"].astype(str).ne("no_priors")
        & np.isfinite(best_df["l1_beta"])
        & np.isfinite(best_df["tv_beta"])
    ].copy()
    if heatmap_df.empty:
        return

    for method in sorted(heatmap_df["reconstruction_method"].unique()):
        method_df = heatmap_df[heatmap_df["reconstruction_method"] == method]
        source_cases = sorted(method_df["source_case"].unique(), key=source_sort_key)
        thresholds = sorted(method_df["activity_component_threshold_fraction"].unique())
        if not source_cases or not thresholds:
            continue

        fig, axes = plt.subplots(
            len(source_cases),
            len(thresholds),
            figsize=(max(6, 3.8 * len(thresholds)), max(5, 3.2 * len(source_cases))),
            squeeze=False,
        )
        color_min = float(method_df["shape_pearson"].min())
        color_max = float(method_df["shape_pearson"].max())
        if color_max <= color_min:
            color_max = color_min + 1.0

        last_image = None
        for row_index, source_case in enumerate(source_cases):
            for column_index, threshold in enumerate(thresholds):
                axis = axes[row_index, column_index]
                table = method_df[
                    (method_df["source_case"] == source_case)
                    & (method_df["activity_component_threshold_fraction"] == threshold)
                ]
                pivot = table.pivot_table(
                    index="l1_beta",
                    columns="tv_beta",
                    values="shape_pearson",
                    aggfunc="mean",
                ).sort_index(ascending=True).sort_index(axis=1, ascending=True)
                image = axis.imshow(
                    pivot.values,
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                    vmin=color_min,
                    vmax=color_max,
                )
                last_image = image
                axis.set_title(f"source={source_case}, thr={float(threshold):g}")
                axis.set_xticks(np.arange(len(pivot.columns)))
                axis.set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=45)
                axis.set_yticks(np.arange(len(pivot.index)))
                axis.set_yticklabels([f"{value:g}" for value in pivot.index])
                if row_index == len(source_cases) - 1:
                    axis.set_xlabel("tv beta")
                if column_index == 0:
                    axis.set_ylabel("l1 beta")

        if last_image is not None:
            colorbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.82)
            colorbar.set_label("best-budget shape_pearson")
        fig.suptitle(f"{method}: Pearson by source, threshold, L1/TV")
        fig.subplots_adjust(left=0.08, right=0.90, bottom=0.10, top=0.90, wspace=0.35, hspace=0.45)
        fig.savefig(
            output_dir / f"tv_l1_by_source_{filename_token(method)}_pearson.png"
        )
        plt.close(fig)


def analyze_iteration(df, output_dir, num_measurements, num_iterations, plt):
    output_dir.mkdir(parents=True, exist_ok=True)
    best_df = best_budget_rows(df)
    ranking = rank_configs(best_df)

    save_table(
        ranking,
        output_dir / "tv_l1_pearson_config_ranking_by_algorithm.csv",
    )
    save_table(
        top_concrete_runs(df),
        output_dir / "tv_l1_pearson_top_concrete_runs_by_algorithm.csv",
    )
    save_table(
        best_df.sort_values(
            ["reconstruction_method", "source_case", "shape_pearson"],
            ascending=[True, True, False],
        ),
        output_dir / "tv_l1_pearson_best_budget_rows.csv",
    )

    for method in sorted(df["reconstruction_method"].unique()):
        method_ranking = ranking[ranking["reconstruction_method"] == method]
        save_table(
            method_ranking,
            output_dir
            / f"tv_l1_pearson_config_ranking_{filename_token(method)}.csv",
        )
        save_table(
            top_concrete_runs(df[df["reconstruction_method"] == method]).head(TOP_N),
            output_dir / f"top_runs_{filename_token(method)}_pearson.csv",
        )

    write_summary(
        df=df,
        best_df=best_df,
        ranking=ranking,
        output_dir=output_dir,
        num_measurements=num_measurements,
        num_iterations=num_iterations,
    )
    plot_top_config_bars(ranking, output_dir, plt)
    plot_heatmaps(ranking, output_dir, plt)
    plot_source_heatmaps(best_df, output_dir, plt)
    return {
        "num_iterations": int(num_iterations),
        "rows": int(len(df)),
        "best_budget_rows": int(len(best_df)),
        "output_dir": str(output_dir),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()

    df = prepare_dataframe(INPUT_CSV)
    df = df[df["num_measurements"].eq(MEASUREMENT_FILTER)].copy()
    if df.empty:
        raise ValueError(f"No rows with num_measurements == {MEASUREMENT_FILTER}.")

    iteration_values = sorted(
        int(value)
        for value in df["num_iterations"].dropna().unique()
    )
    summaries = []
    for num_iterations in iteration_values:
        iteration_df = df[df["num_iterations"].eq(num_iterations)].copy()
        iteration_dir = OUTPUT_DIR / f"measurements_{MEASUREMENT_FILTER}" / f"iter_{num_iterations}"
        summaries.append(
            analyze_iteration(
                df=iteration_df,
                output_dir=iteration_dir,
                num_measurements=MEASUREMENT_FILTER,
                num_iterations=num_iterations,
                plt=plt,
            )
        )

    index = pd.DataFrame(summaries)
    save_table(index, OUTPUT_DIR / f"measurements_{MEASUREMENT_FILTER}_iteration_index.csv")

    print(f"Rows analyzed after measurement filter: {len(df)}")
    print(f"Iterations analyzed: {iteration_values}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
