#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABEL = {"joint": "Joint", "idm": "IDM", "direct": "Direct"}
SUITE_LABEL = {
    "libero_10": "LIBERO-10",
    "libero_goal": "Goal",
    "libero_spatial": "Spatial",
    "libero_object": "Object",
}
COLORS = {"joint": "#7B4CE2", "idm": "#F28E2B", "direct": "#59A14F", "clean": "#4E79A7", "random": "#8C8C8C"}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D0D0D0",
            "grid.alpha": 0.45,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_models(configs: pd.DataFrame) -> list[str]:
    models: list[str] = []
    for raw in configs["models"].dropna().astype(str):
        for model in raw.split():
            if model not in models:
                models.append(model)
    return models or ["joint", "idm"]


def load_task_results(study_root: Path, configs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    models = infer_models(configs)
    for cfg in configs.itertuples(index=False):
        tag = str(cfg.tag)
        attack = str(cfg.attack)
        for model in models:
            run_dir = study_root / "runs" / f"{model}_{attack}_{tag}"
            if not run_dir.exists():
                continue
            result_files = sorted(run_dir.glob("*/*_results.json"))
            for result_path in result_files:
                payload = read_json(result_path)
                suite = str(payload.get("task_suite", result_path.parent.name))
                task_id = int(payload.get("task_id", -1))
                total = int(payload.get("total_episodes", 0))
                successes = int(payload.get("successes", 0))
                success_set = {int(x) for x in payload.get("success_episodes", [])}
                failure_set = {int(x) for x in payload.get("failure_episodes", [])}
                base = {
                    "tag": tag,
                    "family": str(cfg.family),
                    "value": str(cfg.value),
                    "attack": attack,
                    "model": model,
                    "epsilon": float(cfg.epsilon),
                    "budget": int(float(cfg.budget)),
                    "future_weight": float(cfg.future_weight),
                    "suite": suite,
                    "task_id": task_id,
                    "successes": successes,
                    "total_episodes": total,
                    "success_rate": successes / total if total else np.nan,
                    "duration_s": float(payload.get("duration", np.nan)),
                    "attack_action_distance_mean": payload.get("attack_action_distance_mean", np.nan),
                    "attack_future_distance_mean": payload.get("attack_future_distance_mean", np.nan),
                    "attack_score_mean": payload.get("attack_score_mean", np.nan),
                    "attack_perturb_linf_mean": payload.get("attack_perturb_linf_mean", np.nan),
                    "attack_perturb_l1_mean": payload.get("attack_perturb_l1_mean", np.nan),
                    "attack_query_count_mean": payload.get("attack_query_count_mean", np.nan),
                    "attack_attack_time_s_mean": payload.get("attack_attack_time_s_mean", np.nan),
                    "run_dir": str(run_dir),
                    "result_path": str(result_path),
                }
                rows.append(base)
                stats = payload.get("episode_inference_stats", []) or []
                for trial_id in range(total):
                    if trial_id in success_set:
                        success = True
                    elif trial_id in failure_set:
                        success = False
                    else:
                        success = trial_id < successes
                    st = stats[trial_id] if trial_id < len(stats) else {}
                    episode_rows.append(
                        {
                            **{k: base[k] for k in ["tag", "family", "value", "attack", "model", "epsilon", "budget", "future_weight", "suite", "task_id"]},
                            "trial_id": trial_id,
                            "success": success,
                            "episode_wall_time_s": st.get("episode_wall_time_s", np.nan) if isinstance(st, dict) else np.nan,
                            "num_replans": st.get("num_replans", np.nan) if isinstance(st, dict) else np.nan,
                            "executed_action_steps": st.get("executed_action_steps", np.nan) if isinstance(st, dict) else np.nan,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(episode_rows)


def aggregate(task_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if task_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    overall = (
        task_df.groupby(["tag", "family", "value", "attack", "model", "epsilon", "budget", "future_weight"], as_index=False)
        .agg(
            num_tasks=("task_id", "size"),
            successes=("successes", "sum"),
            total_episodes=("total_episodes", "sum"),
            duration_s=("duration_s", "sum"),
            action_distance=("attack_action_distance_mean", "mean"),
            future_distance=("attack_future_distance_mean", "mean"),
            score=("attack_score_mean", "mean"),
            perturb_linf=("attack_perturb_linf_mean", "mean"),
            perturb_l1=("attack_perturb_l1_mean", "mean"),
            query_count=("attack_query_count_mean", "mean"),
            attack_time_s=("attack_attack_time_s_mean", "mean"),
        )
        .assign(success_rate=lambda d: d["successes"] / d["total_episodes"])
    )
    by_suite = (
        task_df.groupby(
            ["tag", "family", "value", "attack", "model", "epsilon", "budget", "future_weight", "suite"],
            as_index=False,
        )
        .agg(
            num_tasks=("task_id", "size"),
            successes=("successes", "sum"),
            total_episodes=("total_episodes", "sum"),
            action_distance=("attack_action_distance_mean", "mean"),
            future_distance=("attack_future_distance_mean", "mean"),
        )
        .assign(success_rate=lambda d: d["successes"] / d["total_episodes"])
    )
    return overall, by_suite


def maybe_float_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def plot_tradeoff(
    overall: pd.DataFrame,
    out_dir: Path,
    *,
    family: str,
    x_col: str,
    title: str,
    filename: str,
    xlabel: str,
) -> None:
    df = overall[overall["family"] == family].copy()
    if df.empty:
        return
    df[x_col] = maybe_float_series(df[x_col])
    df = df.sort_values(["model", x_col])

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2))
    metrics = [
        ("success_rate", "Success rate (%)", lambda y: y * 100),
        ("action_distance", "Action distance", lambda y: y),
        ("future_distance", "Future-video distance", lambda y: y),
    ]
    for ax, (metric, ylabel, transform) in zip(axes, metrics):
        for model in sorted(df["model"].unique()):
            g = df[df["model"] == model].sort_values(x_col)
            y = transform(g[metric].astype(float))
            ax.plot(
                g[x_col],
                y,
                marker="o",
                linewidth=2.7,
                color=COLORS.get(model, None),
                label=MODEL_LABEL.get(model, model),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if x_col == "budget":
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(df[x_col].dropna().unique()))
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0].legend()
    fig.suptitle(title, fontsize=16, fontweight="bold")
    savefig(fig, out_dir, filename)


def plot_random_baseline(overall: pd.DataFrame, out_dir: Path) -> None:
    random_tags = list(overall.loc[overall["family"] == "random", "tag"].dropna().unique())
    random_tag = random_tags[0] if random_tags else "random_uniform_eps0p06"
    keep = overall[overall["tag"].isin(["subset_clean", "base_imgpres", random_tag])].copy()
    if keep.empty:
        return
    order = ["subset_clean", random_tag, "base_imgpres"]
    labels = {
        "subset_clean": "Clean",
        random_tag: "Random noise",
        "base_imgpres": "Img-pres.",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, metric, ylabel, transform in [
        (axes[0], "success_rate", "Success rate (%)", lambda y: y * 100),
        (axes[1], "action_distance", "Action distance", lambda y: y),
    ]:
        x_base = np.arange(len(order))
        width = 0.34
        for mi, model in enumerate(sorted(keep["model"].unique())):
            vals = []
            for tag in order:
                row = keep[(keep["model"] == model) & (keep["tag"] == tag)]
                vals.append(transform(float(row[metric].iloc[0])) if len(row) else np.nan)
            axes_idx = x_base + (mi - 0.5) * width
            ax.bar(axes_idx, vals, width, label=MODEL_LABEL.get(model, model), color=COLORS.get(model), alpha=0.88)
        ax.set_xticks(x_base, [labels.get(tag, tag) for tag in order], rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    axes[0].legend()
    fig.suptitle("Random perturbation baseline on the same subset", fontsize=16, fontweight="bold")
    savefig(fig, out_dir, "random_noise_baseline")


def plot_suite_heatmap(by_suite: pd.DataFrame, out_dir: Path) -> None:
    if by_suite.empty:
        return
    tags = ["subset_clean", "base_imgpres", "random_uniform_eps0p06"]
    df = by_suite[by_suite["tag"].isin(tags)].copy()
    if df.empty:
        return
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(len(models), len(tags), figsize=(4.0 * len(tags), 3.2 * len(models)), squeeze=False)
    for ri, model in enumerate(models):
        for ci, tag in enumerate(tags):
            ax = axes[ri, ci]
            g = df[(df.model == model) & (df.tag == tag)]
            vals = []
            labels = []
            for suite in ["libero_10", "libero_goal", "libero_spatial", "libero_object"]:
                row = g[g.suite == suite]
                vals.append(float(row.success_rate.iloc[0]) * 100 if len(row) else np.nan)
                labels.append(SUITE_LABEL.get(suite, suite))
            ax.bar(np.arange(len(vals)), vals, color=COLORS.get(model), alpha=0.86)
            ax.set_ylim(0, 105)
            ax.set_xticks(np.arange(len(vals)), labels, rotation=20, ha="right")
            ax.set_title(f"{MODEL_LABEL.get(model, model)} / {tag}")
            if ci == 0:
                ax.set_ylabel("Success (%)")
    fig.suptitle("Subset success by suite", fontsize=16, fontweight="bold")
    savefig(fig, out_dir, "subset_suite_success")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", required=True)
    args = parser.parse_args()

    study_root = Path(args.study_root).expanduser().resolve()
    configs_path = study_root / "ablation_configs.csv"
    if not configs_path.exists():
        raise FileNotFoundError(f"Missing ablation config manifest: {configs_path}")

    setup_style()
    analysis_dir = study_root / "analysis"
    plot_dir = analysis_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    configs = pd.read_csv(configs_path).drop_duplicates("tag", keep="first")
    task_df, episode_df = load_task_results(study_root, configs)
    overall, by_suite = aggregate(task_df)

    task_df.to_csv(analysis_dir / "ablation_task_metrics.csv", index=False)
    episode_df.to_csv(analysis_dir / "ablation_episode_metrics.csv", index=False)
    overall.to_csv(analysis_dir / "ablation_overall_metrics.csv", index=False)
    by_suite.to_csv(analysis_dir / "ablation_suite_metrics.csv", index=False)

    plot_tradeoff(
        overall,
        plot_dir,
        family="future",
        x_col="future_weight",
        title="Future-video preservation trade-off",
        filename="future_weight_tradeoff",
        xlabel="future_weight",
    )
    plot_tradeoff(
        overall,
        plot_dir,
        family="budget",
        x_col="budget",
        title="Query budget trade-off",
        filename="query_budget_tradeoff",
        xlabel="query budget",
    )
    plot_tradeoff(
        overall,
        plot_dir,
        family="epsilon",
        x_col="epsilon",
        title="Perturbation budget trade-off",
        filename="epsilon_tradeoff",
        xlabel="epsilon",
    )
    plot_random_baseline(overall, plot_dir)
    plot_suite_heatmap(by_suite, plot_dir)

    summary = {
        "study_root": str(study_root),
        "num_configs_manifested": int(configs.shape[0]),
        "num_task_rows_loaded": int(task_df.shape[0]),
        "num_episode_rows_loaded": int(episode_df.shape[0]),
        "analysis_dir": str(analysis_dir),
        "plots_dir": str(plot_dir),
    }
    with (analysis_dir / "ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
