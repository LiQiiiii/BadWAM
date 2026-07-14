from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUITES = ("libero_10", "libero_goal", "libero_spatial", "libero_object", "libero_90")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_run_name(name: str) -> tuple[str, str]:
    parts = name.split("_", 1)
    if len(parts) == 1:
        return name, "unknown"
    return parts[0], parts[1]


def _iter_task_results(run_dir: Path):
    for suite in SUITES:
        suite_dir = run_dir / suite
        if not suite_dir.exists():
            continue
        for path in sorted(suite_dir.glob("gpu*_task*_results.json")):
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            yield suite, path, payload


def _json_list(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value)
    except TypeError:
        return None


ATTACK_SCALAR_KEYS = (
    "score",
    "action_distance",
    "future_distance",
    "desynchronization_score",
    "perturb_linf",
    "perturb_l1",
    "perturb_left_l1",
    "perturb_right_l1",
    "perturb_left_linf",
    "perturb_right_linf",
    "action_delta_xyz_abs_mean",
    "action_delta_rot_abs_mean",
    "action_delta_gripper_abs_mean",
    "action_delta_horizon_early_l2",
    "action_delta_horizon_middle_l2",
    "action_delta_horizon_late_l2",
    "future_delta_l1_mean",
    "future_delta_linf",
    "input_delta_l2",
    "input_delta_rmse",
    "input_delta_l1",
    "input_delta_linf",
    "input_delta_psnr",
    "input_delta_ssim_global",
    "trajectory_length",
    "query_count",
    "attack_time_s",
)


def summarize_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model, attack = _parse_run_name(run_dir.name)
    rows: list[dict[str, Any]] = []
    replan_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for suite, path, payload in _iter_task_results(run_dir):
        total = int(payload.get("total_episodes", 0))
        successes = int(payload.get("successes", 0))
        row: dict[str, Any] = {
            "run": run_dir.name,
            "model": model,
            "attack": attack,
            "suite": suite,
            "task_id": int(payload.get("task_id", -1)),
            "task_description": payload.get("task_description", ""),
            "result_path": str(path),
            "total_episodes": total,
            "successes": successes,
            "success_rate": successes / total if total else np.nan,
            "duration_s": _safe_float(payload.get("duration")),
            "model_inference_time_mean_s": _safe_float(payload.get("model_inference_time_mean_s")),
            "model_inference_time_p95_s": _safe_float(payload.get("model_inference_time_p95_s")),
            "attack_enabled": bool(payload.get("attack_enabled", False)),
            "attack_mode": payload.get("attack_mode"),
            "attack_search": payload.get("attack_search"),
            "attack_space": payload.get("attack_space"),
            "attack_future_source": payload.get("attack_future_source"),
            "attack_future_video_skip_first": payload.get("attack_future_video_skip_first"),
            "attack_future_video_height": payload.get("attack_future_video_height"),
            "attack_future_video_width": payload.get("attack_future_video_width"),
            "attack_future_video_max_frames": payload.get("attack_future_video_max_frames"),
            "attack_replan_count": payload.get("attack_replan_count", 0),
        }
        for key in ATTACK_SCALAR_KEYS:
            for suffix in ("mean", "p95", "max"):
                row[f"attack_{key}_{suffix}"] = _safe_float(payload.get(f"attack_{key}_{suffix}"))
        row["attack_used_future_term_rate"] = _safe_float(payload.get("attack_used_future_term_rate"))
        rows.append(row)

        success_episodes = set(int(x) for x in payload.get("success_episodes", []))
        for episode_idx, episode_stats in enumerate(payload.get("episode_inference_stats", [])):
            episode_success = episode_idx in success_episodes
            base_episode = {
                "run": run_dir.name,
                "model": model,
                "attack": attack,
                "suite": suite,
                "task_id": int(payload.get("task_id", -1)),
                "task_description": payload.get("task_description", ""),
                "episode_idx": int(episode_idx),
                "episode_success": bool(episode_success),
                "executed_action_steps": int(episode_stats.get("executed_action_steps", 0)),
                "num_replans": int(episode_stats.get("num_replans", 0)),
                "episode_wall_time_s": _safe_float(episode_stats.get("episode_wall_time_s")),
                "attack_raw_path": episode_stats.get("attack_raw_path"),
                "attack_representative_path": episode_stats.get("attack_representative_path"),
            }
            for replan_idx, stats in enumerate(episode_stats.get("attack_replan_stats", [])):
                replan_row = {
                    **base_episode,
                    "replan_idx": int(replan_idx),
                    "used_future_term": bool(stats.get("used_future_term", False)),
                    "future_source": stats.get("future_source"),
                    "space": stats.get("space"),
                    "search": stats.get("search"),
                    "action_delta_abs_by_dim_json": _json_list(stats.get("action_delta_abs_by_dim")),
                    "action_delta_l2_by_timestep_json": _json_list(stats.get("action_delta_l2_by_timestep")),
                    "future_delta_l2_by_frame_json": _json_list(stats.get("future_delta_l2_by_frame")),
                }
                for key in ATTACK_SCALAR_KEYS:
                    replan_row[key] = _safe_float(stats.get(key))
                replan_rows.append(replan_row)

                for trajectory_item in stats.get("trajectory", []) or []:
                    traj_row = {
                        **base_episode,
                        "replan_idx": int(replan_idx),
                        "iteration": int(trajectory_item.get("iteration", -1)),
                        "trajectory_query_count": _safe_float(trajectory_item.get("query_count")),
                    }
                    for key, value in trajectory_item.items():
                        if key in {"iteration", "query_count"}:
                            continue
                        traj_row[key] = _safe_float(value)
                    trajectory_rows.append(traj_row)

    if rows:
        df = pd.DataFrame(rows)
        total_episodes = int(df["total_episodes"].sum())
        total_successes = int(df["successes"].sum())
        summary: dict[str, Any] = {
            "run": run_dir.name,
            "model": model,
            "attack": attack,
            "num_tasks": int(len(df)),
            "total_episodes": total_episodes,
            "successes": total_successes,
            "success_rate": total_successes / total_episodes if total_episodes else np.nan,
            "duration_s": float(df["duration_s"].dropna().sum()) if "duration_s" in df else np.nan,
        }
        for col in df.columns:
            if col.startswith("attack_") or col.startswith("model_inference_time_"):
                if pd.api.types.is_numeric_dtype(df[col]):
                    summary[col] = float(df[col].dropna().mean()) if df[col].dropna().size else np.nan
    else:
        summary = {
            "run": run_dir.name,
            "model": model,
            "attack": attack,
            "num_tasks": 0,
            "total_episodes": 0,
            "successes": 0,
            "success_rate": np.nan,
        }
    return summary, rows, replan_rows, trajectory_rows


def summarize_study(study_root: Path) -> Path:
    runs_dir = study_root / "runs"
    if not runs_dir.exists():
        runs_dir = study_root
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    analysis_dir = study_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    replan_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary, rows, run_replan_rows, run_trajectory_rows = summarize_run(run_dir)
        if summary["num_tasks"] == 0:
            continue
        run_summaries.append(summary)
        task_rows.extend(rows)
        replan_rows.extend(run_replan_rows)
        trajectory_rows.extend(run_trajectory_rows)

    run_df = pd.DataFrame(run_summaries)
    task_df = pd.DataFrame(task_rows)
    replan_df = pd.DataFrame(replan_rows)
    trajectory_df = pd.DataFrame(trajectory_rows)
    run_csv = analysis_dir / "attackwam_run_metrics.csv"
    task_csv = analysis_dir / "attackwam_task_metrics.csv"
    replan_csv = analysis_dir / "attackwam_replan_metrics.csv"
    trajectory_csv = analysis_dir / "attackwam_query_trajectory_metrics.csv"
    run_df.to_csv(run_csv, index=False)
    task_df.to_csv(task_csv, index=False)
    replan_df.to_csv(replan_csv, index=False)
    trajectory_df.to_csv(trajectory_csv, index=False)
    with (analysis_dir / "attackwam_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "study_root": str(study_root),
                "num_runs": int(len(run_df)),
                "num_task_rows": int(len(task_df)),
                "num_replan_rows": int(len(replan_df)),
                "num_trajectory_rows": int(len(trajectory_df)),
                "run_metrics_csv": str(run_csv),
                "task_metrics_csv": str(task_csv),
                "replan_metrics_csv": str(replan_csv),
                "trajectory_metrics_csv": str(trajectory_csv),
            },
            f,
            indent=2,
        )
    print(f"Saved run metrics: {run_csv}")
    print(f"Saved task metrics: {task_csv}")
    print(f"Saved replan metrics: {replan_csv}")
    print(f"Saved query trajectory metrics: {trajectory_csv}")
    if not run_df.empty:
        print(run_df[["run", "success_rate", "attack_action_distance_mean", "attack_future_distance_mean", "attack_desynchronization_score_mean"]].to_string(index=False))
    return analysis_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", required=True, type=Path)
    args = parser.parse_args()
    summarize_study(args.study_root)


if __name__ == "__main__":
    main()
