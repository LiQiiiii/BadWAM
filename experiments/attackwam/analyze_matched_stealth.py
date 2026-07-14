#!/usr/bin/env python3
"""Analyze matched-strength stealth trade-offs for BadWAM.

This script is intentionally independent of the evaluator.  The evaluator saves
all raw clean/adversarial inputs, actions, predicted futures, and per-replan
statistics.  This analyzer then computes paper-facing aggregate metrics,
including standard SSIM from saved raw images and optional LPIPS when the
dependency is installed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

from experiments.attackwam.summarize_attackwam import summarize_study


COLORS = {
    "Action-only objective": "#E15759",
    "Imagination-preserving": "#59A14F",
}


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
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _run_label(run_name: str) -> str:
    if "matched_action" in run_name or "fw0" in run_name and "fw0p" not in run_name:
        return "Action-only objective"
    if "matched_imgpres" in run_name or "imgpres" in run_name:
        return "Imagination-preserving"
    return run_name


def _iter_raw_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/attack_raw/*.npz"))


def _image_metrics(clean_images: np.ndarray, adv_images: np.ndarray) -> list[dict[str, float]]:
    clean = clean_images.astype(np.float32) / 255.0
    adv = adv_images.astype(np.float32) / 255.0
    if clean.shape != adv.shape:
        return []
    rows: list[dict[str, float]] = []
    diff = adv - clean
    n = int(clean.shape[0])
    for i in range(n):
        d = diff[i]
        flat = d.reshape(-1)
        clean_hwc = np.transpose(clean[i, 0], (1, 2, 0)) if clean.ndim == 5 else np.transpose(clean[i], (1, 2, 0))
        adv_hwc = np.transpose(adv[i, 0], (1, 2, 0)) if adv.ndim == 5 else np.transpose(adv[i], (1, 2, 0))
        try:
            ssim = structural_similarity(clean_hwc, adv_hwc, channel_axis=-1, data_range=1.0)
        except TypeError:
            ssim = structural_similarity(clean_hwc, adv_hwc, multichannel=True, data_range=1.0)
        mse = float(np.mean(flat**2))
        rows.append(
            {
                "input_l2_skimage": float(np.linalg.norm(flat)),
                "input_rmse_skimage": float(math.sqrt(mse)),
                "input_l1_skimage": float(np.mean(np.abs(flat))),
                "input_linf_skimage": float(np.max(np.abs(flat))),
                "input_psnr_skimage": float(-10.0 * math.log10(max(mse, 1e-12))),
                "input_ssim_skimage": float(ssim),
            }
        )
    return rows


def _try_lpips() -> Any | None:
    try:
        import lpips  # type: ignore
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss_fn = lpips.LPIPS(net="alex").to(device).eval()
        return loss_fn, torch, device
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[matched-stealth] LPIPS unavailable; skipping LPIPS ({type(exc).__name__}: {exc})")
        return None


def _lpips_batch(lpips_ctx: Any | None, clean_images: np.ndarray, adv_images: np.ndarray) -> list[float]:
    if lpips_ctx is None:
        return [float("nan")] * int(clean_images.shape[0])
    loss_fn, torch, device = lpips_ctx
    clean = clean_images.astype(np.float32) / 127.5 - 1.0
    adv = adv_images.astype(np.float32) / 127.5 - 1.0
    if clean.ndim == 5:
        clean = clean[:, 0]
        adv = adv[:, 0]
    values: list[float] = []
    with torch.no_grad():
        for start in range(0, clean.shape[0], 16):
            c = torch.from_numpy(clean[start : start + 16]).to(device)
            a = torch.from_numpy(adv[start : start + 16]).to(device)
            out = loss_fn(c, a).detach().float().view(-1).cpu().numpy()
            values.extend(float(x) for x in out)
    return values


def compute_raw_image_metrics(study_root: Path, max_replans_per_run: int = 0) -> pd.DataFrame:
    lpips_ctx = _try_lpips()
    rows: list[dict[str, Any]] = []
    for run_dir in sorted((study_root / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        label = _run_label(run_dir.name)
        seen = 0
        for raw_path in _iter_raw_files(run_dir):
            with np.load(raw_path, allow_pickle=True) as payload:
                if "clean_images" not in payload.files or "adv_images" not in payload.files:
                    continue
                clean = payload["clean_images"]
                adv = payload["adv_images"]
                metrics = _image_metrics(clean, adv)
                lpips_values = _lpips_batch(lpips_ctx, clean, adv)
                task_id = int(payload["task_id"]) if "task_id" in payload.files else -1
                trial_id = int(payload["trial_id"]) if "trial_id" in payload.files else -1
                suite = raw_path.parent.parent.name
                for replan_idx, metric in enumerate(metrics):
                    if max_replans_per_run and seen >= max_replans_per_run:
                        break
                    metric.update(
                        {
                            "run": run_dir.name,
                            "setting": label,
                            "suite": suite,
                            "task_id": task_id,
                            "trial_id": trial_id,
                            "replan_idx": replan_idx,
                            "raw_path": str(raw_path),
                            "input_lpips_alex": lpips_values[replan_idx] if replan_idx < len(lpips_values) else float("nan"),
                        }
                    )
                    rows.append(metric)
                    seen += 1
            if max_replans_per_run and seen >= max_replans_per_run:
                break
    return pd.DataFrame(rows)


def aggregate_image_metrics(image_df: pd.DataFrame) -> pd.DataFrame:
    if image_df.empty:
        return pd.DataFrame()
    metric_cols = [
        "input_l2_skimage",
        "input_rmse_skimage",
        "input_l1_skimage",
        "input_linf_skimage",
        "input_psnr_skimage",
        "input_ssim_skimage",
        "input_lpips_alex",
    ]
    rows = []
    for setting, group in image_df.groupby("setting"):
        row: dict[str, Any] = {"setting": setting, "num_replans_with_images": int(len(group))}
        for col in metric_cols:
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_p95"] = float(vals.quantile(0.95)) if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_matched_table(study_root: Path, image_summary: pd.DataFrame) -> pd.DataFrame:
    analysis_dir = study_root / "analysis"
    run_csv = analysis_dir / "attackwam_run_metrics.csv"
    if not run_csv.exists():
        summarize_study(study_root)
    run_df = pd.read_csv(run_csv)
    run_df = run_df[run_df["run"].str.contains("matched_", na=False)].copy()
    if run_df.empty:
        run_df = pd.read_csv(run_csv).copy()
    run_df["setting"] = run_df["run"].map(_run_label)
    keep = [
        "run",
        "setting",
        "success_rate",
        "attack_action_distance_mean",
        "attack_future_distance_mean",
        "attack_desynchronization_score_mean",
        "attack_input_delta_l2_mean",
        "attack_input_delta_rmse_mean",
        "attack_input_delta_l1_mean",
        "attack_input_delta_linf_mean",
        "attack_input_delta_psnr_mean",
        "attack_input_delta_ssim_global_mean",
        "attack_perturb_l1_mean",
        "attack_perturb_linf_mean",
        "attack_query_count_mean",
        "attack_attack_time_s_mean",
    ]
    keep = [c for c in keep if c in run_df.columns]
    table = run_df[keep].copy()
    if not image_summary.empty:
        table = table.merge(image_summary, on="setting", how="left")
    return table


def plot_tradeoff(table: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    table = table.copy()
    table["success_percent"] = table["success_rate"] * 100.0
    order = ["Action-only objective", "Imagination-preserving"]
    table["setting"] = pd.Categorical(table["setting"], categories=order, ordered=True)
    table = table.sort_values("setting")
    metrics = [
        ("success_percent", "Task success (%)", "Lower is stronger"),
        ("attack_future_distance_mean", "$D_{img}$", "Lower is stealthier"),
        ("attack_action_distance_mean", "$D_{act}$", "Higher is stronger"),
        ("input_ssim_skimage_mean", "Input SSIM", "Higher is less visible"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.45))
    for ax, (col, ylabel, subtitle) in zip(axes, metrics):
        vals = table[col].astype(float).to_numpy() if col in table else np.full(len(table), np.nan)
        labels = table["setting"].astype(str).tolist()
        colors = [COLORS.get(label, "#888888") for label in labels]
        ax.bar(np.arange(len(vals)), vals, color=colors, alpha=0.86, width=0.62)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(["Action-only", "Img-pres."], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(subtitle)
    savefig(fig, out_dir, "matched_strength_stealth_tradeoff")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--max-replans-per-run", type=int, default=0)
    args = parser.parse_args()

    analysis_dir = summarize_study(args.study_root)
    image_df = compute_raw_image_metrics(args.study_root, max_replans_per_run=args.max_replans_per_run)
    image_csv = analysis_dir / "matched_strength_raw_image_metrics.csv"
    image_df.to_csv(image_csv, index=False)
    image_summary = aggregate_image_metrics(image_df)
    image_summary_csv = analysis_dir / "matched_strength_image_summary.csv"
    image_summary.to_csv(image_summary_csv, index=False)
    table = build_matched_table(args.study_root, image_summary)
    table_csv = analysis_dir / "matched_strength_stealth_summary.csv"
    table.to_csv(table_csv, index=False)
    if not table.empty:
        plot_tradeoff(table, analysis_dir)
        print(table.to_string(index=False))
    print(f"Saved raw image metrics: {image_csv}")
    print(f"Saved matched-strength summary: {table_csv}")


if __name__ == "__main__":
    main()
