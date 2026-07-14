#!/usr/bin/env bash
set -euo pipefail

# Sweep LIBERO evaluation over action diffusion/flow inference steps and plot a PDF report.
#
# Common usage:
#   bash experiments/libero/run_num_inference_steps_sweep.sh
#
# Useful overrides:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 NUM_TRIALS=10 \
#     bash experiments/libero/run_num_inference_steps_sweep.sh
#
# Extra Hydra overrides passed to this script are forwarded to run_libero_manager.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-libero_uncond_2cam224_1e-4}"
CKPT="${CKPT:-${REPO_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-${REPO_ROOT}/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
STEPS="${STEPS:-8 6 4 2 1}"
NUM_TRIALS="${NUM_TRIALS:-50}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
PLOT_ONLY="${PLOT_ONLY:-0}"
VISUALIZE_FUTURE_VIDEO="${VISUALIZE_FUTURE_VIDEO:-false}"
MONITORING_INTERVAL="${MONITORING_INTERVAL:-10}"
STATUS_INTERVAL="${STATUS_INTERVAL:-60}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/evaluate_results/libero/${TASK}/num_inference_steps_sweep_$(date +%Y%m%d_%H%M%S)}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MONITORING_INTERVAL
export STATUS_INTERVAL

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS="${NUM_GPUS:-$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | sed '/^$/d' | wc -l)}"
else
  NUM_GPUS="${NUM_GPUS:-8}"
fi

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" && -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV_NAME:-fastwam}"
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
EXTRA_ARGS=("$@")

mkdir -p "${OUTPUT_ROOT}"

if [[ "${PLOT_ONLY}" != "1" ]]; then
  if [[ ! -f "${CKPT}" ]]; then
    echo "Checkpoint not found: ${CKPT}" >&2
    exit 1
  fi
  if [[ ! -f "${DATASET_STATS_PATH}" ]]; then
    echo "Dataset stats not found: ${DATASET_STATS_PATH}" >&2
    exit 1
  fi
fi

cat > "${OUTPUT_ROOT}/sweep_config.env" <<EOF
TASK=${TASK}
CKPT=${CKPT}
DATASET_STATS_PATH=${DATASET_STATS_PATH}
STEPS=${STEPS}
NUM_TRIALS=${NUM_TRIALS}
NUM_GPUS=${NUM_GPUS}
MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
VISUALIZE_FUTURE_VIDEO=${VISUALIZE_FUTURE_VIDEO}
MONITORING_INTERVAL=${MONITORING_INTERVAL}
STATUS_INTERVAL=${STATUS_INTERVAL}
PYTHON_BIN=${PYTHON_BIN}
EXTRA_ARGS=${EXTRA_ARGS[*]:-}
EOF

RUN_INDEX="${OUTPUT_ROOT}/run_dirs.tsv"
if [[ "${PLOT_ONLY}" != "1" || ! -f "${RUN_INDEX}" ]]; then
  printf "step\trun_dir\tstatus\twall_time_seconds\n" > "${RUN_INDEX}"
fi

read -r -a STEP_ARRAY <<< "${STEPS}"

plot_sweep() {
  "${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

output_root = Path(sys.argv[1]).expanduser().resolve()


def _float_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _format_cell(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _step_from_name(path: Path):
    match = re.search(r"steps?_([0-9]+)", path.name)
    if match:
        return int(match.group(1))
    return None


def _read_run_index():
    rows = {}
    index_path = output_root / "run_dirs.tsv"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    step = int(row["step"])
                except Exception:
                    continue
                rows[step] = {
                    "run_dir": row.get("run_dir", ""),
                    "status": row.get("status", ""),
                    "wall_time_seconds": _float_or_none(row.get("wall_time_seconds")),
                }

    for run_dir in sorted(output_root.glob("steps_*")):
        if not run_dir.is_dir():
            continue
        step = _step_from_name(run_dir)
        if step is None:
            continue
        wall_file = run_dir / "wall_time_seconds.txt"
        wall = None
        if wall_file.exists():
            wall = _float_or_none(wall_file.read_text(encoding="utf-8").strip())
        rows.setdefault(
            step,
            {
                "run_dir": str(run_dir),
                "status": "existing",
                "wall_time_seconds": wall,
            },
        )
        if rows[step].get("wall_time_seconds") is None and wall is not None:
            rows[step]["wall_time_seconds"] = wall
    return rows


def _percentile(values, q):
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _load_task_records(run_dir: Path):
    records = []
    for path in sorted(run_dir.glob("**/gpu*_task*_results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] failed to read {path}: {exc}")
            continue
        payload["_result_file"] = str(path)
        records.append(payload)
    return records


def _expected_task_count(run_dir: Path):
    task_file = run_dir / "tasks.txt"
    if not task_file.exists():
        return None
    with task_file.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


run_index = _read_run_index()
summary_rows = []
suite_rows = []
task_rows = []

for step, row in sorted(run_index.items()):
    run_dir = Path(row["run_dir"]).expanduser()
    if not run_dir.is_absolute():
        run_dir = output_root / run_dir
    records = _load_task_records(run_dir)
    expected_tasks = _expected_task_count(run_dir)

    total_episodes = sum(int(r.get("total_episodes", 0) or 0) for r in records)
    total_successes = sum(int(r.get("successes", 0) or 0) for r in records)
    total_task_duration = sum(float(r.get("duration", 0.0) or 0.0) for r in records)
    completed_tasks = len(records)
    success_rate = (100.0 * total_successes / total_episodes) if total_episodes else None
    task_seconds_per_episode = (total_task_duration / total_episodes) if total_episodes else None
    avg_task_duration = (total_task_duration / completed_tasks) if completed_tasks else None
    max_task_duration = max((float(r.get("duration", 0.0) or 0.0) for r in records), default=None)

    inference_times = []
    for r in records:
        for episode_stats in r.get("episode_inference_stats", []) or []:
            inference_times.extend(float(x) for x in episode_stats.get("model_inference_times_s", []) or [])
    if not inference_times:
        for r in records:
            count = int(r.get("model_inference_count", 0) or 0)
            mean_s = _float_or_none(r.get("model_inference_time_mean_s"))
            if count > 0 and mean_s is not None:
                inference_times.extend([mean_s] * count)

    inference_count = len(inference_times)
    inference_total = sum(inference_times)
    inference_mean = (inference_total / inference_count) if inference_count else None
    inference_p50 = _percentile(inference_times, 50)
    inference_p95 = _percentile(inference_times, 95)
    inference_per_episode = (inference_total / total_episodes) if total_episodes else None
    replans_per_episode = (inference_count / total_episodes) if total_episodes else None

    psnr_values = [
        _float_or_none(r.get("future_video_psnr_mean"))
        for r in records
        if _float_or_none(r.get("future_video_psnr_mean")) is not None
    ]
    future_video_psnr_mean = mean(psnr_values) if psnr_values else None

    wall_time = row.get("wall_time_seconds")
    wall_minutes = (wall_time / 60.0) if wall_time is not None else None
    completion_rate = (
        100.0 * completed_tasks / expected_tasks
        if expected_tasks not in (None, 0)
        else None
    )

    summary_rows.append(
        {
            "num_inference_steps": step,
            "run_dir": str(run_dir),
            "status": row.get("status", ""),
            "completed_tasks": completed_tasks,
            "expected_tasks": expected_tasks,
            "completion_rate": completion_rate,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "success_rate": success_rate,
            "wall_time_seconds": wall_time,
            "wall_time_minutes": wall_minutes,
            "total_task_duration_seconds": total_task_duration,
            "avg_task_duration_seconds": avg_task_duration,
            "max_task_duration_seconds": max_task_duration,
            "task_seconds_per_episode": task_seconds_per_episode,
            "model_inference_count": inference_count,
            "model_inference_time_total_s": inference_total if inference_count else None,
            "model_inference_time_mean_s": inference_mean,
            "model_inference_time_p50_s": inference_p50,
            "model_inference_time_p95_s": inference_p95,
            "model_inference_time_per_episode_s": inference_per_episode,
            "replans_per_episode": replans_per_episode,
            "future_video_psnr_mean": future_video_psnr_mean,
        }
    )

    suite_stats = defaultdict(lambda: {"episodes": 0, "successes": 0, "duration": 0.0, "tasks": 0})
    for r in records:
        suite = str(r.get("task_suite") or Path(r["_result_file"]).parent.name)
        episodes = int(r.get("total_episodes", 0) or 0)
        successes = int(r.get("successes", 0) or 0)
        duration = float(r.get("duration", 0.0) or 0.0)
        suite_stats[suite]["episodes"] += episodes
        suite_stats[suite]["successes"] += successes
        suite_stats[suite]["duration"] += duration
        suite_stats[suite]["tasks"] += 1
        task_rows.append(
            {
                "num_inference_steps": step,
                "suite": suite,
                "task_id": int(r.get("task_id", -1)),
                "task_description": r.get("task_description", ""),
                "successes": successes,
                "total_episodes": episodes,
                "success_rate": (100.0 * successes / episodes) if episodes else None,
                "duration_seconds": duration,
                "model_inference_count": int(r.get("model_inference_count", 0) or 0),
                "model_inference_time_mean_s": _float_or_none(r.get("model_inference_time_mean_s")),
                "model_inference_time_p95_s": _float_or_none(r.get("model_inference_time_p95_s")),
                "future_video_psnr_mean": _float_or_none(r.get("future_video_psnr_mean")),
                "result_file": r["_result_file"],
            }
        )

    for suite, stats in sorted(suite_stats.items()):
        suite_rows.append(
            {
                "num_inference_steps": step,
                "suite": suite,
                "completed_tasks": stats["tasks"],
                "total_episodes": stats["episodes"],
                "total_successes": stats["successes"],
                "success_rate": (
                    100.0 * stats["successes"] / stats["episodes"]
                    if stats["episodes"]
                    else None
                ),
                "total_task_duration_seconds": stats["duration"],
                "task_seconds_per_episode": (
                    stats["duration"] / stats["episodes"]
                    if stats["episodes"]
                    else None
                ),
            }
        )

summary_rows = sorted(summary_rows, key=lambda r: r["num_inference_steps"])
suite_rows = sorted(suite_rows, key=lambda r: (r["suite"], r["num_inference_steps"]))
task_rows = sorted(task_rows, key=lambda r: (r["suite"], r["task_id"], r["num_inference_steps"]))

baseline = None
if summary_rows:
    largest_step = max(r["num_inference_steps"] for r in summary_rows)
    baseline = next((r for r in summary_rows if r["num_inference_steps"] == largest_step), None)

if baseline is not None:
    baseline_latency = baseline.get("model_inference_time_mean_s") or baseline.get("task_seconds_per_episode")
    for row in summary_rows:
        current_latency = row.get("model_inference_time_mean_s") or row.get("task_seconds_per_episode")
        if baseline_latency and current_latency:
            row["speedup_vs_largest_step"] = baseline_latency / current_latency
        else:
            row["speedup_vs_largest_step"] = None


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


_write_csv(output_root / "all_steps_metrics.csv", summary_rows)
_write_csv(output_root / "all_steps_suite_metrics.csv", suite_rows)
_write_csv(output_root / "all_steps_task_metrics.csv", task_rows)
(output_root / "all_steps_metrics.json").write_text(
    json.dumps(
        {
            "summary": summary_rows,
            "suite": suite_rows,
            "task": task_rows,
        },
        indent=2,
    ),
    encoding="utf-8",
)

if not summary_rows or all(row["completed_tasks"] == 0 for row in summary_rows):
    print(f"No completed LIBERO result JSON files found under {output_root}")
    sys.exit(0)

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception as exc:
    print(f"Failed to import plotting dependencies: {exc}", file=sys.stderr)
    print("CSV/JSON summaries were still written.", file=sys.stderr)
    sys.exit(1)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass

steps = [r["num_inference_steps"] for r in summary_rows]


def series(key):
    return [r.get(key) for r in summary_rows]


def plot_line(ax, x, y, label=None, marker="o", **kwargs):
    xs = []
    ys = []
    for xi, yi in zip(x, y):
        if yi is None:
            continue
        xs.append(xi)
        ys.append(float(yi))
    if xs:
        ax.plot(xs, ys, marker=marker, linewidth=2.0, label=label, **kwargs)
    return bool(xs)


pdf_path = output_root / "num_inference_steps_sweep.pdf"
colors = {
    "success": "#2f7f62",
    "time": "#3066be",
    "speed": "#d95f02",
    "latency": "#4c78a8",
    "psnr": "#7f7f7f",
}

with PdfPages(pdf_path) as pdf:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    fig.suptitle("LIBERO num_inference_steps Sweep", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    plot_line(ax, steps, series("success_rate"), color=colors["success"])
    ax.set_title("Overall Success Rate")
    ax.set_xlabel("num_inference_steps")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(steps)
    ax.set_ylim(0, 105)

    ax = axes[0, 1]
    suites = sorted({r["suite"] for r in suite_rows})
    for suite in suites:
        suite_map = {r["num_inference_steps"]: r["success_rate"] for r in suite_rows if r["suite"] == suite}
        plot_line(ax, steps, [suite_map.get(s) for s in steps], label=suite)
    ax.set_title("Success Rate by Suite")
    ax.set_xlabel("num_inference_steps")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(steps)
    ax.set_ylim(0, 105)
    if suites:
        ax.legend(fontsize=8, loc="best")

    ax = axes[1, 0]
    if any(v is not None for v in series("model_inference_time_mean_s")):
        plot_line(ax, steps, series("model_inference_time_mean_s"), label="mean", color=colors["latency"])
        plot_line(ax, steps, series("model_inference_time_p95_s"), label="p95", color=colors["time"], marker="s")
        ax.set_title("Model Inference Latency per Replan")
        ax.set_ylabel("Seconds")
        ax.legend(fontsize=8)
    else:
        plot_line(ax, steps, series("task_seconds_per_episode"), color=colors["time"])
        ax.set_title("Aggregate Task Time per Episode")
        ax.set_ylabel("Task seconds / episode")
    ax.set_xlabel("num_inference_steps")
    ax.set_xticks(steps)

    ax = axes[1, 1]
    if any(v is not None for v in series("speedup_vs_largest_step")):
        vals = [0.0 if v is None else float(v) for v in series("speedup_vs_largest_step")]
        ax.bar([str(s) for s in steps], vals, color=colors["speed"], alpha=0.85)
        ax.axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
        ax.set_title(f"Latency Speedup vs {max(steps)} Steps")
        ax.set_xlabel("num_inference_steps")
        ax.set_ylabel("Speedup")
    else:
        plot_line(ax, steps, series("wall_time_minutes"), color=colors["time"])
        ax.set_title("Wall Clock Time")
        ax.set_xlabel("num_inference_steps")
        ax.set_ylabel("Minutes")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(11.5, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.0])
    fig.suptitle("Detailed Sweep Diagnostics", fontsize=16, fontweight="bold")

    ax = fig.add_subplot(gs[0])
    if suites:
        heat = np.full((len(suites), len(steps)), np.nan)
        for i, suite in enumerate(suites):
            for j, step in enumerate(steps):
                item = next((r for r in suite_rows if r["suite"] == suite and r["num_inference_steps"] == step), None)
                if item and item["success_rate"] is not None:
                    heat[i, j] = float(item["success_rate"])
        im = ax.imshow(heat, aspect="auto", cmap="YlGn", vmin=0, vmax=100)
        ax.set_title("Suite Success Rate Heatmap")
        ax.set_xticks(range(len(steps)), labels=[str(s) for s in steps])
        ax.set_yticks(range(len(suites)), labels=suites)
        ax.set_xlabel("num_inference_steps")
        for i in range(len(suites)):
            for j in range(len(steps)):
                if not np.isnan(heat[i, j]):
                    ax.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Success rate (%)")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No suite-level results found", ha="center", va="center")

    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    table_cols = [
        "steps",
        "succ %",
        "tasks",
        "trials",
        "lat mean",
        "lat p95",
        "replans/ep",
        "task s/ep",
        "wall min",
        "speedup",
    ]
    table_rows = []
    for row in summary_rows:
        tasks = f"{row['completed_tasks']}"
        if row.get("expected_tasks"):
            tasks = f"{row['completed_tasks']}/{row['expected_tasks']}"
        table_rows.append(
            [
                row["num_inference_steps"],
                _format_cell(row.get("success_rate"), 2),
                tasks,
                row.get("total_episodes", 0),
                _format_cell(row.get("model_inference_time_mean_s"), 3),
                _format_cell(row.get("model_inference_time_p95_s"), 3),
                _format_cell(row.get("replans_per_episode"), 2),
                _format_cell(row.get("task_seconds_per_episode"), 2),
                _format_cell(row.get("wall_time_minutes"), 1),
                _format_cell(row.get("speedup_vs_largest_step"), 2),
            ]
        )
    table = ax.table(cellText=table_rows, colLabels=table_cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    ax.set_title("Summary Table", pad=12)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    if any(v is not None for v in series("future_video_psnr_mean")) or any(
        v is not None for v in series("model_inference_time_per_episode_s")
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        fig.suptitle("Additional Inference Signals", fontsize=15, fontweight="bold")
        ax = axes[0]
        if any(v is not None for v in series("model_inference_time_per_episode_s")):
            plot_line(
                ax,
                steps,
                series("model_inference_time_per_episode_s"),
                color=colors["latency"],
            )
            ax.set_title("Model Time per Episode")
            ax.set_ylabel("Seconds / episode")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "No model-time per episode data", ha="center", va="center")
        ax.set_xlabel("num_inference_steps")
        ax.set_xticks(steps)

        ax = axes[1]
        if any(v is not None for v in series("future_video_psnr_mean")):
            plot_line(ax, steps, series("future_video_psnr_mean"), color=colors["psnr"])
            ax.set_title("Future Video PSNR")
            ax.set_ylabel("dB")
        else:
            plot_line(ax, steps, series("replans_per_episode"), color=colors["time"])
            ax.set_title("Replans per Episode")
            ax.set_ylabel("Count")
        ax.set_xlabel("num_inference_steps")
        ax.set_xticks(steps)

        fig.tight_layout(rect=[0, 0, 1, 0.90])
        pdf.savefig(fig)
        plt.close(fig)

print(f"Wrote summary CSV: {output_root / 'all_steps_metrics.csv'}")
print(f"Wrote suite CSV: {output_root / 'all_steps_suite_metrics.csv'}")
print(f"Wrote task CSV: {output_root / 'all_steps_task_metrics.csv'}")
print(f"Wrote PDF: {pdf_path}")
PY
}

if [[ "${PLOT_ONLY}" == "1" ]]; then
  echo "[plot-only] OUTPUT_ROOT=${OUTPUT_ROOT}"
  plot_sweep
  exit 0
fi

echo "Sweep output root: ${OUTPUT_ROOT}"
echo "Steps: ${STEPS}"
echo "Task: ${TASK}"
echo "Checkpoint: ${CKPT}"
echo "Dataset stats: ${DATASET_STATS_PATH}"
echo "NUM_GPUS=${NUM_GPUS}, MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU}, NUM_TRIALS=${NUM_TRIALS}"
echo "Extra Hydra args: ${EXTRA_ARGS[*]:-<none>}"

for step in "${STEP_ARRAY[@]}"; do
  RUN_DIR="${OUTPUT_ROOT}/steps_${step}"
  mkdir -p "${RUN_DIR}"
  echo "${step}" > "${RUN_DIR}/num_inference_steps.txt"

  if [[ "${SKIP_EXISTING}" == "1" && -s "${RUN_DIR}/summary.json" ]]; then
    wall_time="$(cat "${RUN_DIR}/wall_time_seconds.txt" 2>/dev/null || true)"
    printf "%s\t%s\t%s\t%s\n" "${step}" "${RUN_DIR}" "existing" "${wall_time:-}" >> "${RUN_INDEX}"
    echo "[skip] step=${step} already has ${RUN_DIR}/summary.json"
    continue
  fi

  echo
  echo "============================================================"
  echo "Running LIBERO evaluation with EVALUATION.num_inference_steps=${step}"
  echo "Output: ${RUN_DIR}"
  echo "============================================================"

  start_ts="$(date +%s)"
  set +e
  "${PYTHON_BIN}" experiments/libero/run_libero_manager.py \
    "task=${TASK}" \
    "ckpt=${CKPT}" \
    "EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}" \
    "EVALUATION.num_inference_steps=${step}" \
    "EVALUATION.num_trials=${NUM_TRIALS}" \
    "EVALUATION.output_dir=${RUN_DIR}" \
    "EVALUATION.visualize_future_video=${VISUALIZE_FUTURE_VIDEO}" \
    "MULTIRUN.num_gpus=${NUM_GPUS}" \
    "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/manager.log"
  rc="${PIPESTATUS[0]}"
  set -e
  end_ts="$(date +%s)"
  elapsed="$((end_ts - start_ts))"
  echo "${elapsed}" > "${RUN_DIR}/wall_time_seconds.txt"

  if [[ "${rc}" -eq 0 ]]; then
    status="success"
    if [[ ! -s "${RUN_DIR}/summary.json" ]]; then
      echo "[warn] summary.json missing after successful manager run; generating it now."
      CKPT="${CKPT}" CONFIG="${TASK}" "${PYTHON_BIN}" experiments/libero/summarize_results.py --output_dir="${RUN_DIR}"
    fi
  else
    status="failed"
  fi
  printf "%s\t%s\t%s\t%s\n" "${step}" "${RUN_DIR}" "${status}" "${elapsed}" >> "${RUN_INDEX}"

  if [[ "${rc}" -ne 0 ]]; then
    echo "[error] step=${step} failed with exit code ${rc}. See ${RUN_DIR}/manager.log" >&2
    plot_sweep || true
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit "${rc}"
    fi
  fi
done

plot_sweep

echo
echo "Sweep complete."
echo "Output root: ${OUTPUT_ROOT}"
echo "PDF: ${OUTPUT_ROOT}/num_inference_steps_sweep.pdf"
