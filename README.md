# BadWAM

**BadWAM: World-Action Desynchronization Attacks against World-Action Models**

This repository contains the research code for BadWAM, a framework for evaluating
adversarial observation perturbations against World-Action Models (WAMs).
BadWAM studies two attack objectives:

- **Action-only desynchronization**: perturb observations to maximize action shift.
- **Imagination-preserving desynchronization**: perturb observations to shift actions
  while keeping the predicted future comparatively close to the clean prediction.

The codebase is built on top of the FastWAM implementation and adds:

- query-based action-only and imagination-preserving attacks;
- LIBERO closed-loop attack evaluation;
- matched-strength stealth analysis;
- ablation experiments for attack strength and stealthiness;
- statistics export and plotting utilities.

> **Note**
> This release contains code only. Model checkpoints, dataset statistics, LIBERO data,
> RoboTwin assets, and generated experiment outputs are intentionally not included.

## Repository layout

```text
BadWAM/
├── configs/                         # Hydra configs for WAM variants and eval
├── experiments/
│   ├── attackwam/                   # BadWAM launchers and analysis scripts
│   ├── libero/                      # LIBERO closed-loop evaluation manager
│   └── robotwin/                    # RoboTwin integration hooks
├── scripts/                         # training / preprocessing entry points
├── src/
│   ├── attackwam/                   # attack implementation
│   └── fastwam/                     # FastWAM model/runtime code
├── third_party/RoboTwin/            # lightweight RoboTwin vendor snapshot
└── checkpoints/fastwam_release/     # placeholder for checkpoints and stats
```

## Installation

We recommend using a fresh conda environment with Python 3.10+.

```bash
conda create -n badwam python=3.10 -y
conda activate badwam

# Install PyTorch for your CUDA version first.
# Example only; please choose the command matching your system:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install BadWAM in editable mode.
pip install -e .
```

Some evaluation environments require additional simulator dependencies:

- LIBERO and robosuite for LIBERO closed-loop evaluation;
- RoboTwin/SAPIEN dependencies for RoboTwin experiments;
- `imageio`, `ffmpeg`, and `matplotlib` for rendering/analysis.

The exact simulator installation can be environment-specific, so we keep this
repository focused on the BadWAM attack/evaluation code.

## Checkpoints and dataset statistics

Place trained WAM checkpoints and corresponding dataset statistics under:

```text
checkpoints/fastwam_release/
├── libero_uncond_2cam224.pt
├── libero_uncond_2cam224_dataset_stats.json
├── libero_joint_2cam224.pt
├── libero_joint_2cam224_dataset_stats.json
├── libero_idm_2cam224.pt
└── libero_idm_2cam224_dataset_stats.json
```

Alternatively, pass explicit paths through environment variables:

```bash
CHECKPOINT_DIR=/path/to/checkpoints \
JOINT_DATASET_STATS_PATH=/path/to/joint_dataset_stats.json \
IDM_DATASET_STATS_PATH=/path/to/idm_dataset_stats.json \
DIRECT_DATASET_STATS_PATH=/path/to/direct_dataset_stats.json \
bash experiments/attackwam/run_libero_attackwam_study.sh
```

## Quick start: dry run

Before launching a long evaluation, verify commands with `DRY_RUN=1`:

```bash
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0

CHECKPOINT_DIR=/path/to/checkpoints \
MODELS="joint" \
ATTACKS="clean imagination_preserving" \
NUM_GPUS=1 \
NUM_TRIALS=1 \
DRY_RUN=1 \
bash experiments/attackwam/run_libero_attackwam_study.sh
```

## LIBERO attack evaluation

Run clean, action-only, and imagination-preserving attacks:

```bash
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0,1,2,3

CHECKPOINT_DIR=/path/to/checkpoints \
MODELS="direct joint idm" \
ATTACKS="clean action_only imagination_preserving" \
NUM_GPUS=4 \
NUM_TRIALS=20 \
bash experiments/attackwam/run_libero_attackwam_study.sh
```

Common environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RUN_ROOT` | `./evaluate_results/attackwam/...` | Output directory |
| `CHECKPOINT_DIR` | `./checkpoints/fastwam_release` | Checkpoint directory |
| `MODELS` | `direct joint idm` | WAM variants to evaluate |
| `ATTACKS` | `clean action_only imagination_preserving` | Attack list |
| `EPSILON` | `0.06` | $\ell_\infty$ perturbation bound |
| `ATTACK_SPACE` | `patch` | `patch` or `full_linf` |
| `IMAGINATION_PRESERVING_BUDGET` | `16` | Query budget for imagination-preserving attacks |
| `FUTURE_WEIGHT` | `1.0` | Future-preservation weight |
| `SAVE_WORLD` | `0` | Save predicted futures / heavy visual artifacts |
| `RESUME` | `0` | Skip runs with `RUN_COMPLETE` marker |

The script writes raw per-task JSON files, per-replan attack statistics, optional
representative samples, and aggregate analysis under `RUN_ROOT`.

## Matched-strength stealth experiment

This experiment compares an action-only objective and an imagination-preserving
objective under matched query budget and perturbation bound.

```bash
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0,1,2,3

CHECKPOINT_DIR=/path/to/checkpoints \
NUM_GPUS=4 \
NUM_TRIALS=20 \
EPSILON=0.06 \
BUDGET=16 \
ACTION_FUTURE_WEIGHT=0 \
IMGPRES_FUTURE_WEIGHT=0.05 \
bash experiments/attackwam/run_matched_strength_stealth.sh
```

After completion, the script runs:

```bash
python -m experiments.attackwam.analyze_matched_stealth --study-root <RUN_ROOT>
```

which exports matched-strength summary CSVs and trajectory-level statistics.

## Ablations

Run the balanced LIBERO subset ablation suite:

```bash
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0,1,2,3

CHECKPOINT_DIR=/path/to/checkpoints \
MODELS="joint idm" \
NUM_GPUS=4 \
NUM_TRIALS=10 \
RESUME=1 \
bash experiments/attackwam/run_attackwam_ablation_suite.sh
```

The default ablations cover:

- future-preserving weight;
- random noise baseline;
- query budget;
- perturbation budget.

## RoboTwin notes

`third_party/RoboTwin/` is included as a lightweight vendor snapshot for the
RoboTwin integration hooks. Large RoboTwin assets and task configs are not
included. Please follow RoboTwin's official instructions to populate:

```text
third_party/RoboTwin/assets/
third_party/RoboTwin/task_config/
```

## Reproducibility notes

- Use `DRY_RUN=1` before launching large sweeps.
- Use `RESUME=1` to skip runs containing a `RUN_COMPLETE` marker.
- Heavy artifacts are ignored by `.gitignore`: checkpoints, datasets, run outputs,
  videos, GIFs, `.npz` files, caches, and logs.
- For paper reproduction, keep the exact model checkpoints and dataset statistics
  used for each WAM variant.

## Acknowledgements

This repository builds on FastWAM and uses LIBERO/RoboTwin evaluation environments.
Please cite the corresponding upstream projects when using this code.

## Citation

```bibtex
@article{badwam2026,
  title   = {BadWAM: World-Action Desynchronization Attacks against World-Action Models},
  author  = {Anonymous Authors},
  journal = {Manuscript under submission},
  year    = {2026}
}
```
