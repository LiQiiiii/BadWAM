#!/usr/bin/env bash
set -euo pipefail

# Paper-grade matched-strength stealth trade-off experiment.
#
# Goal:
#   Compare action-only objective vs. imagination-preserving objective under
#   matched closed-loop attack strength on Joint WAM.  Both settings query the
#   predicted future so that future-distance statistics are comparable, but the
#   action-only objective sets future_weight=0.
#
# Default protocol:
#   - model: Joint WAM
#   - benchmark: full LIBERO sweep
#   - trials: 20 per task
#   - GPUs: all visible GPUs, usually 0-7
#   - perturbation: full-image Linf, epsilon=0.06
#   - query budget: 16 perturbation queries + 1 clean query per attacked replan
#   - raw saved data: actions, predicted futures, clean/adv input images,
#     per-replan query trajectory statistics, and representatives.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/evaluate_results/attackwam/matched_strength_stealth_joint_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/fastwam_release}"
NUM_TRIALS="${NUM_TRIALS:-20}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
EPSILON="${EPSILON:-0.06}"
BUDGET="${BUDGET:-16}"
ACTION_FUTURE_WEIGHT="${ACTION_FUTURE_WEIGHT:-0}"
IMGPRES_FUTURE_WEIGHT="${IMGPRES_FUTURE_WEIGHT:-0.05}"
FUTURE_VIDEO_HEIGHT="${FUTURE_VIDEO_HEIGHT:-64}"
FUTURE_VIDEO_WIDTH="${FUTURE_VIDEO_WIDTH:-128}"
FUTURE_VIDEO_MAX_FRAMES="${FUTURE_VIDEO_MAX_FRAMES:-0}"
SAVE_WORLD="${SAVE_WORLD:-1}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ACTION_ONLY="${RUN_ACTION_ONLY:-1}"
RUN_IMGPRES="${RUN_IMGPRES:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
ANALYSIS_MAX_REPLANS_PER_RUN="${ANALYSIS_MAX_REPLANS_PER_RUN:-0}"
EXTRA_HYDRA_OVERRIDES_USER="${EXTRA_HYDRA_OVERRIDES:-}"

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

mkdir -p "${RUN_ROOT}"
cat > "${RUN_ROOT}/matched_strength_manifest.json" <<EOF
{
  "created_at": "$(date '+%Y-%m-%d %H:%M:%S')",
  "run_root": "${RUN_ROOT}",
  "checkpoint_dir": "${CHECKPOINT_DIR}",
  "model": "joint",
  "num_trials": ${NUM_TRIALS},
  "num_gpus": ${NUM_GPUS},
  "epsilon": ${EPSILON},
  "budget": ${BUDGET},
  "action_future_weight": ${ACTION_FUTURE_WEIGHT},
  "imgpres_future_weight": ${IMGPRES_FUTURE_WEIGHT},
  "save_world": ${SAVE_WORLD}
}
EOF

COMMON_EXTRA_OVERRIDES=(
  "EVALUATION.attack.save_raw=true"
  "EVALUATION.attack.save_images=true"
  "+EVALUATION.attack.save_representatives=true"
  "+EVALUATION.attack.representative_keep=40"
  "+EVALUATION.attack.representative_save_images=true"
  "+EVALUATION.attack.representative_save_world=true"
  "+EVALUATION.attack.save_trajectory=true"
)

join_overrides() {
  local out=""
  for item in "$@"; do
    out+="${item} "
  done
  printf '%s' "${out% }"
}

run_one() {
  local tag="$1"
  local future_weight="$2"
  local log_path="${RUN_ROOT}/${tag}_$(date +%Y%m%d_%H%M%S).log"
  echo
  echo "===== Matched-strength setting: ${tag}, future_weight=${future_weight} ====="
  echo "Log: ${log_path}"
  EXTRA_HYDRA_OVERRIDES="$(join_overrides "${COMMON_EXTRA_OVERRIDES[@]}") ${EXTRA_HYDRA_OVERRIDES_USER}" \
  RUN_ROOT="${RUN_ROOT}" \
  CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  MODELS="joint" \
  ATTACKS="imagination_preserving" \
  NUM_GPUS="${NUM_GPUS}" \
  NUM_TRIALS="${NUM_TRIALS}" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  REPLAN_STEPS="${REPLAN_STEPS}" \
  MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU}" \
  RUN_SUFFIX_TAG="${tag}" \
  ATTACK_SPACE="full_linf" \
  EPSILON="${EPSILON}" \
  IMAGINATION_PRESERVING_BUDGET="${BUDGET}" \
  FUTURE_WEIGHT="${future_weight}" \
  FUTURE_VIDEO_HEIGHT="${FUTURE_VIDEO_HEIGHT}" \
  FUTURE_VIDEO_WIDTH="${FUTURE_VIDEO_WIDTH}" \
  FUTURE_VIDEO_MAX_FRAMES="${FUTURE_VIDEO_MAX_FRAMES}" \
  SAVE_WORLD="${SAVE_WORLD}" \
  RESUME="${RESUME}" \
  DRY_RUN="${DRY_RUN}" \
  SKIP_FINAL_ANALYSIS=1 \
  bash experiments/attackwam/run_libero_attackwam_study.sh 2>&1 | tee -a "${log_path}"
}

echo "Matched-strength run root: ${RUN_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} NUM_GPUS=${NUM_GPUS}"
echo "NUM_TRIALS=${NUM_TRIALS} EPSILON=${EPSILON} BUDGET=${BUDGET}"
echo "Action future_weight=${ACTION_FUTURE_WEIGHT}; Img-pres future_weight=${IMGPRES_FUTURE_WEIGHT}"
echo "Raw images/worlds will be saved. Free space:"
df -h "${RUN_ROOT}" || true

if [[ "${RUN_ACTION_ONLY}" == "1" ]]; then
  run_one "matched_action_fw0" "${ACTION_FUTURE_WEIGHT}"
fi

if [[ "${RUN_IMGPRES}" == "1" ]]; then
  run_one "matched_imgpres_fw0p05" "${IMGPRES_FUTURE_WEIGHT}"
fi

if [[ "${DRY_RUN}" != "1" && "${RUN_ANALYSIS}" == "1" ]]; then
  "${PYTHON_BIN}" -m experiments.attackwam.analyze_matched_stealth \
    --study-root "${RUN_ROOT}" \
    --max-replans-per-run "${ANALYSIS_MAX_REPLANS_PER_RUN}" \
    2>&1 | tee -a "${RUN_ROOT}/matched_strength_analysis_$(date +%Y%m%d_%H%M%S).log"
fi

echo "Matched-strength stealth experiment complete or queued: ${RUN_ROOT}"
