#!/usr/bin/env bash
set -euo pipefail

# Lightweight BadWAM ablation launcher.
#
# This runs the paper-ablation subset protocol:
#   - models: Joint + IDM by default
#   - tasks: 3 tasks per LIBERO suite by default
#   - trials: 10 per task by default
#   - ablations: future_weight, random noise, query budget, epsilon
#
# It does not touch any existing full-run directory.  Use DRY_RUN=1 first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/evaluate_results/attackwam/ablations_subset_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/fastwam_release}"
MODELS="${MODELS:-joint idm}"
NUM_TRIALS="${NUM_TRIALS:-10}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-0}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"

BASE_EPSILON="${BASE_EPSILON:-0.06}"
BASE_BUDGET="${BASE_BUDGET:-16}"
BASE_FUTURE_WEIGHT="${BASE_FUTURE_WEIGHT:-0.015}"

FUTURE_WEIGHT_VALUES="${FUTURE_WEIGHT_VALUES:-0 0.005 0.015 0.03 0.05 0.1 0.3}"
BUDGET_VALUES="${BUDGET_VALUES:-1 4 8 16 32}"
EPSILON_VALUES="${EPSILON_VALUES:-0.01 0.03 0.06 0.10 0.20}"
ABLATION_FAMILIES="${ABLATION_FAMILIES:-clean future random budget epsilon}"

LIBERO_10_TASKS="${LIBERO_10_TASKS:-0 4 9}"
LIBERO_GOAL_TASKS="${LIBERO_GOAL_TASKS:-0 4 9}"
LIBERO_SPATIAL_TASKS="${LIBERO_SPATIAL_TASKS:-0 4 9}"
LIBERO_OBJECT_TASKS="${LIBERO_OBJECT_TASKS:-0 4 9}"

FUTURE_VIDEO_HEIGHT="${FUTURE_VIDEO_HEIGHT:-64}"
FUTURE_VIDEO_WIDTH="${FUTURE_VIDEO_WIDTH:-128}"
FUTURE_VIDEO_MAX_FRAMES="${FUTURE_VIDEO_MAX_FRAMES:-0}"
SAVE_WORLD="${SAVE_WORLD:-0}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS="${NUM_GPUS:-$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | sed '/^$/d' | wc -l)}"
else
  NUM_GPUS="${NUM_GPUS:-4}"
fi

mkdir -p "${RUN_ROOT}/manifests"
TASK_FILE="${TASK_FILE:-${RUN_ROOT}/tasks_balanced_12.txt}"
CONFIG_CSV="${RUN_ROOT}/ablation_configs.csv"

normalize_ids() {
  tr ',' ' ' <<< "$1" | xargs
}

write_suite_tasks() {
  local suite="$1"
  local ids="$2"
  local count=0
  for task_id in $(normalize_ids "${ids}"); do
    printf '%s,%s\n' "${suite}" "${task_id}" >> "${TASK_FILE}"
    count=$((count + 1))
  done
  if [[ "${count}" -ne 3 ]]; then
    echo "Warning: ${suite} has ${count} selected tasks, expected 3." >&2
  fi
}

create_task_subset() {
  : > "${TASK_FILE}"
  write_suite_tasks "libero_10" "${LIBERO_10_TASKS}"
  write_suite_tasks "libero_goal" "${LIBERO_GOAL_TASKS}"
  write_suite_tasks "libero_spatial" "${LIBERO_SPATIAL_TASKS}"
  write_suite_tasks "libero_object" "${LIBERO_OBJECT_TASKS}"
  echo "Created ablation task subset: ${TASK_FILE}"
  sed 's/^/  /' "${TASK_FILE}"
}

sanitize_value() {
  python - "$1" <<'PY'
import sys
value = sys.argv[1]
print(value.replace("-", "m").replace(".", "p"))
PY
}

is_base_imgpres() {
  local attack="$1"
  local epsilon="$2"
  local budget="$3"
  local future_weight="$4"
  [[ "${attack}" == "imagination_preserving" \
    && "${epsilon}" == "${BASE_EPSILON}" \
    && "${budget}" == "${BASE_BUDGET}" \
    && "${future_weight}" == "${BASE_FUTURE_WEIGHT}" ]]
}

command_string() {
  printf '%q ' "$@"
}

wait_for_free_badwam_eval() {
  if [[ "${WAIT_FOR_FREE_GPUS}" != "1" ]]; then
    return
  fi
  local escaped_root
  escaped_root="$(printf '%s' "${REPO_ROOT}" | sed 's/[.[\*^$()+?{}|]/\\&/g')"
  while ps -eo pid,cmd | awk -v root="${escaped_root}" '
    /experiments\/libero\/(run_libero_manager.py|eval_libero_single.py)/ && $0 ~ root {
      found = 1
    }
    END { exit found ? 0 : 1 }
  '; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Existing BadWAM eval is still running; waiting ${WAIT_POLL_SECONDS}s..."
    sleep "${WAIT_POLL_SECONDS}"
  done
}

csv_escape() {
  python - "$1" <<'PY'
import csv, io, sys
buf = io.StringIO()
writer = csv.writer(buf)
writer.writerow([sys.argv[1]])
print(buf.getvalue().strip())
PY
}

record_config() {
  local tag="$1"
  local family="$2"
  local attack="$3"
  local epsilon="$4"
  local budget="$5"
  local future_weight="$6"
  local value="$7"
  if [[ ! -f "${CONFIG_CSV}" ]]; then
    printf 'tag,family,attack,epsilon,budget,future_weight,value,models,num_trials,task_file,run_root\n' > "${CONFIG_CSV}"
  fi
  if awk -F, -v tag="${tag}" 'NR > 1 && $1 == tag { found = 1 } END { exit found ? 0 : 1 }' "${CONFIG_CSV}"; then
    return
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(csv_escape "${tag}")" \
    "$(csv_escape "${family}")" \
    "$(csv_escape "${attack}")" \
    "$(csv_escape "${epsilon}")" \
    "$(csv_escape "${budget}")" \
    "$(csv_escape "${future_weight}")" \
    "$(csv_escape "${value}")" \
    "$(csv_escape "${MODELS}")" \
    "$(csv_escape "${NUM_TRIALS}")" \
    "$(csv_escape "${TASK_FILE}")" \
    "$(csv_escape "${RUN_ROOT}")" >> "${CONFIG_CSV}"
}

declare -A SEEN_TAGS=()

run_config() {
  local family="$1"
  local value="$2"
  local attack="$3"
  local epsilon="$4"
  local budget="$5"
  local future_weight="$6"
  local tag="$7"

  if [[ -n "${SEEN_TAGS[${tag}]+x}" ]]; then
    echo "Skipping duplicate ablation config tag=${tag}"
    return
  fi
  SEEN_TAGS["${tag}"]=1
  record_config "${tag}" "${family}" "${attack}" "${epsilon}" "${budget}" "${future_weight}" "${value}"

  local cmd_env=(
    "RUN_ROOT=${RUN_ROOT}"
    "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    "MODELS=${MODELS}"
    "ATTACKS=${attack}"
    "NUM_TRIALS=${NUM_TRIALS}"
    "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}"
    "REPLAN_STEPS=${REPLAN_STEPS}"
    "NUM_GPUS=${NUM_GPUS}"
    "MAX_TASKS_PER_GPU=${MAX_TASKS_PER_GPU}"
    "EVAL_TASK_FILE=${TASK_FILE}"
    "RUN_SUFFIX_TAG=${tag}"
    "ATTACK_SPACE=full_linf"
    "EPSILON=${epsilon}"
    "IMAGINATION_PRESERVING_BUDGET=${budget}"
    "FUTURE_WEIGHT=${future_weight}"
    "FUTURE_VIDEO_HEIGHT=${FUTURE_VIDEO_HEIGHT}"
    "FUTURE_VIDEO_WIDTH=${FUTURE_VIDEO_WIDTH}"
    "FUTURE_VIDEO_MAX_FRAMES=${FUTURE_VIDEO_MAX_FRAMES}"
    "SAVE_WORLD=${SAVE_WORLD}"
    "RESUME=${RESUME}"
    "DRY_RUN=${DRY_RUN}"
    "SKIP_FINAL_ANALYSIS=1"
  )

  echo
  echo "===== Ablation config: ${tag} (${family}=${value}) ====="
  if [[ "${DRY_RUN}" == "1" ]]; then
    command_string env "${cmd_env[@]}" bash experiments/attackwam/run_libero_attackwam_study.sh
    printf '\n'
    return
  fi
  wait_for_free_badwam_eval
  env "${cmd_env[@]}" bash experiments/attackwam/run_libero_attackwam_study.sh
}

has_family() {
  local needle="$1"
  for family in ${ABLATION_FAMILIES}; do
    [[ "${family}" == "${needle}" ]] && return 0
  done
  return 1
}

tag_for_imgpres_config() {
  local family="$1"
  local epsilon="$2"
  local budget="$3"
  local future_weight="$4"
  if is_base_imgpres "imagination_preserving" "${epsilon}" "${budget}" "${future_weight}"; then
    echo "base_imgpres"
  elif [[ "${family}" == "future" ]]; then
    echo "future_fw$(sanitize_value "${future_weight}")"
  elif [[ "${family}" == "budget" ]]; then
    echo "budget_b$(sanitize_value "${budget}")"
  elif [[ "${family}" == "epsilon" ]]; then
    echo "epsilon_eps$(sanitize_value "${epsilon}")"
  else
    echo "${family}_eps$(sanitize_value "${epsilon}")_b$(sanitize_value "${budget}")_fw$(sanitize_value "${future_weight}")"
  fi
}

create_task_subset

echo
echo "Ablation root: ${RUN_ROOT}"
echo "Models: ${MODELS}"
echo "Trials/task: ${NUM_TRIALS}"
echo "GPUs: ${NUM_GPUS}; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Families: ${ABLATION_FAMILIES}"
echo "DRY_RUN=${DRY_RUN} RESUME=${RESUME} WAIT_FOR_FREE_GPUS=${WAIT_FOR_FREE_GPUS}"

if has_family clean; then
  run_config "clean" "clean" "clean" "${BASE_EPSILON}" "${BASE_BUDGET}" "${BASE_FUTURE_WEIGHT}" "subset_clean"
fi

if has_family future; then
  for fw in ${FUTURE_WEIGHT_VALUES}; do
    tag="$(tag_for_imgpres_config future "${BASE_EPSILON}" "${BASE_BUDGET}" "${fw}")"
    run_config "future" "${fw}" "imagination_preserving" "${BASE_EPSILON}" "${BASE_BUDGET}" "${fw}" "${tag}"
  done
fi

if has_family random; then
  run_config "random" "uniform_linf" "random_uniform_noise" "${BASE_EPSILON}" "1" "0.0" "random_uniform_eps$(sanitize_value "${BASE_EPSILON}")"
fi

if has_family budget; then
  for budget in ${BUDGET_VALUES}; do
    tag="$(tag_for_imgpres_config budget "${BASE_EPSILON}" "${budget}" "${BASE_FUTURE_WEIGHT}")"
    run_config "budget" "${budget}" "imagination_preserving" "${BASE_EPSILON}" "${budget}" "${BASE_FUTURE_WEIGHT}" "${tag}"
  done
fi

if has_family epsilon; then
  for eps in ${EPSILON_VALUES}; do
    tag="$(tag_for_imgpres_config epsilon "${eps}" "${BASE_BUDGET}" "${BASE_FUTURE_WEIGHT}")"
    run_config "epsilon" "${eps}" "imagination_preserving" "${eps}" "${BASE_BUDGET}" "${BASE_FUTURE_WEIGHT}" "${tag}"
  done
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  python experiments/attackwam/analyze_ablation_suite.py --study-root "${RUN_ROOT}"
  echo "Ablation suite complete: ${RUN_ROOT}"
else
  echo "DRY_RUN complete. No experiments launched."
fi
