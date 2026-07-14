#!/usr/bin/env bash
set -euo pipefail

# End-to-end LIBERO protocol for BadWAM.
#
# Example:
#   cd /path/to/BadWAM
#   export PYTHONPATH=$PWD/src
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   CHECKPOINT_DIR=/path/to/checkpoints \
#   JOINT_DATASET_STATS_PATH=/path/to/joint_dataset_stats.json \
#   IDM_DATASET_STATS_PATH=/path/to/idm_dataset_stats.json \
#   MODELS="direct joint idm" ATTACKS="clean action_only" NUM_TRIALS=5 \
#   bash experiments/attackwam/run_libero_attackwam_study.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/fastwam_release}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/evaluate_results/attackwam/libero_attackwam_$(date +%Y%m%d_%H%M%S)}"
MODELS="${MODELS:-direct joint idm}"
ATTACKS="${ATTACKS:-clean action_only imagination_preserving}"
NUM_TRIALS="${NUM_TRIALS:-20}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}"
SUITES="${SUITES:-libero_10,libero_goal,libero_spatial,libero_object}"
RUN_UNIVERSAL="${RUN_UNIVERSAL:-0}"
UNIVERSAL_TRAIN_TRIALS="${UNIVERSAL_TRAIN_TRIALS:-3}"
UNIVERSAL_TRAIN_STEPS="${UNIVERSAL_TRAIN_STEPS:-64}"
UNIVERSAL_TRAIN_BATCH_SIZE="${UNIVERSAL_TRAIN_BATCH_SIZE:-2}"
ATTACK_BUDGET="${ATTACK_BUDGET:-16}"
QUERY_ATTACK_BUDGET="${QUERY_ATTACK_BUDGET:-32}"
IMAGINATION_PRESERVING_BUDGET="${IMAGINATION_PRESERVING_BUDGET:-${ATTACK_BUDGET}}"
ATTACK_SPACE="${ATTACK_SPACE:-patch}"
PATCH_SIZE="${PATCH_SIZE:-32}"
EPSILON="${EPSILON:-0.06}"
ACTION_WEIGHT="${ACTION_WEIGHT:-1.0}"
FUTURE_WEIGHT="${FUTURE_WEIGHT:-1.0}"
FUTURE_VIDEO_HEIGHT="${FUTURE_VIDEO_HEIGHT:-64}"
FUTURE_VIDEO_WIDTH="${FUTURE_VIDEO_WIDTH:-128}"
FUTURE_VIDEO_MAX_FRAMES="${FUTURE_VIDEO_MAX_FRAMES:-0}"
SAVE_WORLD="${SAVE_WORLD:-0}"
DRY_RUN="${DRY_RUN:-0}"
PLOT_ONLY="${PLOT_ONLY:-0}"
EVAL_TASK_FILE="${EVAL_TASK_FILE:-}"
RUN_SUFFIX_TAG="${RUN_SUFFIX_TAG:-}"
EXTRA_HYDRA_OVERRIDES="${EXTRA_HYDRA_OVERRIDES:-}"
RESUME="${RESUME:-0}"
SKIP_FINAL_ANALYSIS="${SKIP_FINAL_ANALYSIS:-0}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

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

declare -A TASK_CONFIG=(
  [direct]="libero_uncond_2cam224_1e-4"
  [joint]="libero_joint_2cam224_1e-4"
  [idm]="libero_idm_2cam224_1e-4"
)
declare -A CHECKPOINT_NAME=(
  [direct]="libero_uncond_2cam224.pt"
  [joint]="libero_joint_2cam224.pt"
  [idm]="libero_idm_2cam224.pt"
)

dataset_stats_for_model() {
  local model="$1"
  case "$model" in
    joint) echo "${JOINT_DATASET_STATS_PATH:-${DATASET_STATS_PATH:-${CHECKPOINT_DIR}/libero_joint_2cam224_dataset_stats.json}}" ;;
    idm) echo "${IDM_DATASET_STATS_PATH:-${DATASET_STATS_PATH:-${CHECKPOINT_DIR}/libero_idm_2cam224_dataset_stats.json}}" ;;
    direct) echo "${DIRECT_DATASET_STATS_PATH:-${DATASET_STATS_PATH:-${CHECKPOINT_DIR}/libero_uncond_2cam224_dataset_stats.json}}" ;;
    *) echo "Unknown model '$model'" >&2; return 2 ;;
  esac
}

command_string() {
  printf '%q ' "$@"
}

attack_overrides() {
  local attack="$1"
  local universal_patch_path="${2:-}"
  case "$attack" in
    clean)
      echo "EVALUATION.attack.enabled=false"
      ;;
    action_only)
      echo "EVALUATION.attack.enabled=true EVALUATION.attack.name=action_only EVALUATION.attack.mode=action_only EVALUATION.attack.search=query EVALUATION.attack.space=full_linf EVALUATION.attack.epsilon=${EPSILON} EVALUATION.attack.budget=${ATTACK_BUDGET} EVALUATION.attack.require_future=false EVALUATION.attack.future_weight=0.0 EVALUATION.attack.action_weight=1.0"
      ;;
    imagination_preserving)
      echo "EVALUATION.attack.enabled=true EVALUATION.attack.name=imagination_preserving EVALUATION.attack.mode=imagination_preserving EVALUATION.attack.search=query EVALUATION.attack.space=${ATTACK_SPACE} EVALUATION.attack.patch_mode=additive EVALUATION.attack.patch_size=${PATCH_SIZE} EVALUATION.attack.epsilon=${EPSILON} EVALUATION.attack.budget=${IMAGINATION_PRESERVING_BUDGET} EVALUATION.attack.require_future=true EVALUATION.attack.action_weight=${ACTION_WEIGHT} +EVALUATION.attack.future_source=video EVALUATION.attack.future_weight=${FUTURE_WEIGHT} +EVALUATION.attack.future_video_skip_first=true +EVALUATION.attack.future_video_height=${FUTURE_VIDEO_HEIGHT} +EVALUATION.attack.future_video_width=${FUTURE_VIDEO_WIDTH} +EVALUATION.attack.future_video_max_frames=${FUTURE_VIDEO_MAX_FRAMES} +EVALUATION.attack.save_world=${SAVE_WORLD}"
      ;;
    random_uniform_noise|random_noise)
      echo "EVALUATION.attack.enabled=true EVALUATION.attack.name=random_uniform_noise EVALUATION.attack.mode=random_uniform_noise EVALUATION.attack.search=random EVALUATION.attack.space=full_linf EVALUATION.attack.patch_mode=additive EVALUATION.attack.epsilon=${EPSILON} EVALUATION.attack.budget=1 EVALUATION.attack.require_future=true EVALUATION.attack.action_weight=1.0 +EVALUATION.attack.future_source=video EVALUATION.attack.future_weight=0.0 +EVALUATION.attack.future_video_skip_first=true +EVALUATION.attack.future_video_height=${FUTURE_VIDEO_HEIGHT} +EVALUATION.attack.future_video_width=${FUTURE_VIDEO_WIDTH} +EVALUATION.attack.future_video_max_frames=${FUTURE_VIDEO_MAX_FRAMES} +EVALUATION.attack.save_world=${SAVE_WORLD}"
      ;;
    collect_universal)
      echo "EVALUATION.attack.enabled=true EVALUATION.attack.name=collect_universal EVALUATION.attack.mode=action_only EVALUATION.attack.search=random EVALUATION.attack.space=patch EVALUATION.attack.patch_size=${PATCH_SIZE} EVALUATION.attack.epsilon=${EPSILON} EVALUATION.attack.budget=1 EVALUATION.attack.require_future=false EVALUATION.attack.future_weight=0.0 EVALUATION.attack.save_images=true"
      ;;
    universal_patch)
      if [[ -z "$universal_patch_path" ]]; then
        echo "universal_patch requires a patch path" >&2
        return 2
      fi
      echo "EVALUATION.attack.enabled=true EVALUATION.attack.name=universal_patch EVALUATION.attack.mode=universal_patch EVALUATION.attack.search=load EVALUATION.attack.space=patch EVALUATION.attack.patch_mode=additive EVALUATION.attack.patch_size=${PATCH_SIZE} EVALUATION.attack.epsilon=${EPSILON} EVALUATION.attack.require_future=true EVALUATION.attack.patch_path=${universal_patch_path}"
      ;;
    *)
      echo "Unknown attack '$attack'" >&2
      return 2
      ;;
  esac
}

run_eval() {
  local model="$1"
  local attack="$2"
  local suffix="$3"
  local num_trials="$4"
  local universal_patch_path="${5:-}"
  local checkpoint="${CHECKPOINT_DIR}/${CHECKPOINT_NAME[${model}]}"
  local dataset_stats
  dataset_stats="$(dataset_stats_for_model "$model")"
  local output_suffix="${suffix}"
  if [[ -n "${RUN_SUFFIX_TAG}" ]]; then
    output_suffix="${output_suffix}_${RUN_SUFFIX_TAG}"
  fi
  local output_dir="${RUN_ROOT}/runs/${model}_${output_suffix}"
  local extra
  extra="$(attack_overrides "$attack" "$universal_patch_path")"
  local session_name="attackwam_${model}_${suffix}_$(date +%H%M%S)"
  local cmd=(
    "${PYTHON_BIN}" experiments/libero/run_libero_manager.py
    "task=${TASK_CONFIG[${model}]}"
    "ckpt=${checkpoint}"
    "seed=42"
    "EVALUATION.output_dir=${output_dir}"
    "EVALUATION.dataset_stats_path=${dataset_stats}"
    "EVALUATION.num_trials=${num_trials}"
    "EVALUATION.num_inference_steps=${NUM_INFERENCE_STEPS}"
    "EVALUATION.replan_steps=${REPLAN_STEPS}"
    "MULTIRUN.num_gpus=${NUM_GPUS}"
    "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"
    "MULTIRUN.task_suite_names=[${SUITES}]"
  )
  if [[ -n "${EVAL_TASK_FILE}" ]]; then
    cmd+=("MULTIRUN.task_file=${EVAL_TASK_FILE}")
  fi
  read -r -a extra_array <<< "${extra}"
  cmd+=("${extra_array[@]}")
  if [[ -n "${EXTRA_HYDRA_OVERRIDES}" ]]; then
    read -r -a extra_hydra_array <<< "${EXTRA_HYDRA_OVERRIDES}"
    cmd+=("${extra_hydra_array[@]}")
  fi
  if [[ "${RESUME}" == "1" && -f "${output_dir}/RUN_COMPLETE" ]]; then
    echo "Skipping completed ${model}/${attack}: ${output_dir}"
    return
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    command_string "${cmd[@]}"
    printf '\n'
    return
  fi
  mkdir -p "${output_dir}"
  printf '%s\n' "$(command_string "${cmd[@]}")" > "${output_dir}/command.txt"
  echo "Running ${model}/${attack}: ${output_dir}"
  LIBERO_TMUX_SESSION_NAME="${session_name}" "${cmd[@]}"
  touch "${output_dir}/RUN_COMPLETE"
}

train_universal_patch() {
  local model="$1"
  local collect_dir="${RUN_ROOT}/runs/${model}_collect_universal"
  local output_patch="${RUN_ROOT}/patches/${model}_universal_patch.npz"
  local dataset_stats
  dataset_stats="$(dataset_stats_for_model "$model")"
  mkdir -p "$(dirname "${output_patch}")"
  local input_glob="${collect_dir}/*/attack_raw/*.npz"
  local checkpoint="${CHECKPOINT_DIR}/${CHECKPOINT_NAME[${model}]}"
  local cmd=(
    "${PYTHON_BIN}" -m experiments.attackwam.train_universal_patch
    "task=${TASK_CONFIG[${model}]}"
    "ckpt=${checkpoint}"
    "EVALUATION.dataset_stats_path=${dataset_stats}"
    "EVALUATION.output_dir=${RUN_ROOT}/patch_training/${model}"
    "EVALUATION.attack.enabled=true"
    "EVALUATION.attack.mode=action_only"
    "EVALUATION.attack.space=patch"
    "EVALUATION.attack.patch_mode=additive"
    "EVALUATION.attack.patch_size=${PATCH_SIZE}"
    "EVALUATION.attack.epsilon=${EPSILON}"
    "EVALUATION.attack.budget=${QUERY_ATTACK_BUDGET}"
    "EVALUATION.attack.train_input_glob=${input_glob}"
    "EVALUATION.attack.train_output_path=${output_patch}"
    "EVALUATION.attack.train_steps=${UNIVERSAL_TRAIN_STEPS}"
    "EVALUATION.attack.train_batch_size=${UNIVERSAL_TRAIN_BATCH_SIZE}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    command_string "${cmd[@]}"
    printf '\n'
    return
  fi
  echo "Training universal patch for ${model}: ${output_patch}"
  "${cmd[@]}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUN_ROOT}/runs"
fi

if [[ "${PLOT_ONLY}" != "1" ]]; then
  read -r -a model_array <<< "${MODELS}"
  read -r -a attack_array <<< "${ATTACKS}"
  for model in "${model_array[@]}"; do
    if [[ -z "${TASK_CONFIG[${model}]+x}" ]]; then
      echo "Unknown model '${model}'. Expected direct, joint, or idm." >&2
      exit 1
    fi
    if [[ "${DRY_RUN}" != "1" ]]; then
      [[ -f "${CHECKPOINT_DIR}/${CHECKPOINT_NAME[${model}]}" ]] || {
        echo "Missing checkpoint: ${CHECKPOINT_DIR}/${CHECKPOINT_NAME[${model}]}" >&2
        exit 2
      }
      [[ -f "$(dataset_stats_for_model "$model")" ]] || {
        echo "Missing dataset stats: $(dataset_stats_for_model "$model")" >&2
        exit 2
      }
    fi
    for attack in "${attack_array[@]}"; do
      if [[ "${model}" == "direct" && "${attack}" == "imagination_preserving" ]]; then
        echo "Skipping ${model}/${attack}: Direct is action-only in this protocol; imagination-preserving desynchronization is run on Joint/IDM."
        continue
      fi
      run_eval "$model" "$attack" "$attack" "${NUM_TRIALS}"
    done
    if [[ "${RUN_UNIVERSAL}" == "1" ]]; then
      run_eval "$model" collect_universal collect_universal "${UNIVERSAL_TRAIN_TRIALS}"
      train_universal_patch "$model"
      run_eval "$model" universal_patch universal_patch "${NUM_TRIALS}" "${RUN_ROOT}/patches/${model}_universal_patch.npz"
    fi
  done
fi

if [[ "${DRY_RUN}" != "1" && "${SKIP_FINAL_ANALYSIS}" != "1" ]]; then
  "${PYTHON_BIN}" -m experiments.attackwam.summarize_attackwam --study-root "${RUN_ROOT}"
  "${PYTHON_BIN}" -m experiments.attackwam.plot_attackwam --analysis-dir "${RUN_ROOT}/analysis"
  echo "BadWAM study complete: ${RUN_ROOT}"
elif [[ "${DRY_RUN}" != "1" ]]; then
  echo "BadWAM study runs complete without final summarize/plot: ${RUN_ROOT}"
fi
