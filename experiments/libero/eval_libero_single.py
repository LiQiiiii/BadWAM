import json
import inspect
import fcntl
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark
from action_ensembler import ActionEnsembler
from attackwam.attacks import AttackResult, QueryOutput, build_attack, save_attack_patch

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _synchronize_if_cuda(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    attack_cfg = cfg.EVALUATION.get("attack", {})
    if bool(attack_cfg.get("enabled", False)) and bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError(
            "EVALUATION.attack.enabled=true and EVALUATION.visualize_future_video=true "
            "are intentionally not combined in the main rollout path. Use attack latents/statistics "
            "for full runs and launch a small visualization-only run separately."
        )
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _model_infer(
    model: torch.nn.Module,
    infer_kwargs: dict[str, Any],
    *,
    input_image: torch.Tensor,
    visualize_future_video: bool = False,
    return_world_latents: bool = False,
    future_source: str = "latents",
    decode_video: bool = False,
) -> dict[str, Any]:
    kwargs = dict(infer_kwargs)
    kwargs["input_image"] = input_image
    future_source_key = str(future_source).strip().lower()
    needs_decoded_video = return_world_latents and future_source_key in {
        "video",
        "decoded_video",
        "future_video",
        "predicted_video",
    }
    if visualize_future_video or needs_decoded_video:
        joint_signature = inspect.signature(model.infer_joint)
        if "num_video_frames" not in kwargs:
            raise ValueError(
                "Decoded-video future queries require `num_video_frames` in infer_kwargs. "
                "This is normally set from cfg.data.train.num_frames."
            )
        if "test_action_with_infer_action" in joint_signature.parameters:
            # Avoid the extra action-only consistency query. The attack needs the
            # joint action/video output from this single WAM query.
            kwargs["test_action_with_infer_action"] = False
        return model.infer_joint(**kwargs)

    signature = inspect.signature(model.infer_action)
    if return_world_latents and "return_world_latents" in signature.parameters:
        kwargs["return_world_latents"] = True
    if "decode_video" in signature.parameters:
        kwargs["decode_video"] = bool(decode_video)
    return model.infer_action(**kwargs)


def _attack_cfg_get(attack_cfg: Any, key: str, default: Any = None) -> Any:
    if attack_cfg is None:
        return default
    if isinstance(attack_cfg, dict):
        return attack_cfg.get(key, default)
    if hasattr(attack_cfg, "get"):
        return attack_cfg.get(key, default)
    return getattr(attack_cfg, key, default)


def _as_eval_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _future_source_uses_video(future_source: str) -> bool:
    return str(future_source).strip().lower() in {
        "video",
        "decoded_video",
        "future_video",
        "predicted_video",
    }


def _video_frames_to_world_tensor(
    frames: Any,
    attack_cfg: Any,
) -> Optional[torch.Tensor]:
    if frames is None:
        return None
    if not isinstance(frames, (list, tuple)):
        return None
    selected = list(frames)
    if len(selected) == 0:
        return None

    if _as_eval_bool(_attack_cfg_get(attack_cfg, "future_video_skip_first", True)) and len(selected) > 1:
        selected = selected[1:]

    max_frames = int(_attack_cfg_get(attack_cfg, "future_video_max_frames", 0) or 0)
    if max_frames > 0:
        selected = selected[:max_frames]
    if len(selected) == 0:
        return None

    height = int(_attack_cfg_get(attack_cfg, "future_video_height", 64) or 64)
    width = int(_attack_cfg_get(attack_cfg, "future_video_width", 128) or 128)
    height = max(1, height)
    width = max(1, width)

    arrays: list[np.ndarray] = []
    for frame in selected:
        if isinstance(frame, Image.Image):
            pil = frame.convert("RGB")
        else:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                if float(np.nanmax(arr)) <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)
            pil = Image.fromarray(arr).convert("RGB")
        if pil.size != (width, height):
            pil = pil.resize((width, height), resample=Image.BILINEAR)
        arrays.append(np.asarray(pil, dtype=np.float32) / 255.0)

    video = np.stack(arrays, axis=0)  # [T,H,W,C]
    return torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]


def _query_output_from_pred(
    pred: dict[str, Any],
    *,
    attack_cfg: Any = None,
    require_future: bool = False,
) -> QueryOutput:
    action = pred["action"]
    future_source = str(_attack_cfg_get(attack_cfg, "future_source", "latents"))
    if _future_source_uses_video(future_source):
        world_tensor = _video_frames_to_world_tensor(pred.get("video"), attack_cfg)
    else:
        world = pred.get("world_latents")
        if isinstance(world, torch.Tensor):
            world_tensor = world
        else:
            world_tensor = None
    if require_future and world_tensor is None and _future_source_uses_video(future_source):
        raise ValueError(
            "Attack requested `future_source=video`, but the model prediction did not contain decoded video. "
            "Use this mode with Joint/IDM `infer_joint` capable models."
        )
    return QueryOutput(action=action, world=world_tensor, raw=pred)


def _attack_array(tensor: Optional[torch.Tensor], dtype=np.float16) -> Optional[np.ndarray]:
    if tensor is None:
        return None
    return tensor.detach().cpu().float().numpy().astype(dtype)


def _image_tensor_to_u8(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if tensor is None:
        return None
    arr = (
        ((tensor.detach().cpu().float().clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    return arr


def _metric_from_stats(stats: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = stats.get(key, default)
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _representative_score(success: bool, attack_rows: list[dict[str, Any]]) -> float:
    """Rank examples for paper figures: failures first, then high action / low future drift."""
    if not attack_rows:
        return float("-inf")
    actions = np.asarray([_metric_from_stats(row, "action_distance") for row in attack_rows], dtype=np.float64)
    futures = np.asarray([_metric_from_stats(row, "future_distance") for row in attack_rows], dtype=np.float64)
    desynchronizations = np.asarray([_metric_from_stats(row, "desynchronization_score") for row in attack_rows], dtype=np.float64)
    perturb_l1 = np.asarray([_metric_from_stats(row, "perturb_l1") for row in attack_rows], dtype=np.float64)
    finite_desynchronization = desynchronizations[np.isfinite(desynchronizations)]
    mean_desynchronization = float(np.mean(np.log1p(np.maximum(finite_desynchronization, 0.0)))) if finite_desynchronization.size else 0.0
    return (
        (100.0 if not success else 0.0)
        + 20.0 * float(np.nanmean(actions))
        + 3.0 * mean_desynchronization
        - 5.0 * float(np.nanmean(futures))
        - 0.2 * float(np.nanmean(perturb_l1))
    )


def _stack_optional(rows: list[dict[str, Any]], key: str) -> Optional[np.ndarray]:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return np.stack(values, axis=0)


def _atomic_json_dump(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)
    tmp.replace(path)


def _save_representative_sample(
    cfg: DictConfig,
    episode_idx: int,
    *,
    success: bool,
    episode_stats: dict[str, Any],
    attack_raw_rows: list[dict[str, Any]],
) -> Optional[str]:
    attack_cfg = cfg.EVALUATION.get("attack", {})
    if not attack_raw_rows or not _as_eval_bool(_attack_cfg_get(attack_cfg, "save_representatives", True)):
        return None

    attack_rows = episode_stats.get("attack_replan_stats", [])
    score = _representative_score(success, attack_rows)
    if not np.isfinite(score):
        return None

    keep = max(0, int(_attack_cfg_get(attack_cfg, "representative_keep", 20) or 20))
    if keep <= 0:
        return None

    rep_dir = Path(cfg.EVALUATION.output_dir) / "representatives"
    sample_dir = rep_dir / "samples"
    rep_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = rep_dir / "manifest.json"
    lock_path = rep_dir / "manifest.lock"
    sample_id = (
        f"{cfg.EVALUATION.task_suite_name}_task{int(cfg.EVALUATION.task_id)}"
        f"_trial{int(episode_idx)}_gpu{int(cfg.get('gpu_id', 0))}"
    )
    sample_path = sample_dir / f"{sample_id}.npz"

    metadata = {
        "sample_id": sample_id,
        "score": float(score),
        "success": bool(success),
        "task_suite": str(cfg.EVALUATION.task_suite_name),
        "task_id": int(cfg.EVALUATION.task_id),
        "trial_id": int(episode_idx),
        "gpu_id": int(cfg.get("gpu_id", 0)),
        "num_replans": int(episode_stats.get("num_replans", len(attack_raw_rows))),
        "executed_action_steps": int(episode_stats.get("executed_action_steps", 0)),
        "episode_wall_time_s": float(episode_stats.get("episode_wall_time_s", 0.0)),
        "attack_summary": episode_stats.get("attack_summary", {}),
        "attack_raw_path": episode_stats.get("attack_raw_path"),
    }

    payload: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, cls=NumpyEncoder), dtype=object),
        "stats_json": np.asarray(
            [json.dumps(row["stats"], cls=NumpyEncoder) for row in attack_raw_rows],
            dtype=object,
        ),
    }
    for key, out_key in (
        ("clean_action", "clean_actions"),
        ("adv_action", "adv_actions"),
        ("patch", "patches"),
        ("mask", "masks"),
        ("proprio", "proprios"),
        ("clean_image", "clean_images"),
        ("adv_image", "adv_images"),
        ("clean_world", "clean_worlds"),
        ("adv_world", "adv_worlds"),
    ):
        arr = _stack_optional(attack_raw_rows, key)
        if arr is not None:
            payload[out_key] = arr
    prompts = [row["prompt"] for row in attack_raw_rows if row.get("prompt") is not None]
    if prompts:
        payload["prompts_json"] = np.asarray([json.dumps(prompt) for prompt in prompts], dtype=object)

    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
            else:
                manifest = {"keep": keep, "samples": []}

            samples = [s for s in manifest.get("samples", []) if s.get("sample_id") != sample_id]
            samples.append(
                {
                    "sample_id": sample_id,
                    "score": float(score),
                    "success": bool(success),
                    "path": str(sample_path),
                    "task_suite": str(cfg.EVALUATION.task_suite_name),
                    "task_id": int(cfg.EVALUATION.task_id),
                    "trial_id": int(episode_idx),
                    "attack_action_distance_mean": float(
                        np.mean([_metric_from_stats(row, "action_distance") for row in attack_rows])
                    ),
                    "attack_future_distance_mean": float(
                        np.mean([_metric_from_stats(row, "future_distance") for row in attack_rows])
                    ),
                    "attack_desynchronization_score_mean": float(
                        np.mean([_metric_from_stats(row, "desynchronization_score") for row in attack_rows])
                    ),
                }
            )
            samples.sort(key=lambda item: float(item.get("score", float("-inf"))), reverse=True)
            kept = samples[:keep]
            dropped = samples[keep:]
            kept_ids = {s["sample_id"] for s in kept}

            if sample_id in kept_ids:
                tmp_path = sample_path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp_path, **payload)
                tmp_path.replace(sample_path)
            for item in dropped:
                try:
                    Path(item.get("path", "")).unlink(missing_ok=True)
                except OSError:
                    pass

            _atomic_json_dump(
                manifest_path,
                {
                    "keep": keep,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "ranking": "failure_bonus + action_distance + log(desynchronization_score) - future_distance - perturb_l1",
                    "samples": kept,
                },
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return str(sample_path) if sample_path.exists() else None


def _save_attack_raw(
    cfg: DictConfig,
    episode_idx: int,
    attack_raw_rows: list[dict[str, Any]],
) -> Optional[str]:
    if not attack_raw_rows:
        return None
    attack_cfg = cfg.EVALUATION.get("attack", {})
    if not bool(attack_cfg.get("save_raw", True)):
        return None

    raw_dir = (
        Path(cfg.EVALUATION.output_dir)
        / str(cfg.EVALUATION.task_suite_name)
        / "attack_raw"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    gpu_id = int(cfg.get("gpu_id", 0))
    raw_path = raw_dir / (
        f"gpu{gpu_id}_task{int(cfg.EVALUATION.task_id)}_trial{episode_idx}.npz"
    )

    clean_actions = [row["clean_action"] for row in attack_raw_rows if row.get("clean_action") is not None]
    adv_actions = [row["adv_action"] for row in attack_raw_rows if row.get("adv_action") is not None]
    patches = [row["patch"] for row in attack_raw_rows if row.get("patch") is not None]
    masks = [row["mask"] for row in attack_raw_rows if row.get("mask") is not None]
    save_images = _as_eval_bool(_attack_cfg_get(attack_cfg, "save_images", False))
    save_world = _as_eval_bool(_attack_cfg_get(attack_cfg, "save_world", False))
    clean_images = [row["clean_image"] for row in attack_raw_rows if save_images and row.get("clean_image") is not None]
    adv_images = [row["adv_image"] for row in attack_raw_rows if save_images and row.get("adv_image") is not None]
    clean_worlds = [row["clean_world"] for row in attack_raw_rows if save_world and row.get("clean_world") is not None]
    adv_worlds = [row["adv_world"] for row in attack_raw_rows if save_world and row.get("adv_world") is not None]
    proprios = [row["proprio"] for row in attack_raw_rows if row.get("proprio") is not None]
    prompts = [row["prompt"] for row in attack_raw_rows if row.get("prompt") is not None]
    stats_json = np.asarray(
        [json.dumps(row["stats"], cls=NumpyEncoder) for row in attack_raw_rows],
        dtype=object,
    )
    payload: dict[str, Any] = {
        "stats_json": stats_json,
        "task_id": np.asarray(int(cfg.EVALUATION.task_id), dtype=np.int32),
        "trial_id": np.asarray(int(episode_idx), dtype=np.int32),
    }
    if clean_actions:
        payload["clean_actions"] = np.stack(clean_actions, axis=0)
    if adv_actions:
        payload["adv_actions"] = np.stack(adv_actions, axis=0)
    if patches:
        payload["patches"] = np.stack(patches, axis=0)
    if masks:
        payload["masks"] = np.stack(masks, axis=0)
    if clean_images:
        payload["clean_images"] = np.stack(clean_images, axis=0)
    if adv_images:
        payload["adv_images"] = np.stack(adv_images, axis=0)
    if clean_worlds:
        payload["clean_worlds"] = np.stack(clean_worlds, axis=0)
    if adv_worlds:
        payload["adv_worlds"] = np.stack(adv_worlds, axis=0)
    if proprios:
        payload["proprios"] = np.stack(proprios, axis=0)
    if prompts:
        payload["prompts_json"] = np.asarray([json.dumps(prompt) for prompt in prompts], dtype=object)
    temp_path = raw_path.with_suffix(".tmp.npz")
    np.savez_compressed(temp_path, **payload)
    temp_path.replace(raw_path)
    return str(raw_path)


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]], dict[str, Any]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    attack = build_attack(cfg.EVALUATION)
    attack_cfg = cfg.EVALUATION.get("attack", {})
    attack_future_source = str(_attack_cfg_get(attack_cfg, "future_source", "latents"))
    attack_needs_decoded_video = (
        attack is not None
        and _as_eval_bool(_attack_cfg_get(attack_cfg, "require_future", True))
        and _future_source_uses_video(attack_future_source)
    )

    predicted_future_frames = None
    if visualize_future_video or attack_needs_decoded_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    attack_result: Optional[AttackResult] = None

    _synchronize_if_cuda(model_device)
    inference_start = time.perf_counter()
    with torch.no_grad():
        if attack is not None:
            def query_fn(query_image: torch.Tensor, require_future: bool) -> QueryOutput:
                query_pred = _model_infer(
                    model,
                    infer_kwargs,
                    input_image=query_image,
                    visualize_future_video=False,
                    return_world_latents=bool(require_future),
                    future_source=attack_future_source,
                    decode_video=False,
                )
                return _query_output_from_pred(
                    query_pred,
                    attack_cfg=attack_cfg,
                    require_future=bool(require_future),
                )

            attack_result = attack.optimize(image=image, query_fn=query_fn)
            if attack_result.best_output is None or attack_result.best_output.raw is None:
                pred = _model_infer(
                    model,
                    infer_kwargs,
                    input_image=attack_result.image,
                    visualize_future_video=False,
                    return_world_latents=False,
                    future_source=attack_future_source,
                    decode_video=False,
                )
            else:
                pred = attack_result.best_output.raw
        elif visualize_future_video:
            pred = _model_infer(
                model,
                infer_kwargs,
                input_image=image,
                visualize_future_video=True,
            )
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = _model_infer(
                model,
                infer_kwargs,
                input_image=image,
                visualize_future_video=False,
            )
    _synchronize_if_cuda(model_device)
    inference_time_s = time.perf_counter() - inference_start
    action = pred["action"]  # [T, D]
    normalized_action = action.detach().cpu().float().numpy()

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    infer_stats = {
        "num_inference_steps": num_inference_steps,
        "model_inference_time_s": float(inference_time_s),
        "action_chunk_len": int(action.shape[0]),
        "visualize_future_video": bool(visualize_future_video),
    }
    if attack_result is not None:
        attack_stats = dict(attack_result.stats)
        attack_stats["model_inference_time_includes_attack"] = True
        infer_stats["attack"] = attack_stats
        save_raw_enabled = _as_eval_bool(_attack_cfg_get(attack_cfg, "save_raw", True))
        save_representatives = _as_eval_bool(_attack_cfg_get(attack_cfg, "save_representatives", True))
        save_images_for_raw = _as_eval_bool(_attack_cfg_get(attack_cfg, "save_images", False))
        save_world_for_raw = _as_eval_bool(_attack_cfg_get(attack_cfg, "save_world", False))
        save_images_for_rep = _as_eval_bool(_attack_cfg_get(attack_cfg, "representative_save_images", True))
        save_world_for_rep = _as_eval_bool(_attack_cfg_get(attack_cfg, "representative_save_world", True))
        if save_raw_enabled or save_representatives:
            infer_stats["_attack_raw"] = {
                "stats": attack_stats,
                "prompt": prompt,
                "clean_action": _attack_array(
                    attack_result.clean_output.action if attack_result.clean_output is not None else None
                ),
                "adv_action": normalized_action.astype(np.float16),
                "patch": _attack_array(attack_result.patch),
                "mask": _attack_array(attack_result.mask),
                "proprio": _attack_array(proprio, dtype=np.float32),
                "clean_world": (
                    _attack_array(attack_result.clean_output.world)
                    if (
                        attack_result.clean_output is not None
                        and (save_world_for_raw or (save_representatives and save_world_for_rep))
                    )
                    else None
                ),
                "adv_world": (
                    _attack_array(attack_result.best_output.world)
                    if (
                        attack_result.best_output is not None
                        and (save_world_for_raw or (save_representatives and save_world_for_rep))
                    )
                    else None
                ),
                "clean_image": (
                    _image_tensor_to_u8(image)
                    if (save_images_for_raw or (save_representatives and save_images_for_rep))
                    else None
                ),
                "adv_image": (
                    _image_tensor_to_u8(attack_result.image)
                    if (save_images_for_raw or (save_representatives and save_images_for_rep))
                    else None
                ),
            }
        if bool(cfg.EVALUATION.get("attack", {}).get("save_patch", True)) and attack_result.patch is not None:
            patch_dir = Path(cfg.EVALUATION.output_dir) / str(cfg.EVALUATION.task_suite_name) / "attack_patches"
            patch_path = patch_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_latest_patch.npz"
            save_attack_patch(patch_path, attack_result.patch, metadata=attack_stats)
    return action, imgs, predicted_future_frames, infer_stats


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float], dict[str, Any]]:
    episode_start_time = time.perf_counter()
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1
    model_inference_times_s: list[float] = []
    attack_replan_stats: list[dict[str, Any]] = []
    attack_raw_rows: list[dict[str, Any]] = []
    executed_action_steps = 0

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs, predicted_future_frames, infer_stats = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            model_inference_times_s.append(float(infer_stats["model_inference_time_s"]))
            if "attack" in infer_stats:
                attack_replan_stats.append(infer_stats["attack"])
            raw_attack = infer_stats.pop("_attack_raw", None)
            if raw_attack is not None:
                attack_raw_rows.append(raw_attack)
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        executed_action_steps += 1
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    if model_inference_times_s:
        inference_array = np.asarray(model_inference_times_s, dtype=np.float64)
        mean_inference_time = float(np.mean(inference_array))
        p50_inference_time = float(np.percentile(inference_array, 50))
        p95_inference_time = float(np.percentile(inference_array, 95))
        max_inference_time = float(np.max(inference_array))
    else:
        mean_inference_time = None
        p50_inference_time = None
        p95_inference_time = None
        max_inference_time = None

    episode_stats = {
        "episode_wall_time_s": float(time.perf_counter() - episode_start_time),
        "executed_action_steps": int(executed_action_steps),
        "num_replans": int(len(model_inference_times_s)),
        "model_inference_times_s": model_inference_times_s,
        "model_inference_time_total_s": float(sum(model_inference_times_s)),
        "model_inference_time_mean_s": mean_inference_time,
        "model_inference_time_p50_s": p50_inference_time,
        "model_inference_time_p95_s": p95_inference_time,
        "model_inference_time_max_s": max_inference_time,
    }
    if attack_replan_stats:
        scalar_keys = (
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
        attack_summary: dict[str, Any] = {
            "enabled": True,
            "num_replans": len(attack_replan_stats),
            "mode": attack_replan_stats[0].get("mode"),
            "space": attack_replan_stats[0].get("space"),
            "search": attack_replan_stats[0].get("search"),
            "future_source": attack_replan_stats[0].get("future_source"),
            "future_video_skip_first": attack_replan_stats[0].get("future_video_skip_first"),
            "future_video_height": attack_replan_stats[0].get("future_video_height"),
            "future_video_width": attack_replan_stats[0].get("future_video_width"),
            "future_video_max_frames": attack_replan_stats[0].get("future_video_max_frames"),
            "used_future_term_rate": float(
                np.mean([float(bool(row.get("used_future_term", False))) for row in attack_replan_stats])
            ),
        }
        for key in scalar_keys:
            values = np.asarray(
                [float(row[key]) for row in attack_replan_stats if row.get(key) is not None],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            attack_summary[f"{key}_mean"] = float(values.mean())
            attack_summary[f"{key}_max"] = float(values.max())
            attack_summary[f"{key}_p95"] = float(np.percentile(values, 95))
        episode_stats["attack_replan_stats"] = attack_replan_stats
        episode_stats["attack_summary"] = attack_summary
    raw_path = _save_attack_raw(cfg, episode_idx, attack_raw_rows)
    if raw_path is not None:
        episode_stats["attack_raw_path"] = raw_path
    representative_path = _save_representative_sample(
        cfg,
        episode_idx,
        success=bool(done),
        episode_stats=episode_stats,
        attack_raw_rows=attack_raw_rows,
    )
    if representative_path is not None:
        episode_stats["attack_representative_path"] = representative_path
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr, episode_stats


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
        "episode_inference_stats": [],
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images, predicted_future_video_clips, episode_mean_psnr, episode_stats = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)
        results["episode_inference_stats"].append(episode_stats)

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))

    inference_times = [
        float(value)
        for episode_stats in results["episode_inference_stats"]
        for value in episode_stats.get("model_inference_times_s", [])
    ]
    replan_counts = [
        int(episode_stats.get("num_replans", 0))
        for episode_stats in results["episode_inference_stats"]
    ]
    executed_steps = [
        int(episode_stats.get("executed_action_steps", 0))
        for episode_stats in results["episode_inference_stats"]
    ]
    episode_wall_times = [
        float(episode_stats.get("episode_wall_time_s", 0.0))
        for episode_stats in results["episode_inference_stats"]
    ]
    attack_rows = [
        row
        for episode_stats in results["episode_inference_stats"]
        for row in episode_stats.get("attack_replan_stats", [])
    ]
    results["model_inference_count"] = int(len(inference_times))
    results["model_inference_time_total_s"] = float(sum(inference_times))
    if len(inference_times) > 0:
        inference_array = np.asarray(inference_times, dtype=np.float64)
        results["model_inference_time_mean_s"] = float(np.mean(inference_array))
        results["model_inference_time_p50_s"] = float(np.percentile(inference_array, 50))
        results["model_inference_time_p95_s"] = float(np.percentile(inference_array, 95))
        results["model_inference_time_max_s"] = float(np.max(inference_array))
    else:
        results["model_inference_time_mean_s"] = None
        results["model_inference_time_p50_s"] = None
        results["model_inference_time_p95_s"] = None
        results["model_inference_time_max_s"] = None
    results["replans_per_episode_mean"] = float(np.mean(replan_counts)) if replan_counts else 0.0
    results["executed_action_steps_mean"] = float(np.mean(executed_steps)) if executed_steps else 0.0
    results["episode_wall_time_mean_s"] = float(np.mean(episode_wall_times)) if episode_wall_times else 0.0
    if attack_rows:
        results["attack_enabled"] = True
        results["attack_mode"] = attack_rows[0].get("mode")
        results["attack_search"] = attack_rows[0].get("search")
        results["attack_space"] = attack_rows[0].get("space")
        results["attack_future_source"] = attack_rows[0].get("future_source")
        results["attack_future_video_skip_first"] = attack_rows[0].get("future_video_skip_first")
        results["attack_future_video_height"] = attack_rows[0].get("future_video_height")
        results["attack_future_video_width"] = attack_rows[0].get("future_video_width")
        results["attack_future_video_max_frames"] = attack_rows[0].get("future_video_max_frames")
        results["attack_replan_count"] = int(len(attack_rows))
        for key in (
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
        ):
            values = np.asarray(
                [float(row[key]) for row in attack_rows if row.get(key) is not None],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            results[f"attack_{key}_mean"] = float(values.mean())
            results[f"attack_{key}_p95"] = float(np.percentile(values, 95))
            results[f"attack_{key}_max"] = float(values.max())
        results["attack_used_future_term_rate"] = float(
            np.mean([float(bool(row.get("used_future_term", False))) for row in attack_rows])
        )
    else:
        results["attack_enabled"] = False
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
