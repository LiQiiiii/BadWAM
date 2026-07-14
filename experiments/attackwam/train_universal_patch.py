from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from attackwam.attacks import AttackConfig, DesynchronizationObjective, QueryOutput, _patch_slice, save_attack_patch
from experiments.libero.eval_libero_single import (
    _mixed_precision_to_model_dtype,
    _model_infer,
    _query_output_from_pred,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _synchronize_if_cuda,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _load_attack_examples(patterns: str, max_examples: int | None = None) -> list[dict[str, Any]]:
    files: list[str] = []
    for pattern in str(patterns).split(","):
        files.extend(glob.glob(os.path.expanduser(os.path.expandvars(pattern.strip()))))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No attack raw NPZ files matched: {patterns}")

    examples: list[dict[str, Any]] = []
    for path in files:
        payload = np.load(path, allow_pickle=True)
        if "clean_images" not in payload:
            continue
        images = payload["clean_images"]
        proprios = payload["proprios"] if "proprios" in payload else None
        prompts = payload["prompts_json"] if "prompts_json" in payload else None
        n = int(images.shape[0])
        for idx in range(n):
            image = images[idx]
            if image.ndim == 4 and image.shape[0] == 1:
                image = image[0]
            proprio = None
            if proprios is not None:
                proprio = proprios[idx]
                if proprio.ndim == 2 and proprio.shape[0] == 1:
                    proprio = proprio[0]
            prompt = None
            if prompts is not None:
                prompt = json.loads(str(prompts[idx]))
            examples.append(
                {
                    "image_u8": image.astype(np.uint8),
                    "proprio": None if proprio is None else proprio.astype(np.float32),
                    "prompt": prompt,
                    "source": path,
                }
            )
            if max_examples is not None and len(examples) >= max_examples:
                return examples
    if not examples:
        raise ValueError(
            "Matched files did not contain clean_images. Collect data with "
            "EVALUATION.attack.enabled=true EVALUATION.attack.save_images=true."
        )
    return examples


def _image_u8_to_tensor(image_u8: np.ndarray, device: str, dtype: torch.dtype) -> torch.Tensor:
    if image_u8.ndim != 3:
        raise ValueError(f"Expected image [C,H,W], got {image_u8.shape}")
    image = torch.as_tensor(image_u8, device=device, dtype=dtype).unsqueeze(0)
    return image * (2.0 / 255.0) - 1.0


def _make_patch(cfg: AttackConfig, image: torch.Tensor) -> torch.Tensor:
    ph = int(cfg.patch_h or cfg.patch_size)
    pw = int(cfg.patch_w or cfg.patch_size)
    ph = max(1, min(ph, int(image.shape[-2])))
    pw = max(1, min(pw, int(image.shape[-1])))
    patch = torch.empty((1, image.shape[1], ph, pw), device=image.device, dtype=image.dtype)
    if cfg.patch_mode == "replace":
        patch.uniform_(-1.0, 1.0)
    else:
        patch.uniform_(-float(cfg.epsilon), float(cfg.epsilon))
    return patch


def _apply_patch(image: torch.Tensor, patch: torch.Tensor, cfg: AttackConfig) -> tuple[torch.Tensor, torch.Tensor]:
    ys, xs = _patch_slice(tuple(image.shape), int(patch.shape[-2]), int(patch.shape[-1]), cfg.patch_location, cfg.camera)
    mask = torch.zeros_like(image)
    mask[:, :, ys, xs] = 1.0
    adv = image.clone()
    if cfg.patch_mode == "replace":
        adv[:, :, ys, xs] = patch
    else:
        adv[:, :, ys, xs] = torch.clamp(adv[:, :, ys, xs] + patch, -1.0, 1.0)
    return adv, (adv - image) * mask


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig) -> None:
    attack_cfg = AttackConfig.from_cfg(cfg.EVALUATION)
    if not attack_cfg.enabled:
        raise ValueError("Set EVALUATION.attack.enabled=true for universal patch training.")
    train_glob = _cfg_get(cfg.EVALUATION.attack, "train_input_glob", None)
    if train_glob is None:
        raise ValueError("Set EVALUATION.attack.train_input_glob=/path/to/attack_raw/*.npz")
    output_path = _cfg_get(
        cfg.EVALUATION.attack,
        "train_output_path",
        str(Path(cfg.EVALUATION.output_dir) / "universal_patch.npz"),
    )
    steps = int(_cfg_get(cfg.EVALUATION.attack, "train_steps", 100))
    batch_size = int(_cfg_get(cfg.EVALUATION.attack, "train_batch_size", 4))
    max_examples_cfg = _cfg_get(cfg.EVALUATION.attack, "train_max_examples", None)
    max_examples = None if max_examples_cfg is None else int(max_examples_cfg)

    device = _resolve_eval_device(cfg)
    dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    model.load_checkpoint(str(cfg.ckpt))
    model = model.to(device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    examples = _load_attack_examples(str(train_glob), max_examples=max_examples)
    rng = np.random.default_rng(int(attack_cfg.seed))
    first_image = _image_u8_to_tensor(examples[0]["image_u8"], device, dtype)
    patch = _make_patch(attack_cfg, first_image)
    objective = DesynchronizationObjective(attack_cfg)
    clean_cache: dict[int, QueryOutput] = {}

    num_inference_steps = int(
        cfg.EVALUATION.get("num_inference_steps")
        if cfg.EVALUATION.get("num_inference_steps") is not None
        else cfg.get("eval_num_inference_steps", 20)
    )
    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = int(action_horizon_cfg) if action_horizon_cfg is not None else int(cfg.data.train.num_frames) - 1
    num_video_frames = (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1

    def query(idx: int, image: torch.Tensor) -> QueryOutput:
        ex = examples[idx]
        prompt = ex["prompt"] or DEFAULT_PROMPT.format(
            task=str(_cfg_get(cfg.EVALUATION.attack, "train_task_description", "perform the task"))
        )
        proprio = ex["proprio"]
        proprio_tensor = None
        if proprio is not None:
            proprio_tensor = torch.as_tensor(proprio, device=device, dtype=torch.float32)
            if proprio_tensor.ndim == 1:
                proprio_tensor = proprio_tensor.unsqueeze(0)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image,
            "action_horizon": action_horizon,
            "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
            "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
            "num_inference_steps": num_inference_steps,
            "proprio": proprio_tensor,
            "sigma_shift": None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.get("sigma_shift")),
            "seed": None if cfg.get("seed") is None else int(cfg.seed),
            "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
            "tiled": bool(cfg.EVALUATION.get("tiled", False)),
            "num_video_frames": num_video_frames,
        }
        pred = _model_infer(
            model,
            infer_kwargs,
            input_image=image,
            return_world_latents=bool(attack_cfg.require_future),
            decode_video=False,
        )
        return _query_output_from_pred(pred)

    def objective_for_patch(candidate_patch: torch.Tensor, batch_indices: np.ndarray) -> torch.Tensor:
        scores = []
        with torch.no_grad():
            for idx in batch_indices:
                idx_int = int(idx)
                image = _image_u8_to_tensor(examples[idx_int]["image_u8"], device, dtype)
                if idx_int not in clean_cache:
                    clean_cache[idx_int] = query(idx_int, image)
                adv_image, perturb = _apply_patch(image, candidate_patch, attack_cfg)
                adv = query(idx_int, adv_image)
                comps = objective.components(clean_cache[idx_int], adv, perturb)
                scores.append(comps["score"])
        return torch.stack(scores).mean()

    start = time.perf_counter()
    history: list[dict[str, Any]] = []
    for step in range(steps):
        batch = rng.choice(len(examples), size=min(batch_size, len(examples)), replace=len(examples) < batch_size)
        direction = torch.empty_like(patch)
        direction.bernoulli_(0.5).mul_(2.0).sub_(1.0)
        plus = patch + float(attack_cfg.sigma) * direction
        minus = patch - float(attack_cfg.sigma) * direction
        if attack_cfg.patch_mode == "replace":
            plus = torch.clamp(plus, -1.0, 1.0)
            minus = torch.clamp(minus, -1.0, 1.0)
        else:
            plus = torch.clamp(plus, -float(attack_cfg.epsilon), float(attack_cfg.epsilon))
            minus = torch.clamp(minus, -float(attack_cfg.epsilon), float(attack_cfg.epsilon))
        _synchronize_if_cuda(device)
        score_plus = objective_for_patch(plus, batch)
        score_minus = objective_for_patch(minus, batch)
        _synchronize_if_cuda(device)
        grad_est = ((score_plus - score_minus) / max(2.0 * float(attack_cfg.sigma), 1e-8)) * direction
        patch = patch + float(attack_cfg.step_size) * torch.sign(grad_est)
        if attack_cfg.patch_mode == "replace":
            patch = torch.clamp(patch, -1.0, 1.0)
        else:
            patch = torch.clamp(patch, -float(attack_cfg.epsilon), float(attack_cfg.epsilon))
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            row = {
                "step": int(step),
                "score_plus": float(score_plus.detach().cpu().item()),
                "score_minus": float(score_minus.detach().cpu().item()),
                "elapsed_s": float(time.perf_counter() - start),
            }
            history.append(row)
            print(row)

    metadata = {
        "mode": attack_cfg.mode,
        "objective": attack_cfg.objective,
        "search": "universal_query_search",
        "num_examples": len(examples),
        "steps": steps,
        "batch_size": batch_size,
        "history": history,
        "dataset_stats_path": str(dataset_stats_path),
        "ckpt": str(cfg.ckpt),
    }
    save_attack_patch(output_path, patch, metadata=metadata)
    print(f"Saved universal patch to: {output_path}")


if __name__ == "__main__":
    main()
