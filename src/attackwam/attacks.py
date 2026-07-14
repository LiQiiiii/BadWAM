from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch


TensorQueryFn = Callable[[torch.Tensor, bool], "QueryOutput"]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float()


def _tensor_l2_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = _to_float_tensor(a)
    b = _to_float_tensor(b)
    diff = a - b
    if diff.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=diff.device)
    if diff.ndim >= 2:
        diff = diff.reshape(diff.shape[0], -1)
        return torch.linalg.vector_norm(diff, dim=-1).mean()
    return torch.linalg.vector_norm(diff.reshape(-1), dim=0)


def _tensor_l1_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (_to_float_tensor(a) - _to_float_tensor(b)).abs().mean()


def _metric_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float("nan")
        return float(np.asarray(value, dtype=np.float32).mean())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _components_to_metrics(components: dict[str, Any]) -> dict[str, float]:
    return {
        key: _metric_float(value)
        for key, value in components.items()
        if key
        in {
            "score",
            "action_distance",
            "future_distance",
            "desynchronization_score",
            "perturb_linf",
            "perturb_l1",
        }
    }


def _action_delta_stats(clean_action: Optional[torch.Tensor], adv_action: Optional[torch.Tensor]) -> dict[str, Any]:
    if clean_action is None or adv_action is None:
        return {}
    clean = _to_float_tensor(clean_action)
    adv = _to_float_tensor(adv_action)
    if clean.shape != adv.shape or clean.numel() == 0:
        return {}
    diff = adv - clean
    if diff.ndim < 2:
        return {}

    d = diff.reshape(-1, diff.shape[-2], diff.shape[-1])  # [B,T,D]
    abs_by_dim = d.abs().mean(dim=(0, 1))
    l2_by_timestep = torch.linalg.vector_norm(d, dim=-1).mean(dim=0)
    horizon = int(d.shape[1])
    third = max(1, int(math.ceil(horizon / 3.0)))
    early = l2_by_timestep[:third]
    middle = l2_by_timestep[third : min(2 * third, horizon)]
    late = l2_by_timestep[min(2 * third, horizon) :]

    def mean_or_nan(x: torch.Tensor) -> float:
        if x.numel() == 0:
            return float("nan")
        return float(x.mean().detach().cpu().item())

    stats: dict[str, Any] = {
        "action_delta_abs_by_dim": _jsonable(abs_by_dim),
        "action_delta_l2_by_timestep": _jsonable(l2_by_timestep),
        "action_delta_horizon_early_l2": mean_or_nan(early),
        "action_delta_horizon_middle_l2": mean_or_nan(middle),
        "action_delta_horizon_late_l2": mean_or_nan(late),
        "action_delta_horizon_argmax": int(torch.argmax(l2_by_timestep).detach().cpu().item()),
    }
    dim = int(d.shape[-1])
    if dim >= 3:
        stats["action_delta_xyz_abs_mean"] = float(abs_by_dim[:3].mean().detach().cpu().item())
    if dim >= 6:
        stats["action_delta_rot_abs_mean"] = float(abs_by_dim[3:6].mean().detach().cpu().item())
    if dim >= 7:
        stats["action_delta_gripper_abs_mean"] = float(abs_by_dim[6].detach().cpu().item())
    return stats


def _world_delta_stats(clean_world: Optional[torch.Tensor], adv_world: Optional[torch.Tensor]) -> dict[str, Any]:
    if clean_world is None or adv_world is None:
        return {}
    clean = _to_float_tensor(clean_world)
    adv = _to_float_tensor(adv_world)
    if clean.shape != adv.shape or clean.numel() == 0:
        return {}
    diff = adv - clean
    flat = diff.reshape(diff.shape[0], -1) if diff.ndim >= 2 else diff.reshape(1, -1)
    l2_by_frame = torch.linalg.vector_norm(flat, dim=-1)
    l1_by_frame = flat.abs().mean(dim=-1)
    return {
        "future_delta_shape": list(diff.shape),
        "future_delta_l2_by_frame": _jsonable(l2_by_frame),
        "future_delta_l1_by_frame": _jsonable(l1_by_frame),
        "future_delta_l2_max_frame": int(torch.argmax(l2_by_frame).detach().cpu().item()),
        "future_delta_l1_mean": float(diff.abs().mean().detach().cpu().item()),
        "future_delta_linf": float(diff.abs().max().detach().cpu().item()),
    }


def _input_image_delta_stats(clean_image: Optional[torch.Tensor], adv_image: Optional[torch.Tensor]) -> dict[str, Any]:
    """Per-replan input perceptibility metrics on normalized image tensors.

    FastWAM image inputs are normalized to roughly [-1, 1].  We report distances
    after mapping back to [0, 1], which makes the numbers directly comparable to
    standard image metrics.  The SSIM value is a lightweight global SSIM proxy
    computed over the full image tensor; the matched-strength analyzer recomputes
    standard skimage SSIM from saved raw images when available.
    """
    if clean_image is None or adv_image is None:
        return {}
    clean = _to_float_tensor(clean_image)
    adv = _to_float_tensor(adv_image)
    if clean.shape != adv.shape or clean.numel() == 0:
        return {}

    clean01 = ((clean.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
    adv01 = ((adv.clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
    diff = adv01 - clean01
    flat = diff.reshape(diff.shape[0], -1) if diff.ndim >= 2 else diff.reshape(1, -1)
    mse = diff.square().mean()
    input_l2_by_image = torch.linalg.vector_norm(flat, dim=-1)

    # Global SSIM proxy.  This is intentionally cheap because it is computed
    # during closed-loop evaluation at every attacked replan.
    c1 = 0.01**2
    c2 = 0.03**2
    reduce_dims = tuple(range(1, clean01.ndim))
    mu_x = clean01.mean(dim=reduce_dims)
    mu_y = adv01.mean(dim=reduce_dims)
    var_x = ((clean01 - mu_x.reshape(-1, *([1] * (clean01.ndim - 1)))) ** 2).mean(dim=reduce_dims)
    var_y = ((adv01 - mu_y.reshape(-1, *([1] * (adv01.ndim - 1)))) ** 2).mean(dim=reduce_dims)
    cov_xy = (
        (clean01 - mu_x.reshape(-1, *([1] * (clean01.ndim - 1))))
        * (adv01 - mu_y.reshape(-1, *([1] * (adv01.ndim - 1))))
    ).mean(dim=reduce_dims)
    ssim_global = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2) + 1e-12
    )

    return {
        "input_delta_l2": float(input_l2_by_image.mean().detach().cpu().item()),
        "input_delta_l2_by_image": _jsonable(input_l2_by_image),
        "input_delta_rmse": float(torch.sqrt(mse + 1e-12).detach().cpu().item()),
        "input_delta_l1": float(diff.abs().mean().detach().cpu().item()),
        "input_delta_linf": float(diff.abs().max().detach().cpu().item()),
        "input_delta_psnr": float((-10.0 * torch.log10(mse + 1e-12)).detach().cpu().item()),
        "input_delta_ssim_global": float(ssim_global.mean().detach().cpu().item()),
    }


def _perturb_camera_stats(perturb: Optional[torch.Tensor]) -> dict[str, Any]:
    if perturb is None or perturb.numel() == 0 or perturb.ndim < 4:
        return {}
    p = perturb.detach().float().abs()
    width = int(p.shape[-1])
    left = p[..., : width // 2]
    right = p[..., width // 2 :]
    return {
        "perturb_left_l1": float(left.mean().cpu().item()) if left.numel() else float("nan"),
        "perturb_right_l1": float(right.mean().cpu().item()) if right.numel() else float("nan"),
        "perturb_left_linf": float(left.max().cpu().item()) if left.numel() else float("nan"),
        "perturb_right_linf": float(right.max().cpu().item()) if right.numel() else float("nan"),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


@dataclass
class QueryOutput:
    """Output visible to the attacker for a single WAM query.

    `action` should be the normalized WAM action tensor, usually [T, D].
    `world` can be decoded future features, predicted video latents, or any
    model-returned future representation available under the chosen threat model.
    `raw` is carried through so eval code can reuse the best WAM prediction.
    """

    action: torch.Tensor
    world: Optional[torch.Tensor] = None
    raw: Optional[dict[str, Any]] = None


@dataclass
class AttackResult:
    image: torch.Tensor
    stats: dict[str, Any]
    best_output: Optional[QueryOutput] = None
    clean_output: Optional[QueryOutput] = None
    patch: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


@dataclass
class AttackConfig:
    enabled: bool = False
    name: str = "clean"
    mode: str = "action_only"
    objective: str = "desynchronization"
    search: str = "query"
    space: str = "patch"
    patch_mode: str = "additive"
    patch_size: int = 32
    patch_h: Optional[int] = None
    patch_w: Optional[int] = None
    patch_location: str = "bottom_right"
    camera: str = "both"
    epsilon: float = 0.06
    step_size: float = 0.02
    sigma: float = 0.02
    budget: int = 16
    restarts: int = 1
    action_weight: float = 1.0
    future_weight: float = 1.0
    perturb_weight: float = 0.05
    future_source: str = "latents"
    future_video_skip_first: bool = True
    future_video_height: int = 64
    future_video_width: int = 128
    future_video_max_frames: int = 0
    save_world: bool = False
    save_trajectory: bool = True
    save_representatives: bool = True
    representative_keep: int = 20
    representative_save_images: bool = True
    representative_save_world: bool = True
    targeted: bool = False
    target_action_scale: float = -1.0
    require_future: bool = True
    seed: int = 0
    patch_path: Optional[str] = None
    save_raw: bool = True
    save_images: bool = False
    save_patch: bool = True

    @classmethod
    def from_cfg(cls, cfg: Any) -> "AttackConfig":
        attack_cfg = _cfg_get(cfg, "attack", cfg)
        if attack_cfg is None:
            return cls(enabled=False)
        mode = str(_cfg_get(attack_cfg, "mode", _cfg_get(attack_cfg, "type", "action_only")))
        return cls(
            enabled=_as_bool(_cfg_get(attack_cfg, "enabled", False)),
            name=str(_cfg_get(attack_cfg, "name", mode)),
            mode=mode,
            objective=str(_cfg_get(attack_cfg, "objective", "desynchronization")),
            search=str(_cfg_get(attack_cfg, "search", "query")),
            space=str(_cfg_get(attack_cfg, "space", "patch")),
            patch_mode=str(_cfg_get(attack_cfg, "patch_mode", "additive")),
            patch_size=int(_cfg_get(attack_cfg, "patch_size", 32)),
            patch_h=(
                None
                if _cfg_get(attack_cfg, "patch_h", None) is None
                else int(_cfg_get(attack_cfg, "patch_h"))
            ),
            patch_w=(
                None
                if _cfg_get(attack_cfg, "patch_w", None) is None
                else int(_cfg_get(attack_cfg, "patch_w"))
            ),
            patch_location=str(_cfg_get(attack_cfg, "patch_location", "bottom_right")),
            camera=str(_cfg_get(attack_cfg, "camera", "both")),
            epsilon=float(_cfg_get(attack_cfg, "epsilon", 0.06)),
            step_size=float(_cfg_get(attack_cfg, "step_size", 0.02)),
            sigma=float(_cfg_get(attack_cfg, "sigma", 0.02)),
            budget=int(_cfg_get(attack_cfg, "budget", 16)),
            restarts=max(1, int(_cfg_get(attack_cfg, "restarts", 1))),
            action_weight=float(_cfg_get(attack_cfg, "action_weight", 1.0)),
            future_weight=float(_cfg_get(attack_cfg, "future_weight", 1.0)),
            perturb_weight=float(_cfg_get(attack_cfg, "perturb_weight", 0.05)),
            future_source=str(_cfg_get(attack_cfg, "future_source", "latents")),
            future_video_skip_first=_as_bool(_cfg_get(attack_cfg, "future_video_skip_first", True)),
            future_video_height=int(_cfg_get(attack_cfg, "future_video_height", 64)),
            future_video_width=int(_cfg_get(attack_cfg, "future_video_width", 128)),
            future_video_max_frames=int(_cfg_get(attack_cfg, "future_video_max_frames", 0)),
            save_world=_as_bool(_cfg_get(attack_cfg, "save_world", False)),
            save_trajectory=_as_bool(_cfg_get(attack_cfg, "save_trajectory", True)),
            save_representatives=_as_bool(_cfg_get(attack_cfg, "save_representatives", True)),
            representative_keep=int(_cfg_get(attack_cfg, "representative_keep", 20)),
            representative_save_images=_as_bool(_cfg_get(attack_cfg, "representative_save_images", True)),
            representative_save_world=_as_bool(_cfg_get(attack_cfg, "representative_save_world", True)),
            targeted=_as_bool(_cfg_get(attack_cfg, "targeted", False)),
            target_action_scale=float(_cfg_get(attack_cfg, "target_action_scale", -1.0)),
            require_future=_as_bool(_cfg_get(attack_cfg, "require_future", True)),
            seed=int(_cfg_get(attack_cfg, "seed", 0)),
            patch_path=(
                None
                if _cfg_get(attack_cfg, "patch_path", None) in {None, ""}
                else str(_cfg_get(attack_cfg, "patch_path"))
            ),
            save_raw=_as_bool(_cfg_get(attack_cfg, "save_raw", True)),
            save_images=_as_bool(_cfg_get(attack_cfg, "save_images", False)),
            save_patch=_as_bool(_cfg_get(attack_cfg, "save_patch", True)),
        )

    @property
    def is_universal(self) -> bool:
        return self.mode == "universal_patch" or (
            self.patch_path is not None and self.search == "load"
        )

    @property
    def uses_video_future(self) -> bool:
        return self.future_source.strip().lower() in {
            "video",
            "decoded_video",
            "future_video",
            "predicted_video",
        }


class DesynchronizationObjective:
    def __init__(self, cfg: AttackConfig):
        self.cfg = cfg

    def target_action(self, clean_action: torch.Tensor) -> torch.Tensor:
        return clean_action * float(self.cfg.target_action_scale)

    def components(
        self,
        clean: QueryOutput,
        adv: QueryOutput,
        perturb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.cfg.targeted:
            action_distance = _tensor_l2_mean(adv.action, self.target_action(clean.action))
            action_score = -action_distance
        else:
            action_distance = _tensor_l2_mean(adv.action, clean.action)
            action_score = action_distance

        if clean.world is not None and adv.world is not None:
            future_distance = _tensor_l2_mean(adv.world, clean.world)
        else:
            future_distance = torch.zeros((), device=adv.action.device, dtype=torch.float32)

        target_device = perturb.device
        action_distance = action_distance.to(device=target_device)
        action_score = action_score.to(device=target_device)
        future_distance = future_distance.to(device=target_device)

        perturb_linf = perturb.detach().float().abs().max()
        perturb_l1 = perturb.detach().float().abs().mean()
        score = (
            float(self.cfg.action_weight) * action_score
            - float(self.cfg.future_weight) * future_distance
            - float(self.cfg.perturb_weight) * perturb_l1
        )
        desynchronization_score = action_distance / (future_distance + 1e-6)
        return {
            "score": score,
            "action_distance": action_distance,
            "future_distance": future_distance,
            "desynchronization_score": desynchronization_score,
            "perturb_linf": perturb_linf,
            "perturb_l1": perturb_l1,
        }


class BaseAttack:
    def __init__(self, cfg: AttackConfig):
        self.cfg = cfg
        self.objective = DesynchronizationObjective(cfg)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(cfg.seed))

    def optimize(self, image: torch.Tensor, query_fn: TensorQueryFn) -> AttackResult:
        raise NotImplementedError

    def _stats_header(self, start_time: float, query_count: int, best_components: dict[str, Any]) -> dict[str, Any]:
        stats = {
            "enabled": True,
            "name": self.cfg.name,
            "mode": self.cfg.mode,
            "objective": self.cfg.objective,
            "search": self.cfg.search,
            "space": self.cfg.space,
            "future_source": self.cfg.future_source,
            "future_video_skip_first": bool(self.cfg.future_video_skip_first),
            "future_video_height": int(self.cfg.future_video_height),
            "future_video_width": int(self.cfg.future_video_width),
            "future_video_max_frames": int(self.cfg.future_video_max_frames),
            "patch_mode": self.cfg.patch_mode,
            "patch_location": self.cfg.patch_location,
            "camera": self.cfg.camera,
            "budget": int(self.cfg.budget),
            "query_count": int(query_count),
            "attack_time_s": float(time.perf_counter() - start_time),
        }
        stats.update({k: _jsonable(v) for k, v in best_components.items()})
        return stats

    def _add_explanation_stats(
        self,
        stats: dict[str, Any],
        *,
        clean: Optional[QueryOutput],
        adv: Optional[QueryOutput],
        perturb: Optional[torch.Tensor],
        clean_image: Optional[torch.Tensor] = None,
        adv_image: Optional[torch.Tensor] = None,
    ) -> None:
        if clean is not None and adv is not None:
            stats.update(_action_delta_stats(clean.action, adv.action))
            stats.update(_world_delta_stats(clean.world, adv.world))
        stats.update(_input_image_delta_stats(clean_image, adv_image))
        stats.update(_perturb_camera_stats(perturb))


def _camera_bounds(width: int, camera: str) -> tuple[int, int]:
    camera = camera.lower()
    if camera in {"left", "agent", "agentview", "primary"}:
        return 0, width // 2
    if camera in {"right", "wrist", "eye_in_hand"}:
        return width // 2, width
    return 0, width


def _patch_slice(
    image_shape: tuple[int, ...],
    patch_h: int,
    patch_w: int,
    location: str,
    camera: str,
) -> tuple[slice, slice]:
    _, _, height, width = image_shape
    c0, c1 = _camera_bounds(width, camera)
    span_w = max(c1 - c0, patch_w)
    location = location.lower()
    if location in {"top_left", "tl"}:
        y0, x0 = 0, c0
    elif location in {"top_right", "tr"}:
        y0, x0 = 0, c0 + span_w - patch_w
    elif location in {"bottom_left", "bl"}:
        y0, x0 = height - patch_h, c0
    elif location in {"center", "middle"}:
        y0, x0 = (height - patch_h) // 2, c0 + (span_w - patch_w) // 2
    else:
        y0, x0 = height - patch_h, c0 + span_w - patch_w
    y0 = int(max(0, min(height - patch_h, y0)))
    x0 = int(max(0, min(width - patch_w, x0)))
    return slice(y0, y0 + patch_h), slice(x0, x0 + patch_w)


class PatchMixin:
    cfg: AttackConfig

    def _patch_shape(self, image: torch.Tensor) -> tuple[int, int]:
        ph = int(self.cfg.patch_h or self.cfg.patch_size)
        pw = int(self.cfg.patch_w or self.cfg.patch_size)
        ph = max(1, min(ph, int(image.shape[-2])))
        pw = max(1, min(pw, int(image.shape[-1])))
        return ph, pw

    def _new_patch(self, image: torch.Tensor) -> torch.Tensor:
        ph, pw = self._patch_shape(image)
        if self.cfg.patch_mode == "replace":
            patch = torch.empty((1, image.shape[1], ph, pw), device=image.device, dtype=image.dtype)
            patch.uniform_(-1.0, 1.0)
            return patch
        patch = torch.empty((1, image.shape[1], ph, pw), device=image.device, dtype=image.dtype)
        patch.uniform_(-float(self.cfg.epsilon), float(self.cfg.epsilon))
        return patch

    def _apply_patch(self, image: torch.Tensor, patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adv = image.clone()
        mask = torch.zeros_like(image)
        ys, xs = _patch_slice(
            tuple(image.shape),
            int(patch.shape[-2]),
            int(patch.shape[-1]),
            self.cfg.patch_location,
            self.cfg.camera,
        )
        mask[:, :, ys, xs] = 1.0
        if self.cfg.patch_mode == "replace":
            adv[:, :, ys, xs] = patch
        else:
            adv[:, :, ys, xs] = torch.clamp(adv[:, :, ys, xs] + patch, -1.0, 1.0)
        return adv, mask


class QuerySearchAttack(BaseAttack, PatchMixin):
    """Zeroth-order optimizer for BadWAM.

    The same query-based optimizer supports both public attack objectives:
    action-only desynchronization and imagination-preserving desynchronization.
    The query function may expose only actions or additionally expose WAM
    imagination outputs such as future latents or decoded future videos.
    """

    def optimize(self, image: torch.Tensor, query_fn: TensorQueryFn) -> AttackResult:
        start_time = time.perf_counter()
        require_future = bool(self.cfg.require_future)
        clean = query_fn(image, require_future)
        query_count = 1

        if self.cfg.space == "full_linf":
            variable = torch.zeros_like(image)
            variable.uniform_(-float(self.cfg.epsilon), float(self.cfg.epsilon))
        else:
            variable = self._new_patch(image)

        best_image = image
        best_output = clean
        best_patch = variable.detach().clone() if self.cfg.space != "full_linf" else None
        best_mask = torch.zeros_like(image)
        best_components: dict[str, torch.Tensor] = {
            "score": torch.tensor(float("-inf"), device=image.device),
            "action_distance": torch.tensor(0.0, device=image.device),
            "future_distance": torch.tensor(0.0, device=image.device),
            "desynchronization_score": torch.tensor(0.0, device=image.device),
            "perturb_linf": torch.tensor(0.0, device=image.device),
            "perturb_l1": torch.tensor(0.0, device=image.device),
        }
        trajectory: list[dict[str, Any]] = []

        def apply_variable(var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if self.cfg.space == "full_linf":
                delta = torch.clamp(var, -float(self.cfg.epsilon), float(self.cfg.epsilon))
                return torch.clamp(image + delta, -1.0, 1.0), delta
            adv, mask = self._apply_patch(image, var)
            perturb = (adv - image) * mask
            return adv, perturb

        def evaluate(var: torch.Tensor) -> tuple[torch.Tensor, QueryOutput, dict[str, torch.Tensor], torch.Tensor]:
            adv_img, perturb = apply_variable(var)
            out = query_fn(adv_img, require_future)
            comps = self.objective.components(clean, out, perturb)
            return comps["score"], out, comps, adv_img

        for iteration in range(max(1, int(self.cfg.budget // 2))):
            direction = torch.empty_like(variable)
            direction.bernoulli_(0.5).mul_(2.0).sub_(1.0)
            var_plus = variable + float(self.cfg.sigma) * direction
            var_minus = variable - float(self.cfg.sigma) * direction
            if self.cfg.patch_mode == "replace" and self.cfg.space != "full_linf":
                var_plus = torch.clamp(var_plus, -1.0, 1.0)
                var_minus = torch.clamp(var_minus, -1.0, 1.0)
            else:
                var_plus = torch.clamp(var_plus, -float(self.cfg.epsilon), float(self.cfg.epsilon))
                var_minus = torch.clamp(var_minus, -float(self.cfg.epsilon), float(self.cfg.epsilon))

            score_plus, out_plus, comps_plus, img_plus = evaluate(var_plus)
            score_minus, out_minus, comps_minus, img_minus = evaluate(var_minus)
            query_count += 2
            best_before = _components_to_metrics(best_components)

            if float(score_plus.detach().cpu()) > float(best_components["score"].detach().cpu()):
                best_components = comps_plus
                best_output = out_plus
                best_image = img_plus
                best_patch = var_plus.detach().clone() if self.cfg.space != "full_linf" else None
                best_mask = (img_plus != image).to(image.dtype)
            if float(score_minus.detach().cpu()) > float(best_components["score"].detach().cpu()):
                best_components = comps_minus
                best_output = out_minus
                best_image = img_minus
                best_patch = var_minus.detach().clone() if self.cfg.space != "full_linf" else None
                best_mask = (img_minus != image).to(image.dtype)

            if self.cfg.save_trajectory:
                row = {
                    "iteration": int(iteration),
                    "query_count": int(query_count),
                    "score_plus": _metric_float(score_plus),
                    "score_minus": _metric_float(score_minus),
                    "grad_scale": _metric_float(
                        (score_plus - score_minus) / max(2.0 * float(self.cfg.sigma), 1e-8)
                    ),
                    "best_score_before": best_before.get("score"),
                }
                for prefix, comps in (
                    ("plus", comps_plus),
                    ("minus", comps_minus),
                    ("best", best_components),
                ):
                    for key, value in _components_to_metrics(comps).items():
                        row[f"{prefix}_{key}"] = value
                trajectory.append(row)

            grad_est = ((score_plus - score_minus) / max(2.0 * float(self.cfg.sigma), 1e-8)) * direction
            variable = variable + float(self.cfg.step_size) * torch.sign(grad_est)
            if self.cfg.patch_mode == "replace" and self.cfg.space != "full_linf":
                variable = torch.clamp(variable, -1.0, 1.0)
            else:
                variable = torch.clamp(variable, -float(self.cfg.epsilon), float(self.cfg.epsilon))

        stats = self._stats_header(start_time, query_count, best_components)
        stats["used_future_term"] = bool(clean.world is not None and best_output.world is not None)
        if self.cfg.save_trajectory:
            stats["trajectory"] = trajectory
            stats["trajectory_length"] = int(len(trajectory))
        self._add_explanation_stats(
            stats,
            clean=clean,
            adv=best_output,
            perturb=(best_image - image),
            clean_image=image,
            adv_image=best_image,
        )
        return AttackResult(
            image=best_image.detach(),
            stats=stats,
            best_output=best_output,
            clean_output=clean,
            patch=best_patch,
            mask=best_mask.detach() if best_mask is not None else None,
        )


class RandomSearchAttack(BaseAttack, PatchMixin):
    def optimize(self, image: torch.Tensor, query_fn: TensorQueryFn) -> AttackResult:
        start_time = time.perf_counter()
        require_future = bool(self.cfg.require_future)
        clean = query_fn(image, require_future)
        query_count = 1
        best_image = image
        best_output = clean
        best_patch = None
        best_mask = torch.zeros_like(image)
        best_components = {"score": torch.tensor(float("-inf"), device=image.device)}
        trajectory: list[dict[str, Any]] = []
        for iteration in range(max(1, int(self.cfg.budget))):
            if self.cfg.space == "full_linf":
                delta = torch.empty_like(image)
                delta.uniform_(-float(self.cfg.epsilon), float(self.cfg.epsilon))
                adv = torch.clamp(image + delta, -1.0, 1.0)
                perturb = delta
                patch = None
                mask = (adv != image).to(image.dtype)
            else:
                patch = self._new_patch(image)
                adv, mask = self._apply_patch(image, patch)
                perturb = (adv - image) * mask
            out = query_fn(adv, require_future)
            query_count += 1
            comps = self.objective.components(clean, out, perturb)
            if float(comps["score"].detach().cpu()) > float(best_components["score"].detach().cpu()):
                best_components = comps
                best_image = adv
                best_output = out
                best_patch = patch
                best_mask = mask
            if self.cfg.save_trajectory:
                row = {
                    "iteration": int(iteration),
                    "query_count": int(query_count),
                }
                for prefix, values in (("candidate", comps), ("best", best_components)):
                    for key, value in _components_to_metrics(values).items():
                        row[f"{prefix}_{key}"] = value
                trajectory.append(row)
        stats = self._stats_header(start_time, query_count, best_components)
        stats["used_future_term"] = bool(clean.world is not None and best_output.world is not None)
        if self.cfg.save_trajectory:
            stats["trajectory"] = trajectory
            stats["trajectory_length"] = int(len(trajectory))
        self._add_explanation_stats(
            stats,
            clean=clean,
            adv=best_output,
            perturb=(best_image - image),
            clean_image=image,
            adv_image=best_image,
        )
        return AttackResult(
            image=best_image.detach(),
            stats=stats,
            best_output=best_output,
            clean_output=clean,
            patch=best_patch.detach() if best_patch is not None else None,
            mask=best_mask.detach(),
        )


class UniversalPatchAttack(BaseAttack, PatchMixin):
    def __init__(self, cfg: AttackConfig):
        super().__init__(cfg)
        if not cfg.patch_path:
            raise ValueError("UniversalPatchAttack requires EVALUATION.attack.patch_path.")
        self.patch_path = Path(os.path.expanduser(os.path.expandvars(cfg.patch_path)))
        if not self.patch_path.exists():
            raise FileNotFoundError(f"Universal patch not found: {self.patch_path}")
        payload = np.load(self.patch_path)
        if "patch" not in payload:
            raise ValueError(f"Universal patch file must contain array 'patch': {self.patch_path}")
        self.patch_np = payload["patch"].astype(np.float32)

    def optimize(self, image: torch.Tensor, query_fn: TensorQueryFn) -> AttackResult:
        start_time = time.perf_counter()
        clean = query_fn(image, bool(self.cfg.require_future))
        patch = torch.as_tensor(self.patch_np, device=image.device, dtype=image.dtype)
        if patch.ndim == 3:
            patch = patch.unsqueeze(0)
        adv, mask = self._apply_patch(image, patch)
        adv_out = query_fn(adv, bool(self.cfg.require_future))
        perturb = (adv - image) * mask
        comps = self.objective.components(clean, adv_out, perturb)
        stats = self._stats_header(start_time, 2, comps)
        stats["patch_path"] = str(self.patch_path)
        stats["used_future_term"] = bool(clean.world is not None and adv_out.world is not None)
        self._add_explanation_stats(
            stats,
            clean=clean,
            adv=adv_out,
            perturb=perturb,
            clean_image=image,
            adv_image=adv,
        )
        return AttackResult(
            image=adv.detach(),
            stats=stats,
            best_output=adv_out,
            clean_output=clean,
            patch=patch.detach(),
            mask=mask.detach(),
        )


def build_attack(cfg: Any) -> Optional[BaseAttack]:
    attack_cfg = AttackConfig.from_cfg(cfg)
    if not attack_cfg.enabled:
        return None
    if attack_cfg.is_universal:
        return UniversalPatchAttack(attack_cfg)
    if attack_cfg.search.lower() in {"random", "random_search"}:
        return RandomSearchAttack(attack_cfg)
    return QuerySearchAttack(attack_cfg)


def save_attack_patch(path: str | Path, patch: torch.Tensor, metadata: Optional[dict[str, Any]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"patch": patch.detach().cpu().float().numpy()}
    if metadata is not None:
        arrays["metadata_json"] = np.asarray(json.dumps(_jsonable(metadata)), dtype=object)
    np.savez_compressed(path, **arrays)
