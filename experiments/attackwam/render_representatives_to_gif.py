#!/usr/bin/env python3
"""Render BadWAM representative samples into GIFs and per-frame PNGs.

Each representative `.npz` stores a compact episode-level archive.  For WAM
variants with predicted futures, this script selects one representative replan
inside each episode, renders:

    clean future | adversarial future | abs diff x N

as a GIF, and also writes every GIF frame as a PNG.  A corresponding
clean/adv/diff observation panel at the selected replan is saved as a PNG.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _load_json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    return json.loads(str(value))


def _load_stats(data: np.lib.npyio.NpzFile) -> list[dict[str, Any]]:
    if "stats_json" not in data.files:
        return []
    stats: list[dict[str, Any]] = []
    for item in data["stats_json"]:
        try:
            stats.append(json.loads(str(item)))
        except json.JSONDecodeError:
            stats.append({})
    return stats


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _choose_replan(stats: list[dict[str, Any]], n_replans: int) -> int:
    """Choose the most visually/analytically useful replan within an episode."""
    if n_replans <= 0:
        return 0
    if not stats:
        return min(n_replans - 1, 0)

    best_idx = 0
    best_score = float("-inf")
    for idx, row in enumerate(stats[:n_replans]):
        desynchronization = _to_float(row.get("desynchronization_score"), 0.0)
        action = _to_float(row.get("action_distance"), 0.0)
        future = _to_float(row.get("future_delta_l1_mean", row.get("future_distance")), 0.0)
        perturb = _to_float(row.get("perturb_l1"), 0.0)
        # Prefer high action/future desynchronization, while avoiding purely noisy frames.
        score = desynchronization + 0.25 * action - 0.02 * future - 0.01 * perturb
        if score > best_score:
            best_score = score
            best_idx = idx
    return int(min(max(best_idx, 0), n_replans - 1))


def _chw_to_hwc(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected image-like array, got shape {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    arr = _chw_to_hwc(frame)
    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        max_val = float(np.max(finite)) if finite.size else 1.0
        if max_val <= 1.5:
            arr = arr * 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _resize(img: Image.Image, scale: float) -> Image.Image:
    if abs(scale - 1.0) < 1e-6:
        return img
    w, h = img.size
    return img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.BICUBIC)


def _font() -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    ):
        try:
            return ImageFont.truetype(path, 14)
        except OSError:
            pass
    return ImageFont.load_default()


def _make_panel(
    clean: np.ndarray,
    adv: np.ndarray,
    *,
    diff_scale: float,
    scale: float,
    labels: tuple[str, str, str],
) -> Image.Image:
    clean_u8 = _to_uint8(clean)
    adv_u8 = _to_uint8(adv)
    diff = np.clip(np.abs(clean_u8.astype(np.int16) - adv_u8.astype(np.int16)) * diff_scale, 0, 255).astype(np.uint8)

    imgs = [_resize(Image.fromarray(x), scale) for x in (clean_u8, adv_u8, diff)]
    w = max(img.width for img in imgs)
    h = max(img.height for img in imgs)
    label_h = 24
    pad = 4
    canvas = Image.new("RGB", (3 * w + 2 * pad, h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font()

    for i, (img, label) in enumerate(zip(imgs, labels)):
        x = i * (w + pad)
        canvas.paste(img, (x, label_h))
        draw.text((x + 3, 4), label, fill=(20, 20, 20), font=font)
    return canvas


def _resolve_sample_path(manifest_path: Path, item: dict[str, Any]) -> Path | None:
    raw = item.get("path") or item.get("sample_path")
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else (manifest_path.parent / p))
    sample_id = item.get("sample_id")
    if sample_id:
        candidates.append(manifest_path.parent / "samples" / f"{sample_id}.npz")
    for path in candidates:
        if path.exists():
            return path
    return None


def render_sample(
    sample_path: Path,
    *,
    out_dir: Path,
    diff_scale: float,
    world_scale: float,
    obs_scale: float,
    duration_ms: int,
    overwrite: bool,
) -> dict[str, Any]:
    data = np.load(sample_path, allow_pickle=True)
    metadata = _load_json_scalar(data["metadata_json"]) if "metadata_json" in data.files else {}
    sample_id = str(metadata.get("sample_id") or sample_path.stem)
    safe_id = _safe_name(sample_id)
    stats = _load_stats(data)

    result: dict[str, Any] = {
        "sample_id": sample_id,
        "sample_path": str(sample_path),
        "success": metadata.get("success"),
        "score": metadata.get("score"),
    }

    sample_out = out_dir / safe_id
    sample_out.mkdir(parents=True, exist_ok=True)

    has_world = "clean_worlds" in data.files and "adv_worlds" in data.files
    if has_world:
        clean_worlds = data["clean_worlds"]
        adv_worlds = data["adv_worlds"]
        n_replans = min(clean_worlds.shape[0], adv_worlds.shape[0])
        replan_idx = _choose_replan(stats, n_replans)
        n_future = min(clean_worlds.shape[1], adv_worlds.shape[1])

        gif_path = sample_out / f"{safe_id}_future_replan{replan_idx:03d}.gif"
        frames_dir = sample_out / f"{safe_id}_future_replan{replan_idx:03d}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frames: list[Image.Image] = []
        if overwrite or not gif_path.exists():
            for t in range(n_future):
                frame = _make_panel(
                    clean_worlds[replan_idx, t],
                    adv_worlds[replan_idx, t],
                    diff_scale=diff_scale,
                    scale=world_scale,
                    labels=(f"clean future t={t}", f"adv future t={t}", f"abs diff x{diff_scale:g}"),
                )
                frame_path = frames_dir / f"frame_{t:03d}.png"
                if overwrite or not frame_path.exists():
                    frame.save(frame_path)
                frames.append(frame)
            if frames:
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=duration_ms,
                    loop=0,
                    optimize=False,
                )

        result.update(
            {
                "replan_idx": replan_idx,
                "future_gif": str(gif_path),
                "future_frames_dir": str(frames_dir),
                "num_future_frames": int(n_future),
                "chosen_replan_stats": stats[replan_idx] if replan_idx < len(stats) else {},
            }
        )
    else:
        result["warning"] = "missing clean_worlds/adv_worlds; skipped future GIF"
        n_replans = len(stats)
        replan_idx = _choose_replan(stats, n_replans) if n_replans else 0

    if "clean_images" in data.files and "adv_images" in data.files:
        clean_images = data["clean_images"]
        adv_images = data["adv_images"]
        n_obs = min(clean_images.shape[0], adv_images.shape[0])
        if n_obs > 0:
            if "replan_idx" not in result:
                replan_idx = min(_choose_replan(stats, n_obs), n_obs - 1)
            else:
                replan_idx = min(int(result["replan_idx"]), n_obs - 1)
            obs_path = sample_out / f"{safe_id}_obs_replan{replan_idx:03d}.png"
            if overwrite or not obs_path.exists():
                obs = _make_panel(
                    clean_images[replan_idx],
                    adv_images[replan_idx],
                    diff_scale=diff_scale,
                    scale=obs_scale,
                    labels=("clean observation", "adv observation", f"abs diff x{diff_scale:g}"),
                )
                obs.save(obs_path)
            result["obs_png"] = str(obs_path)

    summary_path = sample_out / f"{safe_id}_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary_json"] = str(summary_path)
    return result


def iter_manifests(root: Path) -> list[Path]:
    if root.is_file() and root.name == "manifest.json":
        return [root]
    return sorted(root.rglob("representatives/manifest.json"))


def render_manifest(
    manifest_path: Path,
    *,
    top_k: int,
    output_dir_name: str,
    diff_scale: float,
    world_scale: float,
    obs_scale: float,
    duration_ms: int,
    overwrite: bool,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = sorted(
        manifest.get("samples", []),
        key=lambda item: _to_float(item.get("score"), float("-inf")),
        reverse=True,
    )[:top_k]

    run_dir = manifest_path.parent.parent
    out_dir = run_dir / "analysis" / output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in samples:
        sample_path = _resolve_sample_path(manifest_path, item)
        if sample_path is None:
            missing.append(item)
            continue
        try:
            rendered.append(
                render_sample(
                    sample_path,
                    out_dir=out_dir,
                    diff_scale=diff_scale,
                    world_scale=world_scale,
                    obs_scale=obs_scale,
                    duration_ms=duration_ms,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append({"sample": item, "error": repr(exc)})

    summary = {
        "manifest": str(manifest_path),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "top_k": int(top_k),
        "num_requested": len(samples),
        "num_rendered": len(rendered),
        "num_missing": len(missing),
        "num_errors": len(errors),
        "rendered": rendered,
        "missing": missing,
        "errors": errors,
    }
    (out_dir / "render_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Experiment root or a representatives/manifest.json.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of representative samples to render per manifest.")
    parser.add_argument("--output-dir-name", default="representative_gifs", help="Subdirectory under each run's analysis/.")
    parser.add_argument("--diff-scale", type=float, default=8.0, help="Multiplier for absolute-difference visualization.")
    parser.add_argument("--world-scale", type=float, default=2.0, help="Scale factor for predicted-future frames.")
    parser.add_argument("--obs-scale", type=float, default=0.5, help="Scale factor for observation panels.")
    parser.add_argument("--duration-ms", type=int, default=450, help="GIF frame duration.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing GIFs/PNGs.")
    args = parser.parse_args()

    manifests = iter_manifests(args.root)
    if not manifests:
        raise SystemExit(f"No representatives/manifest.json found under {args.root}")

    all_summaries = []
    for manifest_path in manifests:
        print(f"[render] {manifest_path}")
        summary = render_manifest(
            manifest_path,
            top_k=args.top_k,
            output_dir_name=args.output_dir_name,
            diff_scale=args.diff_scale,
            world_scale=args.world_scale,
            obs_scale=args.obs_scale,
            duration_ms=args.duration_ms,
            overwrite=args.overwrite,
        )
        print(
            "  rendered={num_rendered}/{num_requested} missing={num_missing} errors={num_errors} -> {output_dir}".format(
                **summary
            )
        )
        all_summaries.append(summary)

    root_summary = {
        "root": str(args.root),
        "num_manifests": len(manifests),
        "num_rendered": int(sum(s["num_rendered"] for s in all_summaries)),
        "num_missing": int(sum(s["num_missing"] for s in all_summaries)),
        "num_errors": int(sum(s["num_errors"] for s in all_summaries)),
        "summaries": all_summaries,
    }
    summary_path = args.root / f"{args.output_dir_name}_render_summary.json" if args.root.is_dir() else args.root.with_suffix(".render_summary.json")
    summary_path.write_text(json.dumps(root_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {summary_path}")


if __name__ == "__main__":
    main()
