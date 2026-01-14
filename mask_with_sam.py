#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageOps


SAM_VIT_B_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


@dataclass(frozen=True)
class SamConfig:
    model_type: str
    checkpoint_path: Path
    device: str


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    print(f"[sam3d-ready/mask_with_sam] downloading {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, tmp)  # noqa: S310
    tmp.replace(dest)


def _ensure_sam_checkpoint(checkpoint_path: Path) -> None:
    if checkpoint_path.exists():
        return
    _download(SAM_VIT_B_URL, checkpoint_path)


def _load_image_rgb(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _load_alpha_mask(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGBA")
    rgba = np.array(img, dtype=np.uint8)
    alpha = rgba[..., 3]
    return alpha > 0


def _mask_to_bbox(mask: np.ndarray, *, pad_ratio: float = 0.05) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    h, w = mask.shape[:2]
    pad_x = int((x1 - x0 + 1) * pad_ratio)
    pad_y = int((y1 - y0 + 1) * pad_ratio)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _mask_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def _largest_connected_component(mask_u8: np.ndarray) -> np.ndarray:
    import cv2

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _postprocess_mask(mask_bool: np.ndarray) -> np.ndarray:
    import cv2

    mask_u8 = (mask_bool.astype(np.uint8) * 255).copy()
    mask_u8 = _largest_connected_component(mask_u8)

    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = _largest_connected_component(mask_u8)
    return mask_u8


def _make_rgba_mask(mask_u8: np.ndarray) -> Image.Image:
    h, w = mask_u8.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = mask_u8
    return Image.fromarray(rgba)


def _make_preview(rgb: np.ndarray, mask_u8: np.ndarray) -> Image.Image:
    white = np.full_like(rgb, 255)
    alpha = (mask_u8.astype(np.float32) / 255.0)[..., None]
    comp = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
    return Image.fromarray(comp)


def _auto_default_bbox(h: int, w: int) -> np.ndarray:
    x0 = int(w * 0.12)
    x1 = int(w * 0.88)
    y0 = int(h * 0.03)
    y1 = int(h * 0.97)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _auto_points(h: int, w: int, centroid: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    # Positive point: centroid if available, else image center.
    if centroid is None:
        centroid = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    # Negative points: four corners.
    neg = np.array(
        [
            [1.0, 1.0],
            [w - 2.0, 1.0],
            [1.0, h - 2.0],
            [w - 2.0, h - 2.0],
        ],
        dtype=np.float32,
    )
    pts = np.vstack([centroid[None, :], neg])
    labels = np.array([1] + [0] * len(neg), dtype=np.int32)
    return pts, labels


def _pick_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    prompt_mask: Optional[np.ndarray],
) -> np.ndarray:
    if masks.ndim != 3:
        raise ValueError(f"Unexpected masks shape: {masks.shape}")
    if prompt_mask is None:
        return masks[int(np.argmax(scores))]

    prompt = prompt_mask.astype(bool)
    best_idx = 0
    best_val = -1.0
    for i in range(masks.shape[0]):
        m = masks[i].astype(bool)
        inter = np.logical_and(m, prompt).sum()
        union = np.logical_or(m, prompt).sum()
        iou = float(inter) / float(union) if union else 0.0
        val = iou + 0.15 * float(scores[i])
        if val > best_val:
            best_val = val
            best_idx = i
    return masks[best_idx]


def _iter_image_names(images_dir: Path, names: Optional[str]) -> list[str]:
    if names:
        return [n.strip() for n in names.split(",") if n.strip()]
    files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
    files = [p for p in files if "_mask" not in p.name]

    def sort_key(p: Path):
        try:
            return (0, int(p.stem), p.stem)
        except ValueError:
            return (1, 0, p.stem)

    files = sorted(files, key=sort_key)
    return [p.stem for p in files]


def _load_sam(cfg: SamConfig):
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "segment_anything is not installed in this env. Install with:\n"
            "  pip install 'git+https://github.com/facebookresearch/segment-anything.git'"
        ) from exc

    _ensure_sam_checkpoint(cfg.checkpoint_path)
    sam = sam_model_registry[cfg.model_type](checkpoint=str(cfg.checkpoint_path))
    sam.to(device=cfg.device)
    return SamPredictor(sam)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate higher-quality masks using Segment Anything (SAM).")
    parser.add_argument("--input_dir", type=str, required=True, help="Dataset root (contains images/ and mask folders)")
    parser.add_argument("--images_dir", type=str, default="images", help="Relative images directory under input_dir")
    parser.add_argument("--mask_prompt", type=str, required=True, help="Output mask folder name (created under input_dir)")
    parser.add_argument(
        "--prompt_mask_prompt",
        type=str,
        default=None,
        help="Optional existing mask folder name under input_dir to use as prompt (bbox/centroid).",
    )
    parser.add_argument("--image_names", type=str, default=None, help="Comma-separated image stems (default: auto-detect)")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint (default: ~/.cache/segment_anything/sam_vit_b_01ec64.pth for vit_b)",
    )
    parser.add_argument("--device", type=str, default="auto", help="cuda/cpu/auto")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks in output folder")

    args = parser.parse_args(list(argv) if argv is not None else None)

    input_dir = Path(args.input_dir).expanduser().resolve()
    images_dir = input_dir / args.images_dir
    if not images_dir.is_dir():
        raise SystemExit(f"Missing images dir: {images_dir}")

    out_masks_dir = input_dir / args.mask_prompt
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    prompt_masks_dir: Optional[Path] = None
    if args.prompt_mask_prompt:
        candidate = input_dir / args.prompt_mask_prompt
        if not candidate.is_dir():
            raise SystemExit(f"prompt_mask_prompt directory not found: {candidate}")
        prompt_masks_dir = candidate

    device = _resolve_device(args.device)
    if args.model_type != "vit_b" and args.checkpoint is None:
        raise SystemExit("For vit_l/vit_h, please pass --checkpoint explicitly.")

    if args.checkpoint is None:
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        checkpoint_path = cache_dir / "segment_anything" / "sam_vit_b_01ec64.pth"
    else:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    predictor = _load_sam(SamConfig(model_type=args.model_type, checkpoint_path=checkpoint_path, device=device))

    names = _iter_image_names(images_dir, args.image_names)
    print(f"[sam3d-ready/mask_with_sam] processing {len(names)} images: {names}", flush=True)

    for name in names:
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = images_dir / f"{name}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"[sam3d-ready/mask_with_sam] skip (missing): {name}", flush=True)
            continue

        out_mask_path = out_masks_dir / f"{name}_mask.png"
        if out_mask_path.exists() and not args.overwrite:
            print(f"[sam3d-ready/mask_with_sam] exists, skipping: {out_mask_path}", flush=True)
            continue

        rgb = _load_image_rgb(img_path)
        h, w = rgb.shape[:2]

        prompt_mask = None
        if prompt_masks_dir is not None:
            pm_path = prompt_masks_dir / f"{name}_mask.png"
            if pm_path.exists():
                prompt_mask = _load_alpha_mask(pm_path)

        box = _mask_to_bbox(prompt_mask) if prompt_mask is not None else None
        if box is None:
            box = _auto_default_bbox(h, w)

        centroid = _mask_centroid(prompt_mask) if prompt_mask is not None else None
        point_coords, point_labels = _auto_points(h, w, centroid)

        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        chosen = _pick_mask(masks, scores, prompt_mask)
        mask_u8 = _postprocess_mask(chosen)

        rgba = _make_rgba_mask(mask_u8)
        rgba.save(out_mask_path)

        preview = _make_preview(rgb, mask_u8)
        preview_path = out_masks_dir / f"{name}_preview.jpg"
        preview.save(preview_path, quality=85)

        coverage = float((mask_u8 > 0).mean())
        print(f"[sam3d-ready/mask_with_sam] {name}: coverage={coverage:.3f} -> {out_mask_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

