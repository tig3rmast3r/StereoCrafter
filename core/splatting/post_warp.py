"""Post-warp processing helpers.

Includes:
- Stair-step smoothing in a thin band near disparity edges.
- Replace-mask generation for hole-run expansion in output space.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _shift_right_bool(mask: torch.Tensor, dx: int) -> torch.Tensor:
    if dx <= 0:
        return mask
    out = torch.zeros_like(mask, dtype=torch.bool)
    out[..., :, dx:] = mask[..., :, :-dx]
    return out


def _shift_left_bool(mask: torch.Tensor, dx: int) -> torch.Tensor:
    if dx <= 0:
        return mask
    out = torch.zeros_like(mask, dtype=torch.bool)
    out[..., :, :-dx] = mask[..., :, dx:]
    return out


def _close1d_x_bool(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """1D closing along X on boolean masks [T,1,H,W]."""
    if radius <= 0:
        return mask
    x = mask.float()
    k = 2 * radius + 1
    dil = F.max_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, radius))
    ero = 1.0 - F.max_pool2d(1.0 - dil, kernel_size=(1, k), stride=1, padding=(0, radius))
    return ero > 0.5


def left_black_run_mask_from_rgb(
    rgb: torch.Tensor,
    *,
    tol: float = 0.0,
    max_px: Optional[int] = None,
) -> torch.Tensor:
    """Detect left-border black run per row.

    Args:
        rgb: [T,3,H,W] tensor.
        tol: Pixel threshold to classify black.
        max_px: Optional clamp for run length.

    Returns:
        Bool mask [T,1,H,W], True where replace should happen.
    """
    if rgb.dim() != 4 or rgb.shape[1] < 3:
        raise ValueError(f"left_black_run_mask_from_rgb expects (T,3,H,W), got {tuple(rgb.shape)}")

    x = rgb.float() if not torch.is_floating_point(rgb) else rgb
    black = (x[:, 0:3] <= float(tol)).all(dim=1)  # [T,H,W]

    nonblack = ~black
    any_nonblack = nonblack.any(dim=-1)  # [T,H]
    first_nonblack = nonblack.float().argmax(dim=-1)  # [T,H]
    width = black.shape[-1]
    first_nonblack = torch.where(any_nonblack, first_nonblack, torch.full_like(first_nonblack, width))

    if max_px is not None:
        clamped = max(1, min(int(max_px), width))
        first_nonblack = torch.minimum(first_nonblack, torch.full_like(first_nonblack, clamped))

    ar = torch.arange(width, device=rgb.device).view(1, 1, width)
    left_run = ar < first_nonblack.unsqueeze(-1)  # [T,H,W]
    left_run = left_run & black[:, :, 0].unsqueeze(-1)
    left_run = left_run & any_nonblack.unsqueeze(-1)
    return left_run.unsqueeze(1)


def build_replace_mask_edge_hole_run(
    disp_out_winner: torch.Tensor,
    hole_mask: torch.Tensor,
    *,
    grad_thr_px: float = 1.0,
    min_px: int = 1,
    max_px: int = 32,
    scale: float = 1.0,
    gap_tol: int = 0,
    draw_edge: bool = True,
) -> torch.Tensor:
    """Build replace mask from output-space hole runs.

    Note:
        `disp_out_winner` and `grad_thr_px` are currently kept for API compatibility.
    """
    del disp_out_winner, grad_thr_px

    if hole_mask.dim() != 4:
        raise ValueError(f"hole_mask must be [T,1,H,W], got {tuple(hole_mask.shape)}")

    _, _, _, width = hole_mask.shape
    hole_s = _close1d_x_bool(hole_mask.bool(), int(gap_tol))

    hole_prev = _shift_right_bool(hole_s, 1)
    seeds = hole_s & (~hole_prev)

    eff_max = int(max(0, round(int(max_px) * float(scale))))
    eff_max = max(0, min(eff_max, width))
    min_px_i = max(0, int(min_px))

    if min_px_i <= 1 or eff_max <= 0:
        seeds_ok = seeds
    else:
        steps = min(min_px_i - 1, eff_max)
        test = seeds
        for _ in range(steps):
            test = _shift_right_bool(test, 1) & hole_s
        seeds_ok = _shift_left_bool(test, steps)

    active = seeds_ok
    replace = seeds_ok.clone()

    for _ in range(eff_max):
        active = _shift_right_bool(active, 1) & hole_s
        replace |= active

    # Always keep a 1px stability boundary on the left side of hole-runs.
    # This avoids reintroducing internal waviness when the optional visible edge is disabled.
    boundary_core = _shift_left_bool(seeds_ok, 1) & (~hole_s)
    replace |= boundary_core

    # Optional visual edge: adds one extra left pixel line only.
    # This toggle is intentionally cosmetic and does not control stabilization anymore.
    if draw_edge:
        boundary_extra = _shift_left_bool(boundary_core, 1) & (~hole_s)
        replace |= boundary_extra

    return replace


def _sanitize_box_kernel_size(ksize: int) -> int:
    k = int(ksize)
    if k < 3:
        return 3
    if k % 2 == 0:
        k += 1
    return min(k, 15)


def apply_staircase_smooth_bgside(
    right_img: torch.Tensor,
    occlu_map: torch.Tensor,
    disp_out_winner: torch.Tensor,
    *,
    max_disp: float,
    edge_mode: str = "pos",
    grad_thr_px: float = 1.0,
    strip_px: int = 3,
    strength: float = 1.0,
    right_margin_extra: int = 0,
    debug_mask: bool = False,
    exclude_near_holes: bool = True,
    hole_dilate: int = 8,
    edge_x_offset: int = 0,
    blur_kernel: int = 3,
) -> torch.Tensor:
    """Smooth stair-step artifacts on the background side of splat edges."""
    if right_img.dim() not in (3, 4):
        raise ValueError(f"right_img must be [3,H,W] or [T,3,H,W], got {tuple(right_img.shape)}")

    single_frame = right_img.dim() == 3
    img = right_img.unsqueeze(0) if single_frame else right_img
    occ = occlu_map.unsqueeze(0) if occlu_map.dim() == 3 else occlu_map
    disp = disp_out_winner.unsqueeze(0) if disp_out_winner.dim() == 3 else disp_out_winner

    if img.shape[0] != occ.shape[0] and occ.shape[0] == 1:
        occ = occ.expand(img.shape[0], -1, -1, -1)
    if img.shape[0] != disp.shape[0] and disp.shape[0] == 1:
        disp = disp.expand(img.shape[0], -1, -1, -1)

    if strip_px <= 0 or strength <= 0:
        return right_img

    _, _, _, width = img.shape

    oc = occ.float()
    valid = (oc < 0.5).float() if float(oc.mean()) < 0.5 else (oc > 0.5).float()

    hole = (valid < 0.5).float()
    hole_d = hole
    if exclude_near_holes and int(hole_dilate) > 0:
        for _ in range(int(hole_dilate)):
            up = torch.zeros_like(hole_d)
            dn = torch.zeros_like(hole_d)
            lf = torch.zeros_like(hole_d)
            rt = torch.zeros_like(hole_d)
            up[..., 1:, :] = hole_d[..., :-1, :]
            dn[..., :-1, :] = hole_d[..., 1:, :]
            lf[..., :, 1:] = hole_d[..., :, :-1]
            rt[..., :, :-1] = hole_d[..., :, 1:]
            hole_d = torch.maximum(hole_d, torch.maximum(torch.maximum(up, dn), torch.maximum(lf, rt)))

    grad_x = disp[..., 1:] - disp[..., :-1]
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode="replicate")

    if edge_mode == "neg":
        mag = (-grad_x).clamp_min(0.0)
    elif edge_mode == "abs":
        mag = grad_x.abs()
    else:
        mag = grad_x.clamp_min(0.0)

    mag_l = F.pad(mag[..., :-1], (1, 0, 0, 0), mode="replicate")
    mag_r = F.pad(mag[..., 1:], (0, 1, 0, 0), mode="replicate")
    nms = (mag >= mag_l) & (mag > mag_r)
    edge = ((mag > float(grad_thr_px)) & nms).float()

    if edge_mode in ("pos", "abs"):
        left_nb = torch.zeros_like(edge)
        left_nb[..., 1:] = edge[..., :-1]
        edge = edge * (1.0 - left_nb)
    if edge_mode == "neg":
        right_nb = torch.zeros_like(edge)
        right_nb[..., :-1] = edge[..., 1:]
        edge = edge * (1.0 - right_nb)

    if exclude_near_holes and int(hole_dilate) > 0:
        edge = edge * (1.0 - (hole_d > 0).float())

    off = int(edge_x_offset)
    if off != 0:
        shifted = torch.zeros_like(edge)
        if off > 0:
            shifted[..., off:] = edge[..., :-off]
        else:
            oo = -off
            shifted[..., :-oo] = edge[..., oo:]
        edge = shifted

    right_margin = int(math.ceil(float(max_disp))) + int(right_margin_extra)
    safe = None
    if 0 < right_margin < width:
        safe = img.new_ones((1, 1, 1, width))
        safe[..., width - right_margin :] = 0.0
        if img.shape[0] > 1:
            safe = safe.expand(img.shape[0], -1, -1, -1)

    w = torch.zeros_like(edge)
    denom = max(1.0, float(strip_px))
    for i in range(1, int(strip_px) + 1):
        ring = torch.zeros_like(edge)
        ring[..., :-i] = edge[..., i:]
        ring = ring * valid
        if exclude_near_holes and int(hole_dilate) > 0:
            ring = ring * (1.0 - (hole_d > 0).float())
        wi = float(strength) * (1.0 - ((i - 1) / denom))
        if wi > 0:
            w = torch.maximum(w, ring * wi)

    if safe is not None:
        w = w * safe.to(dtype=w.dtype)

    if debug_mask:
        mask3 = (w > 0).expand(-1, 3, -1, -1).float()
        alpha = 0.65
        green = img.new_zeros(img.shape)
        green[:, 1:2] = 1.0
        out_dbg = img * (1.0 - alpha * mask3) + green * (alpha * mask3)
        return out_dbg[0] if single_frame else out_dbg

    ksz = _sanitize_box_kernel_size(blur_kernel)
    kernel = img.new_ones((1, 1, ksz, ksz)) / float(ksz * ksz)
    kernel = kernel.repeat(3, 1, 1, 1)
    blur = F.conv2d(img, kernel, padding=ksz // 2, groups=3)
    w3 = w.expand(-1, 3, -1, -1).clamp(0.0, 1.0)
    out = img * (1.0 - w3) + blur * w3

    return out[0] if single_frame else out
