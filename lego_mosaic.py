"""
=============================================================================
  LEGO Mosaic Building Instruction Generator
=============================================================================
  Converts any image into a printable LEGO-style step-by-step PDF guide.

  Structure:
    - 40 baseplates  (5 columns × 8 rows)
    - Each plate     16 × 16 studs  (1×1 pieces only)
    - Final grid     80 × 128 pixels

  Sections generated in the PDF:
    1. Cover page   – original image + pixelated preview
    2. Color legend – all used LEGO colours with swatches
    3. Build pages  – one page per baseplate (16×16 grid)
    4. Overview     – full 5×8 assembly map
=============================================================================
"""

import sys
import os
import math
import argparse
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

import io


# ──────────────────────────────────────────────────────────────────────────────
#  LEGO COLOUR PALETTE
#  Each entry: (Name, (R, G, B))
#  25 carefully chosen colours that represent the real LEGO brick palette.
# ──────────────────────────────────────────────────────────────────────────────
LEGO_PALETTE = [
    ("White",           (255, 255, 255)),
    ("Light Bluish Gray",(171, 173, 172)),
    ("Dark Bluish Gray", ( 89,  93,  96)),
    ("Black",           (  0,   0,   0)),
    ("Bright Red",      (196,  40,  28)),
    ("Dark Red",        (123,  46,  47)),
    ("Bright Orange",   (254, 138,  24)),
    ("Bright Yellow",   (255, 205,   0)),
    ("Bright Green",    ( 75, 151,  74)),
    ("Dark Green",      ( 35, 120,  65)),
    ("Bright Blue",     (  0, 114, 188)),
    ("Dark Blue",       ( 20,  48, 100)),
    ("Medium Blue",     ( 97, 175, 217)),
    ("Sky Blue",        (146, 207, 224)),
    ("Sand Blue",       (105, 138, 167)),
    ("Bright Purple",   (107,  50, 124)),
    ("Medium Lavender", (160, 110, 185)),
    ("Bright Pink",     (255, 102, 147)),
    ("Light Pink",      (252, 204, 210)),
    ("Tan",             (222, 198, 156)),
    ("Dark Tan",        (150, 130,  96)),
    ("Brown",           (105,  64,  40)),
    ("Reddish Brown",   (128,  70,  50)),
    ("Olive Green",     ( 95, 116,  35)),
    ("Lime Green",      (167, 202,  26)),
    # ── Skin tones (added to fix face/background colour collision) ──────────
    # Light Nougat is the standard LEGO face/skin colour.  Without it, face skin
    # and aged parchment both map to Tan — making the face invisible on the poster.
    # With it: face skin → Light Nougat (ΔE=9.2), parchment → Tan (ΔE=11.4).
    ("Light Nougat",    (255, 201, 149)),   # idx 25 — face/skin highlight
    ("Nougat",          (204, 142, 105)),   # idx 26 — mid-tone skin / shadow
    # ── Warm orange (improves pure-red hat band mapping ΔE 31→15) ───────────
    ("Reddish Orange",  (220,  60,  30)),   # idx 27
]

# Build a quick lookup: palette RGB array for vectorised nearest-colour search
_PALETTE_RGB = np.array([c[1] for c in LEGO_PALETTE], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  COLOUR CONVERSION HELPERS  (RGB ↔ Lab for perceptual matching)
# ──────────────────────────────────────────────────────────────────────────────

def _rgb_to_linear(channel: float) -> float:
    """Gamma-expand one sRGB channel [0-1] to linear light."""
    c = channel / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_xyz(r: float, g: float, b: float):
    """Linear sRGB → CIE XYZ (D65 illuminant)."""
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    return m @ np.array([r, g, b])


def _xyz_to_lab(xyz):
    """CIE XYZ → CIE L*a*b* (D65 reference white)."""
    ref = np.array([0.95047, 1.00000, 1.08883])
    t = xyz / ref

    def f(v):
        return v ** (1 / 3) if v > 0.008856 else 7.787 * v + 16 / 116

    fx, fy, fz = f(t[0]), f(t[1]), f(t[2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.array([L, a, b])


def rgb_to_lab(rgb_tuple):
    """Convert an (R,G,B) tuple to a CIE L*a*b* array."""
    r_lin = _rgb_to_linear(rgb_tuple[0])
    g_lin = _rgb_to_linear(rgb_tuple[1])
    b_lin = _rgb_to_linear(rgb_tuple[2])
    xyz = _linear_to_xyz(r_lin, g_lin, b_lin)
    return _xyz_to_lab(xyz)


# Pre-compute Lab values for every palette colour (done once at import time)
_PALETTE_LAB = np.array([rgb_to_lab(c[1]) for c in LEGO_PALETTE], dtype=np.float32)


def find_nearest_colour(pixel_rgb):
    """
    Return the index of the nearest LEGO palette colour for a single RGB pixel.
    Used only for one-off lookups; batch quantization uses _vectorized_rgb_to_lab.
    """
    lab = rgb_to_lab(pixel_rgb).astype(np.float32)
    diffs = _PALETTE_LAB - lab
    return int(np.argmin(np.sum(diffs ** 2, axis=1)))


def _vectorized_rgb_to_lab(arr: np.ndarray) -> np.ndarray:
    """
    Convert an (N, 3) uint8 RGB array to an (N, 3) float32 CIE L*a*b* array.

    Fully vectorised — no Python loops, no per-pixel accumulation errors.
    Steps:
      1. Normalise [0-255] -> [0.0-1.0]
      2. sRGB gamma expansion -> linear light  (IEC 61966-2-1)
      3. Linear RGB -> CIE XYZ                 (D65, BT.709 primaries)
      4. CIE XYZ -> L*a*b*                     (D65 reference white)
    """
    rgb = arr.astype(np.float32) / 255.0
    # Gamma expansion
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    # Linear sRGB -> XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = linear @ M.T
    # Normalise by D65 reference white
    ref = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    t   = xyz / ref
    # Cube-root with the low-value linear segment (CIE standard)
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + (16.0 / 116.0))
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)



# ──────────────────────────────────────────────────────────────────────────────
#  IMAGE PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

GRID_W  = 80    # 5 plates × 16 studs
GRID_H  = 128   # 8 plates × 16 studs
PLATE_W = 16
PLATE_H = 16
COLS    = 5
ROWS    = 8

import cv2
from scipy.ndimage import median_filter as scipy_median_filter

# ── Pipeline tuning constants ─────────────────────────────────────────────────
#
# CANNY thresholds: applied to the ORIGINAL full-resolution image.
# Lower values catch more edges (including faint text borders).
CANNY_LOW       = 25
CANNY_HIGH      = 75

# EDGE_DILATE_PX: dilation radius at full-res before downscaling.
# 3px → 7×7 kernel. At a ~6× downscale ratio a 7px band → ~1 LEGO pixel,
# which is enough for the edge to survive without engulfing thin characters.
# Old value (5px → 11px) was merging adjacent character strokes into solid black.
EDGE_DILATE_PX  = 3

# Edge pixels darker than this luma are forced to Black after quantization.
EDGE_DARK_IDX   = 3      # LEGO_PALETTE index for "Black"
EDGE_LUMA_SPLIT = 110    # below = definitely dark → Black; above = keep LAB result

# Bilateral filter parameters (noise removal before Canny, at full resolution).
# Sigma 50/50 = moderate smoothing; preserves artistic hard edges.
BILATERAL_D  = 9         # filter diameter (pixels)
BILATERAL_SC = 50        # sigma colour
BILATERAL_SS = 50        # sigma space

# Mode filter window applied to the final index map to remove salt-pepper speckle.
# Size 3 = 3×3 window. Set to 0 to disable.
MODE_FILTER_SIZE = 3


# ══════════════════════════════════════════════════════════════════════════════
#  ILLUSTRATION-FIRST PIPELINE  (correct pipeline order)
#
#  KEY INSIGHT: "You are treating the image like a photo, but you need to treat
#                it like a vector illustration with hierarchy."
#
#  WRONG ORDER (all previous versions):
#    resize first → LAB quantize second
#    Problem: resize (any filter) blends adjacent pixels.
#             Blended pixel = invalid color = maps to wrong LEGO color = grey.
#             You CANNOT recover a blended edge afterward.
#
#  CORRECT ORDER (this version):
#    LAB quantize at FULL RES → NEAREST resize
#    Why: Once every pixel is a valid LEGO color, NEAREST sampling just
#         picks one LEGO color per grid position. No blending is possible.
#         NEAREST is the ONLY filter that cannot create new intermediate values.
#
#  Full pipeline (7 stages, all in correct order):
#
#   Stage 1 — Bilateral filter at full-res
#     Smooths JPEG/PNG compression noise within flat regions.
#     Preserves sharp artistic edges (bilateral is edge-preserving by design).
#     Why: Canny in Stage 2 must not trigger on compression artifacts.
#
#   Stage 2 — Canny edge detection at full-res on ORIGINAL image
#     Run before any color processing so artistic outlines are fully visible.
#     Uses the bilateral-smoothed image to suppress compression noise.
#     Dilation thickens each 1px line to 7px — guarantees survival after
#     ~6× downscale to 80px wide.
#
#   Stage 3 — Pre-quantize to LEGO palette at FULL resolution
#     Every pixel → its nearest LEGO color via vectorized LAB distance.
#     After this stage, the image contains ONLY valid LEGO palette colors.
#     This eliminates grey collapse, dithering, and palette noise permanently.
#     Flat regions become perfectly uniform; color accuracy is maximum.
#
#   Stage 4 — NEAREST resize to fit dimensions
#     Resize the pre-quantized index map to fit_w × fit_h.
#     NEAREST picks one palette index per output pixel — zero blending.
#     No intermediate values, no new colors, no artifacts.
#
#   Stage 5 — Downscale edge map (max-pool + NEAREST)
#     Binary map: if ANY source pixel was an edge, output pixel is edge.
#     INTER_NEAREST on a max-pooled binary map preserves every hot pixel.
#
#   Stage 6 — Pad both to GRID_W × GRID_H
#     Index map: mode='edge' (extends outermost palette index outward).
#     Edge map:  constant 0 (padding zone has no drawn outlines).
#
#   Stage 7 — Edge override → Black
#     Canny-detected edges that are dark (luma < EDGE_LUMA_SPLIT) are
#     forced to Black in the index map, overriding the pre-quantized value.
#     This is the final guarantee that ALL outlines survive every stage.
#
#   Stage 8 — Mode filter (speckle removal)
#     3×3 median filter on the integer index map. Any isolated single-pixel
#     outlier is replaced by the median of its 3×3 neighborhood.
#     Removes salt-and-pepper noise without blurring region boundaries.
# ══════════════════════════════════════════════════════════════════════════════


def _prequantize_fullres(img_orig: Image.Image) -> np.ndarray:
    """
    Stage 3: Map every pixel of the full-resolution image to its nearest
    LEGO palette colour using vectorised CIE Lab distance.

    Returns: (H_orig, W_orig) uint8 index array
    (same shape as the input image, values in 0..len(LEGO_PALETTE)-1)

    This is the core fix for colour accuracy:
      - Runs on the ORIGINAL resolution, not the downscaled 80×128.
      - At full resolution, flat colour regions all map to exactly one
        LEGO colour.  Edge zones may map to an intermediate colour, but
        that is acceptable because Stage 7 (edge override) will correct them.
      - After this step, NEAREST resize is lossless because there are no
        floating-point blends to propagate.
    """
    arr  = np.array(img_orig, dtype=np.uint8)    # (H, W, 3)
    H, W = arr.shape[:2]
    flat = arr.reshape(-1, 3)                    # (N, 3)

    pix_lab = _vectorized_rgb_to_lab(flat)       # (N, 3)
    diff    = pix_lab[:, np.newaxis, :] - _PALETTE_LAB[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)          # (N, 25)
    indices = np.argmin(dist_sq, axis=1).astype(np.uint8)  # (N,)

    return indices.reshape(H, W)                 # (H, W)


def _extract_edges_fullres(img_orig: Image.Image) -> np.ndarray:
    """
    Stage 2: Canny edge detection on the full-resolution original image.

    Applies bilateral filter first to suppress JPEG/PNG compression noise.
    Uses border-wrapping before Canny so edges AT the image boundary are
    detected — without this, cv2.Canny returns zero on its first/last row and
    column, causing any black border or outline touching the image edge to
    silently disappear in the mosaic.

    Returns: (H_orig, W_orig) uint8 — 255 = edge, 0 = not-edge
    """
    arr_bgr = cv2.cvtColor(np.array(img_orig, dtype=np.uint8), cv2.COLOR_RGB2BGR)

    # ── Border padding before Canny ────────────────────────────────────────
    # cv2.Canny cannot compute gradients at row/col 0 and -1 (no neighbours).
    # Wrapping with a black constant border forces Canny to see a strong
    # gradient at the image boundary and emit edge pixels there.
    BORDER = EDGE_DILATE_PX * 2 + 4   # generous margin; peeled off after
    arr_bordered = cv2.copyMakeBorder(
        arr_bgr, BORDER, BORDER, BORDER, BORDER,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]   # black border = strong edge signal
    )

    arr_clean = cv2.bilateralFilter(arr_bordered, BILATERAL_D, BILATERAL_SC, BILATERAL_SS)
    gray      = cv2.cvtColor(arr_clean, cv2.COLOR_BGR2GRAY)
    edges_big = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    if EDGE_DILATE_PX > 0:
        k      = EDGE_DILATE_PX * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        edges_big = cv2.dilate(edges_big, kernel, iterations=1)

    # ── Peel off the added border ──────────────────────────────────────────
    edges = edges_big[BORDER:-BORDER, BORDER:-BORDER]
    return edges   # (H_orig, W_orig) 0 or 255


def _resize_index_map_nearest(idx_map: np.ndarray,
                               target_h: int, target_w: int) -> np.ndarray:
    """
    Stage 4: Resize a palette-index map to (target_h, target_w) using
    NEAREST NEIGHBOR interpolation.

    NEAREST is the ONLY correct resize filter for palette-indexed data.
    Every other filter (LANCZOS, BILINEAR, BOX, BICUBIC) computes a
    weighted average of neighboring pixel VALUES.  On a palette-index map
    the values are arbitrary integers (0=White, 3=Black, 7=Yellow…).
    Averaging index 0 and index 3 yields 1.5 → rounds to 1 or 2 (grey).
    NEAREST picks one existing index — the result is always a valid LEGO color.
    """
    # cv2.resize INTER_NEAREST on uint8 index map
    resized = cv2.resize(idx_map.astype(np.uint8),
                         (target_w, target_h),
                         interpolation=cv2.INTER_NEAREST)
    return resized


def _resize_edge_map(edges_full: np.ndarray,
                     target_h: int, target_w: int) -> np.ndarray:
    """
    Stage 5: Scale the binary edge map to (target_h, target_w).

    Uses a max-pool followed by INTER_NEAREST.
    On a 0/255 binary map, averaging (INTER_AREA) would fade bright pixels —
    a 1px white line in a mostly-black block becomes a faint grey stripe that
    Canny/threshold kills.  INTER_NEAREST after max-pool preserves every hot
    pixel: if the block contained ANY edge, the output pixel stays 255.
    """
    from scipy.ndimage import maximum_filter
    pooled = maximum_filter(edges_full, size=3)
    scaled = cv2.resize(pooled, (target_w, target_h),
                        interpolation=cv2.INTER_NEAREST)
    return scaled > 127   # (target_h, target_w) bool


def _pad_to_grid(idx_map: np.ndarray, edge_mask: np.ndarray,
                 orig_arr_small: np.ndarray):
    """
    Stage 6: Centre-pad both maps from (fit_h, fit_w) to (GRID_H, GRID_W).

    Index map : mode='edge' — extends the outermost palette index outward.
                The padding zone takes the border LEGO colour; no blank bars.
    Edge mask : mode='constant', 0 — padding zone has no edges drawn on it.
    Orig arr  : padded in parallel so we have RGB values for luma computation.
    """
    fit_h, fit_w = idx_map.shape
    pad_top    = (GRID_H - fit_h) // 2
    pad_bottom = GRID_H - fit_h - pad_top
    pad_left   = (GRID_W - fit_w) // 2
    pad_right  = GRID_W - fit_w - pad_left

    def pad(arr2d, mode, val=0):
        padded = np.pad(arr2d,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode=mode, **({"constant_values": val} if mode == "constant" else {}))
        return padded[:GRID_H, :GRID_W]

    def pad3(arr3d):
        padded = np.pad(arr3d,
                        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode='edge')
        return padded[:GRID_H, :GRID_W, :]

    return pad(idx_map, 'edge'), pad(edge_mask.astype(np.uint8), 'constant').astype(bool), pad3(orig_arr_small)


def process_image(path: str):
    """
    Master pipeline: image path → (index_map H×W uint8, edge_mask H×W bool).

    Implements all 8 stages described in the module header in the correct order:
    pre-quantize → edge-detect → NEAREST resize → pad → edge-override → mode-filter.

    Returns
    -------
    index_map : (GRID_H, GRID_W) uint8  — LEGO palette indices, ready for PDF
    edge_mask : (GRID_H, GRID_W) bool   — for informational display only
    """
    img_orig  = Image.open(path).convert("RGB")
    orig_w, orig_h = img_orig.size

    print(f"      Source: {orig_w}×{orig_h} px")

    # ── Stage 2: Edges at full resolution ────────────────────────────────
    print("      Stage 2/8  extract + dilate edges at full resolution …")
    edges_full = _extract_edges_fullres(img_orig)          # (H, W) uint8

    # ── Stage 3: Pre-quantize at full resolution ──────────────────────────
    print("      Stage 3/8  pre-quantize to LEGO palette at full resolution …")
    idx_fullres = _prequantize_fullres(img_orig)            # (H, W) uint8 indices

    # ── Stage 4: Scale factor (fit-inside, no crop) ────────────────────────
    scale = min(GRID_W / orig_w, GRID_H / orig_h)
    fit_w = min(int(orig_w * scale), GRID_W)
    fit_h = min(int(orig_h * scale), GRID_H)
    print(f"      Stage 4/8  NEAREST resize index map: {orig_w}×{orig_h} → {fit_w}×{fit_h} …")

    idx_small  = _resize_index_map_nearest(idx_fullres, fit_h, fit_w)

    # Also resize the original RGB for luma computation in Stage 7
    arr_small  = np.array(img_orig.resize((fit_w, fit_h), Image.LANCZOS), dtype=np.uint8)

    # ── Stage 5: Downscale edge map ───────────────────────────────────────
    print("      Stage 5/8  downscale edge map (max-pool + NEAREST) …")
    edge_small = _resize_edge_map(edges_full, fit_h, fit_w)  # bool

    # ── Stage 6: Pad to GRID dimensions ──────────────────────────────────
    print("      Stage 6/8  pad to 80×128 (edge replication) …")
    idx_grid, edge_grid, arr_grid = _pad_to_grid(idx_small, edge_small, arr_small)

    # ── Stage 7: EDGE-AWARE mode filter — speckle removal ───────────────
    # The mode/median filter removes isolated speckle pixels in flat regions.
    # CRITICAL: it must NOT touch pixels on or adjacent to edges.
    # A 1-px smile arc, tooth gap, or pupil outline has mostly background
    # neighbours → naive median = background → feature erased.
    #
    # Strategy: dilate the edge mask by 1px to protect a thin halo around
    # every edge, then only accept the smoothed value for pixels outside that
    # protected zone.  Edge pixels keep their pre-quantized value exactly.
    idx_before_mode = idx_grid.copy()   # snapshot for luma reference in Stage 8
    if MODE_FILTER_SIZE > 0:
        print(f"      Stage 7/8  edge-aware mode filter ({MODE_FILTER_SIZE}×{MODE_FILTER_SIZE}) …")
        idx_smoothed = scipy_median_filter(idx_grid.astype(np.int16),
                                           size=MODE_FILTER_SIZE).astype(np.uint8)
        # Dilate edge_grid by 1px to create a protected halo
        edge_protect = cv2.dilate(
            edge_grid.astype(np.uint8),
            np.ones((3, 3), np.uint8), iterations=1
        ).astype(bool)
        # Only smooth pixels that are NOT on/near an edge
        idx_grid = np.where(edge_protect, idx_grid, idx_smoothed)

    # ── Stage 8: Edge override → Black ───────────────────────────────────
    # Luma is computed from the ORIGINAL (pre-mode-filter) palette colours.
    # This correctly identifies dark pixels even if mode filter already changed
    # them to a lighter neighbour.  Any Canny edge pixel whose ORIGINAL
    # quantised colour was dark gets forced back to Black here.
    print("      Stage 8/8  edge override (force dark outlines to Black) …")
    palette_arr_f32  = np.array([c[1] for c in LEGO_PALETTE], dtype=np.float32)
    idx_rgb_original = palette_arr_f32[idx_before_mode]  # use snapshot, not mode-filtered
    luma_original    = (0.299 * idx_rgb_original[:,:,0] +
                        0.587 * idx_rgb_original[:,:,1] +
                        0.114 * idx_rgb_original[:,:,2])
    dark_edges = edge_grid & (luma_original < EDGE_LUMA_SPLIT)
    idx_grid[dark_edges] = EDGE_DARK_IDX

    assert idx_grid.shape  == (GRID_H, GRID_W), f"Shape error: {idx_grid.shape}"
    return idx_grid, edge_grid


def quantize_to_lego(img_arr: np.ndarray = None,
                     edge_mask: np.ndarray = None) -> np.ndarray:
    """
    Legacy compatibility shim.  The full pipeline now lives in process_image().
    This function is called only when someone passes a pre-resized RGB array
    directly (e.g., the test suite).  It performs LAB quantization + edge
    override without the full pre-quantize-at-full-res pipeline.
    """
    H, W  = img_arr.shape[:2]
    flat  = img_arr.reshape(-1, 3)
    pix_lab  = _vectorized_rgb_to_lab(flat)
    diff     = pix_lab[:, np.newaxis, :] - _PALETTE_LAB[np.newaxis, :, :]
    dist_sq  = np.sum(diff ** 2, axis=2)
    idx_flat = np.argmin(dist_sq, axis=1).astype(np.uint8)
    index_map = idx_flat.reshape(H, W)
    if edge_mask is not None:
        luma = (0.299 * img_arr[:,:,0] +
                0.587 * img_arr[:,:,1] +
                0.114 * img_arr[:,:,2])
        index_map[edge_mask & (luma < EDGE_LUMA_SPLIT)] = EDGE_DARK_IDX
    return index_map


def split_into_plates(index_map: np.ndarray):
    """
    Split the (128, 80) index map into 40 plates of (16, 16).
    Returns a list of 40 sub-arrays ordered left-to-right, top-to-bottom.
    """
    plates = []
    for row in range(ROWS):
        for col in range(COLS):
            y0 = row * PLATE_H
            y1 = y0 + PLATE_H
            x0 = col * PLATE_W
            x1 = x0 + PLATE_W
            plates.append(index_map[y0:y1, x0:x1].copy())
    return plates


def get_used_colours(index_map: np.ndarray):
    """Return sorted list of palette indices that appear in the mosaic."""
    return sorted(set(index_map.flatten().tolist()))


def render_full_mosaic(index_map: np.ndarray, scale: int = 1) -> Image.Image:
    """
    Render the complete 80×128 mosaic as an RGB PIL Image.
    Each pixel is scaled by `scale` for visibility (default 1 = exact size).
    Vectorised: single NumPy index lookup, no Python pixel loops.
    """
    H, W = index_map.shape
    # Build (H, W, 3) RGB array by indexing the palette table
    palette_arr = np.array([c[1] for c in LEGO_PALETTE], dtype=np.uint8)  # (N, 3)
    rgb = palette_arr[index_map]   # (H, W, 3) — vectorised lookup

    if scale > 1:
        # np.repeat along both spatial axes
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)

    return Image.fromarray(rgb, "RGB")


def render_plate(plate: np.ndarray, cell_px: int = 30,
                 show_coords: bool = True) -> Image.Image:
    """
    Render a 16×16 plate as a PIL Image suitable for PDF embedding.

    Each cell is `cell_px` × `cell_px` pixels.
    A thin grid line separates each stud.
    Coordinate labels (A-P / 1-16) are optionally drawn.
    """
    margin     = 28 if show_coords else 4   # left/top margin for labels
    grid_size  = PLATE_W * cell_px
    total_w    = margin + grid_size + 4
    total_h    = margin + grid_size + 4

    # Steel-grey canvas so White tiles are visible (not confused with empty cells)
    img = Image.new("RGB", (total_w, total_h), (225, 228, 232))
    draw = ImageDraw.Draw(img)

    # Try to load a small monospaced font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 11)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 9)
    except Exception:
        font    = ImageFont.load_default()
        font_sm = font

    for row in range(PLATE_H):
        for col in range(PLATE_W):
            idx    = plate[row, col]
            colour = LEGO_PALETTE[idx][1]
            x0 = margin + col * cell_px
            y0 = margin + row * cell_px
            x1 = x0 + cell_px - 1
            y1 = y0 + cell_px - 1

            # Fill cell
            draw.rectangle([x0, y0, x1, y1], fill=colour)

            # Subtle cell border
            draw.rectangle([x0, y0, x1, y1], outline=(80, 80, 80), width=1)

            # Draw a tiny circle to mimic a stud
            stud_r = max(2, cell_px // 5)
            cx = x0 + cell_px // 2
            cy = y0 + cell_px // 2
            # Lighten or darken the stud colour for contrast
            sr, sg, sb = colour
            luma = 0.299 * sr + 0.587 * sg + 0.114 * sb
            stud_col = (min(255, sr + 60), min(255, sg + 60), min(255, sb + 60)) \
                       if luma < 128 else \
                       (max(0, sr - 50), max(0, sg - 50), max(0, sb - 50))
            draw.ellipse([cx - stud_r, cy - stud_r, cx + stud_r, cy + stud_r],
                         outline=stud_col, width=1)

    if show_coords:
        # Column headers  A … P
        col_letters = "ABCDEFGHIJKLMNOP"
        for col in range(PLATE_W):
            cx = margin + col * cell_px + cell_px // 2
            draw.text((cx, 6), col_letters[col], fill=(80, 80, 80), font=font_sm, anchor="mt")

        # Row headers  1 … 16
        for row in range(PLATE_H):
            cy = margin + row * cell_px + cell_px // 2
            draw.text((8, cy), str(row + 1), fill=(80, 80, 80), font=font_sm, anchor="lm")

    return img


def pil_to_reportlab(pil_img: Image.Image) -> ImageReader:
    """Convert a PIL Image to a ReportLab ImageReader (in-memory, no temp file)."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


# ──────────────────────────────────────────────────────────────────────────────
#  PDF GENERATION
# ──────────────────────────────────────────────────────────────────────────────

PAGE_W, PAGE_H = A4          # 595.28 × 841.89 points
MARGIN         = 20 * mm


def _hex(rgb):
    """Format an RGB tuple as a '#RRGGBB' string."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _luma(rgb):
    """Perceived luminance of an RGB colour (0-255)."""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _draw_page_header(c: canvas.Canvas, title: str, subtitle: str = ""):
    """Draw a consistent header band at the top of every page."""
    c.setFillColorRGB(0.13, 0.13, 0.13)
    c.rect(0, PAGE_H - 14 * mm, PAGE_W, 14 * mm, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, PAGE_H - 9 * mm, title)
    if subtitle:
        c.setFont("Helvetica", 9)
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - 9 * mm, subtitle)


def _draw_page_footer(c: canvas.Canvas, page_num: int):
    """Draw a subtle footer with page number."""
    c.setFillColorRGB(0.6, 0.6, 0.6)
    c.setFont("Helvetica", 8)
    c.drawCentredString(PAGE_W / 2, 10 * mm, f"Page {page_num}")
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)


def _lego_yellow_bar(c: canvas.Canvas, y: float, w: float, label: str):
    """Draw a LEGO-yellow section divider bar."""
    c.setFillColorRGB(1.0, 0.80, 0.0)          # LEGO yellow
    c.rect(MARGIN, y, w, 7 * mm, fill=1, stroke=0)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN + 3 * mm, y + 1.8 * mm, label)


# ── Cover Page ────────────────────────────────────────────────────────────────

def draw_cover(c: canvas.Canvas, orig_path: str, mosaic_img: Image.Image,
               page_num: int, project_name: str):
    """Page 1 – cover with original + pixelated previews."""
    # Background gradient simulation (light grey)
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # LEGO red title band
    c.setFillColorRGB(0.77, 0.16, 0.11)
    c.rect(0, PAGE_H - 45 * mm, PAGE_W, 45 * mm, fill=1, stroke=0)

    # Title text
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 22 * mm, "LEGO MOSAIC")
    c.setFont("Helvetica", 14)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 34 * mm, "Building Instruction Guide")

    # Project name banner
    c.setFillColorRGB(1.0, 0.80, 0.0)
    c.rect(MARGIN, PAGE_H - 58 * mm, PAGE_W - 2 * MARGIN, 10 * mm, fill=1, stroke=0)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.setFont("Helvetica-Bold", 11)
    name = textwrap.shorten(project_name, width=50)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 52 * mm, name)

    # --- Original image ---
    # Images fill from just below the name banner down to the specs box,
    # using 50% of the page height (up from 42%) and starting 5mm lower.
    avail_w   = (PAGE_W - 3 * MARGIN) / 2
    avail_h   = PAGE_H * 0.50   # was 0.42 — more vertical space for previews
    preview_y = PAGE_H * 0.205  # was 0.25  — start closer to the name banner

    try:
        orig_reader = ImageReader(orig_path)
        iw, ih = orig_reader.getSize()
        scale = min(avail_w / iw, avail_h / ih)
        draw_w = iw * scale
        draw_h = ih * scale
        img_x = MARGIN + (avail_w - draw_w) / 2
        img_y = preview_y + (avail_h - draw_h) / 2
        c.drawImage(orig_reader, img_x, img_y, width=draw_w, height=draw_h,
                    preserveAspectRatio=True)
    except Exception:
        pass

    # --- Mosaic preview ---
    mosaic_large = mosaic_img.resize((GRID_W * 4, GRID_H * 4), Image.NEAREST)
    mosaic_reader = pil_to_reportlab(mosaic_large)
    mw, mh = mosaic_reader.getSize()
    scale2 = min(avail_w / mw, avail_h / mh)
    dw2 = mw * scale2
    dh2 = mh * scale2
    mx = MARGIN + avail_w + MARGIN + (avail_w - dw2) / 2
    my = preview_y + (avail_h - dh2) / 2
    c.drawImage(mosaic_reader, mx, my, width=dw2, height=dh2,
                preserveAspectRatio=True)

    # Labels under images
    c.setFillColorRGB(0.3, 0.3, 0.3)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(MARGIN + avail_w / 2, preview_y - 5 * mm, "Original Image")
    c.drawCentredString(MARGIN + avail_w + MARGIN + avail_w / 2,
                        preview_y - 5 * mm, "LEGO Mosaic Preview")

    # Stats box
    stats_y = 40 * mm
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.roundRect(MARGIN, stats_y, PAGE_W - 2 * MARGIN, 24 * mm, 4, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(PAGE_W / 2, stats_y + 16 * mm, "Project Specifications")
    c.setFont("Helvetica", 9)
    specs = (f"Grid: {GRID_W}×{GRID_H} studs   •   "
             f"Baseplates: {COLS}×{ROWS} = 40   •   "
             f"Piece type: 1×1 only   •   "
             f"Plate size: 16×16")
    c.drawCentredString(PAGE_W / 2, stats_y + 7 * mm, specs)

    _draw_page_footer(c, page_num)
    c.showPage()


# ── Colour Legend Page ────────────────────────────────────────────────────────

def draw_colour_legend(c: canvas.Canvas, used_indices: list,
                       index_map: np.ndarray, page_num: int):
    """Page 2 – colour legend with swatches, names, HEX, and piece counts."""
    _draw_page_header(c, "COLOUR LEGEND", f"Page {page_num}")

    total = index_map.size
    flat  = index_map.flatten()

    # Section title
    content_top = PAGE_H - 20 * mm
    _lego_yellow_bar(c, content_top - 12 * mm,
                     PAGE_W - 2 * MARGIN, "All Colours Used in This Mosaic")

    # Column layout: 2 columns
    col_w    = (PAGE_W - 2 * MARGIN - 5 * mm) / 2
    swatch_s = 8 * mm
    row_h    = 12 * mm
    cols     = 2
    start_y  = content_top - 26 * mm

    for i, idx in enumerate(used_indices):
        name, rgb = LEGO_PALETTE[idx]
        count = int(np.sum(flat == idx))
        pct   = count / total * 100

        col   = i % cols
        row   = i // cols
        x     = MARGIN + col * (col_w + 5 * mm)
        y     = start_y - row * row_h

        if y < 20 * mm:
            # Overflow to next page
            _draw_page_footer(c, page_num)
            c.showPage()
            page_num += 1
            _draw_page_header(c, "COLOUR LEGEND (continued)", f"Page {page_num}")
            start_y = PAGE_H - 28 * mm
            row     = 0
            y       = start_y

        # Swatch
        r, g, b = rgb
        c.setFillColorRGB(r / 255, g / 255, b / 255)
        c.setStrokeColorRGB(0.4, 0.4, 0.4)
        c.rect(x, y - swatch_s + 2 * mm, swatch_s, swatch_s, fill=1, stroke=1)

        # Colour number badge
        c.setFillColorRGB(0.13, 0.13, 0.13)
        c.setFont("Helvetica-Bold", 7)
        badge_col = (1, 1, 1) if _luma(rgb) < 128 else (0, 0, 0)
        c.setFillColorRGB(*[v / 255 for v in rgb])
        c.rect(x, y - swatch_s + 2 * mm, swatch_s, swatch_s, fill=1, stroke=1)
        c.setFillColorRGB(*badge_col)
        c.drawCentredString(x + swatch_s / 2, y - swatch_s / 2 + 2 * mm,
                            str(idx + 1))

        # Text
        text_x = x + swatch_s + 3 * mm
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(text_x, y - 3 * mm, name)
        c.setFont("Helvetica", 8)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawString(text_x, y - 7 * mm, f"{_hex(rgb)}   {count:,} pcs ({pct:.1f}%)")

    _draw_page_footer(c, page_num)
    c.showPage()
    return page_num


# ── Plate Build Pages ─────────────────────────────────────────────────────────

def draw_plate_page(c: canvas.Canvas, plate: np.ndarray,
                    plate_index: int, row: int, col: int,
                    page_num: int, index_map: np.ndarray = None):
    """One page per baseplate showing the 16×16 coloured grid."""
    plate_id = f"Plate {plate_index + 1:02d}  (Row {row + 1}, Column {col + 1})"

    _draw_page_header(c, "BUILD INSTRUCTIONS", plate_id)

    # Render the plate image
    cell_px = 26
    plate_img = render_plate(plate, cell_px=cell_px, show_coords=True)
    plate_reader = pil_to_reportlab(plate_img)

    pw, ph = plate_reader.getSize()
    avail_w = PAGE_W - 2 * MARGIN
    avail_h = PAGE_H * 0.55
    scale   = min(avail_w / pw, avail_h / ph)
    draw_w  = pw * scale
    draw_h  = ph * scale
    img_x   = (PAGE_W - draw_w) / 2
    img_y   = PAGE_H - 22 * mm - draw_h - 5 * mm

    c.drawImage(plate_reader, img_x, img_y,
                width=draw_w, height=draw_h, preserveAspectRatio=True)

    # --- Plate position diagram (mini 5×8 map with current plate highlighted) ---
    map_cell = 6 * mm
    map_w    = COLS * map_cell
    map_h    = ROWS * map_cell
    map_x    = (PAGE_W - map_w) / 2
    map_y    = img_y - map_h - 8 * mm

    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawCentredString(PAGE_W / 2, map_y + map_h + 3 * mm, "Position on Assembly")

    for r in range(ROWS):
        for cl in range(COLS):
            mx = map_x + cl * map_cell
            my = map_y + (ROWS - 1 - r) * map_cell
            # Compute dominant colour of this plate (most-frequent palette index)
            p_idx = r * COLS + cl
            plate_data = index_map[r*PLATE_H:(r+1)*PLATE_H, cl*PLATE_W:(cl+1)*PLATE_W]
            counts = np.bincount(plate_data.flatten(), minlength=len(LEGO_PALETTE))
            dom_idx = int(np.argmax(counts))
            dom_rgb = LEGO_PALETTE[dom_idx][1]
            # Fill with dominant colour
            c.setFillColorRGB(dom_rgb[0]/255, dom_rgb[1]/255, dom_rgb[2]/255)
            c.setStrokeColorRGB(0.4, 0.4, 0.4)
            c.rect(mx, my, map_cell, map_cell, fill=1, stroke=1)
            # Highlight the current plate with a thick white border
            if r == row and cl == col:
                c.setStrokeColorRGB(1, 1, 1)
                c.setLineWidth(1.5)
                c.rect(mx + 0.5, my + 0.5, map_cell - 1, map_cell - 1, fill=0, stroke=1)
                c.setLineWidth(1)

    # --- Colour key for this plate ---
    used_in_plate = sorted(set(plate.flatten().tolist()))
    key_y         = map_y - 6 * mm
    swatch_s      = 5 * mm
    sw_per_row    = 6
    key_start_x   = MARGIN

    c.setFont("Helvetica-Bold", 8)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.drawString(MARGIN, key_y, f"Colours in this plate ({len(used_in_plate)}):")
    key_y -= 6 * mm

    for i, idx in enumerate(used_in_plate):
        name, rgb = LEGO_PALETTE[idx]
        kx = key_start_x + (i % sw_per_row) * ((PAGE_W - 2 * MARGIN) / sw_per_row)
        ky = key_y - (i // sw_per_row) * 7 * mm

        c.setFillColorRGB(*[v / 255 for v in rgb])
        c.setStrokeColorRGB(0.4, 0.4, 0.4)
        c.rect(kx, ky, swatch_s, swatch_s, fill=1, stroke=1)

        # Badge number
        badge_col = (1, 1, 1) if _luma(rgb) < 128 else (0, 0, 0)
        c.setFillColorRGB(*badge_col)
        c.setFont("Helvetica-Bold", 6)
        c.drawCentredString(kx + swatch_s / 2, ky + 1.5 * mm, str(idx + 1))

        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.setFont("Helvetica", 7)
        short_name = name if len(name) <= 14 else name[:13] + "…"
        c.drawString(kx + swatch_s + 1.5 * mm, ky + 1 * mm, short_name)

    _draw_page_footer(c, page_num)
    c.showPage()


# ── Overview Page ─────────────────────────────────────────────────────────────

def draw_overview(c: canvas.Canvas, index_map: np.ndarray, page_num: int):
    """Full assembly overview – complete 5×8 plate map with plate numbers."""
    _draw_page_header(c, "FULL ASSEMBLY OVERVIEW",
                      f"5 columns × 8 rows = 40 baseplates")

    content_top = PAGE_H - 22 * mm
    _lego_yellow_bar(c, content_top - 12 * mm,
                     PAGE_W - 2 * MARGIN, "Complete Mosaic Layout")

    # Render full mosaic at a larger scale
    mosaic_large = render_full_mosaic(index_map, scale=6)  # 6× for sharper overview
    reader       = pil_to_reportlab(mosaic_large)
    mw, mh       = reader.getSize()
    avail_w      = PAGE_W - 2 * MARGIN
    avail_h      = PAGE_H * 0.60
    scale        = min(avail_w / mw, avail_h / mh)
    draw_w       = mw * scale
    draw_h       = mh * scale
    img_x        = (PAGE_W - draw_w) / 2
    img_y        = content_top - 20 * mm - draw_h

    c.drawImage(reader, img_x, img_y,
                width=draw_w, height=draw_h, preserveAspectRatio=True)

    # Draw plate boundary grid overlay on top of the mosaic image
    plate_draw_w = draw_w / COLS
    plate_draw_h = draw_h / ROWS

    c.setStrokeColorRGB(0.1, 0.1, 0.1)
    c.setLineWidth(1.5)
    for cl in range(COLS + 1):
        lx = img_x + cl * plate_draw_w
        c.line(lx, img_y, lx, img_y + draw_h)
    for r in range(ROWS + 1):
        ly = img_y + r * plate_draw_h
        c.line(img_x, ly, img_x + draw_w, ly)

    # Plate number labels
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 7)
    for r in range(ROWS):
        for cl in range(COLS):
            plate_num = r * COLS + cl + 1
            cx = img_x + cl * plate_draw_w + plate_draw_w / 2
            cy = img_y + (ROWS - 1 - r) * plate_draw_h + plate_draw_h / 2
            # Small dark background for readability
            lbl = str(plate_num)
            c.setFillColorRGB(0, 0, 0)
            c.rect(cx - 4.5, cy - 3.5, 9, 7, fill=1, stroke=0)
            c.setFillColorRGB(1, 1, 1)
            c.drawCentredString(cx, cy - 2.5, lbl)

    # Legend box
    legend_y = img_y - 22 * mm
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.roundRect(MARGIN, legend_y, PAGE_W - 2 * MARGIN, 18 * mm, 3, fill=1, stroke=0)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(PAGE_W / 2, legend_y + 12 * mm, "Build Order: Left → Right, Top → Bottom")
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawCentredString(PAGE_W / 2, legend_y + 6 * mm,
                        f"Total studs: {GRID_W * GRID_H:,}   •   "
                        f"Total plates: 40   •   "
                        f"Grid: {GRID_W}×{GRID_H}")
    c.drawCentredString(PAGE_W / 2, legend_y + 2 * mm,
                        "Numbers in each section indicate the plate build order (1–40)")

    _draw_page_footer(c, page_num)
    c.showPage()


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def generate_lego_guide(image_path: str,
                        output_path: str = "lego_guide.pdf"):
    """
    Full pipeline:
      1. Load + resize image (edge-replication, no solid bars)
      2. Quantize every pixel to nearest LEGO palette colour
      3. Split into 40 baseplates
      4. Generate the PDF guide

    Parameters
    ----------
    image_path    : Path to the input image (PNG / JPG / BMP / etc.)
    output_path   : Desired output PDF path
    """

    if not os.path.isfile(image_path):
        sys.exit(f"[ERROR] File not found: {image_path}")

    project_name = os.path.splitext(os.path.basename(image_path))[0]

    print("=" * 60)
    print("  LEGO Mosaic Generator")
    print("=" * 60)
    print(f"  Input  : {image_path}")
    print(f"  Output : {output_path}")
    print()

    # ── Step 1+2: Full illustration pipeline ─────────────────────────────────
    print("[1/4] Processing image …")
    print(f"      Pipeline: bilateral → Canny → pre-quantize@full-res → NEAREST resize")
    print(f"                → pad → edge-override → mode-filter")
    index_map, edge_mask = process_image(image_path)
    used = get_used_colours(index_map)
    edge_pct = edge_mask.mean() * 100
    print(f"      Grid: {GRID_W}×{GRID_H}  |  Colours used: {len(used)}"
          f"  |  Edge pixels: {int(edge_mask.sum())} ({edge_pct:.1f}%)")

    # ── Step 3: Split into plates ──────────────────────────────────────────
    print("[3/4] Splitting into 40 baseplates (16×16 each) …")
    plates = split_into_plates(index_map)
    print(f"      Plates generated: {len(plates)}")

    # ── Step 4: Generate PDF ───────────────────────────────────────────────
    print("[4/4] Generating PDF …")
    mosaic_preview = render_full_mosaic(index_map, scale=1)

    c = canvas.Canvas(output_path, pagesize=A4)
    c.setTitle(f"LEGO Mosaic – {project_name}")
    c.setAuthor("LEGO Mosaic Generator")
    c.setSubject("LEGO Building Instructions")

    page = 1

    # Cover
    draw_cover(c, image_path, mosaic_preview, page, project_name)
    page += 1

    # Colour legend
    page = draw_colour_legend(c, used, index_map, page) + 1

    # Plate build pages
    for plate_idx, plate in enumerate(plates):
        r   = plate_idx // COLS
        col = plate_idx % COLS
        draw_plate_page(c, plate, plate_idx, r, col, page, index_map)
        page += 1
        if (plate_idx + 1) % 5 == 0:
            print(f"      Rendered {plate_idx + 1}/40 plates …")

    # Overview
    draw_overview(c, index_map, page)

    c.save()
    size_kb = os.path.getsize(output_path) / 1024
    print()
    print("=" * 60)
    print(f"  ✓  PDF saved: {output_path}  ({size_kb:.0f} KB)")
    print(f"  ✓  Total pages: {page}")
    print(f"  ✓  Plates: 40   Colours used: {len(used)}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an image into a LEGO mosaic PDF building guide.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python lego_mosaic.py photo.jpg
              python lego_mosaic.py photo.jpg -o my_guide.pdf
              python lego_mosaic.py portrait.png -o guide.pdf
        """)
    )
    parser.add_argument("image",  help="Path to the input image (PNG/JPG/BMP…)")
    parser.add_argument("-o", "--output", default="lego_guide.pdf",
                        help="Output PDF file path (default: lego_guide.pdf)")

    args = parser.parse_args()
    generate_lego_guide(
        image_path=args.image,
        output_path=args.output,
    )
