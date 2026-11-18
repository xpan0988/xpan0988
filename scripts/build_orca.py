#!/usr/bin/env python3
"""
ASCII orca for a GitHub profile README.

Idea (inspired by OGHong/dsh-ascii-sharks): no image assets. The orca is
defined by math -- a depth profile along a spine, plus fin shapes -- then
rasterised into a character grid every frame. The spine undulates (orcas
beat their flukes up and down), and the frames are baked into one SVG
with a pure-CSS frame loop, because GitHub READMEs cannot run JS.

Outputs:
  assets/orca-dark.svg, assets/orca-light.svg   animated, for <picture>
  assets/orca.txt                                one static frame
  preview/*.png, preview/orca-dark.gif           local previews

Usage: python3 scripts/build_orca.py
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
COLS, ROWS = 150, 42          # character grid
ORCA_COLS = 112               # body length (snout -> fluke tip) in columns
CELL_ASPECT = 2.0             # cell height / cell width (monospace ~ 1:2)
SS = 4                        # supersamples per cell width
FRAMES = 36                   # frames per loop
LOOP_SECONDS = 3.0
SEED = 7

# Body (units: body length L = 1, snout at s = 0)
DEPTH_MAX = 0.235             # max body depth
DEPTH_AT = 0.34               # where the body is deepest
DORSAL_H = 0.20               # tall, upright dorsal fin of an adult male
TAIL_TAPER = 1.25             # how fast the body narrows behind DEPTH_AT
PEDUNCLE_D = 0.055            # floor on the tail-stock depth: stays solid into the flukes
PEDUNCLE_EASE = 0.03          # how softly the taper settles onto PEDUNCLE_D
CAM_ELEV = math.radians(26)   # camera slightly above: flukes read as a shape
BEATS_PER_LOOP = 2
TAIL_AMP = 0.055

# Flukes (top-view planform, offsets along the chord measured from FLUKE_ROOT)
FLUKE_SPAN = 0.26             # half-span; foreshortened on screen by sin(CAM_ELEV)
FLUKE_ROOT = 0.955            # where the flukes hang off the spine
FLUKE_LEAD0 = -0.035          # leading edge at the midline (overlaps the peduncle)
FLUKE_TIP = 0.17              # how far back the pointed tips sit (leading-edge sweep)
FLUKE_NOTCH = 0.060           # trailing edge at the midline: the median notch
FLUKE_LEAD_CURVE = 2.2        # >1: leading edge leaves the peduncle square, then sweeps back
FLUKE_TRAIL_CURVE = 0.7       # <1: trailing edge folds into a sharp cusp at the notch
FLUKE_FOLLOW = 0.80           # fluke pitch = atan(spine slope) * this
FLUKE_NEAR_SHADE = 0.66       # near lobe (below, in front of the body)
FLUKE_FAR_SHADE = 0.44        # far lobe (above, behind the body) reads a little darker
FLUKE_RIM = 0.22              # extra light on the near lobe's leading edge

# Haze
EDGE_DITHER = 0.85            # >0 breaks up partially covered edge cells
TAIL_DITHER = 0.30            # gentler dither from the peduncle on, so the tail stays solid
TAIL_DITHER_S = (0.79, 0.87)  # ramp from EDGE_DITHER to TAIL_DITHER over this range of s
EDGE_GLYPH = 0.6              # share of dissolving edge cells that get a faint glyph
HALO = 0.10                   # share of cells next to the body that get a faint dot
                              # (both picked by a fixed per-cell hash: no per-frame noise)
FRAME_OVERLAP_MS = 0.002      # frames hand over with this overlap: never a blank instant

rng = np.random.default_rng(SEED)

# materials
EMPTY, BLACK, WHITE, GREY, EYE = -1, 0, 1, 2, 3


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def depth(s):
    s = np.asarray(s, dtype=float)
    head = np.sqrt(np.clip(1 - (1 - s / DEPTH_AT) ** 2, 0, 1))
    u = np.clip((s - DEPTH_AT) / (1 - DEPTH_AT), 0, 1)
    tail = np.cos(u * np.pi / 2) ** TAIL_TAPER * DEPTH_MAX
    # soft floor: the tail stock eases into PEDUNCLE_D instead of pinching off
    # (polynomial smooth max: identical to the plain taper once it is more than
    # PEDUNCLE_EASE above the floor, so the rest of the body is untouched)
    gap = np.maximum(PEDUNCLE_EASE - np.abs(tail - PEDUNCLE_D), 0)
    tail = np.maximum(tail, PEDUNCLE_D) + gap ** 2 / (4 * PEDUNCLE_EASE)
    d = np.where(s < DEPTH_AT, head * DEPTH_MAX, tail)
    return np.maximum(d, 0.032 * (s > 0.02))


def split(s):
    """fraction of depth above the spine (back) vs below (belly)."""
    s = np.asarray(s, dtype=float)
    # rounded melon sits a bit lower at the very front; belly is fuller mid-body
    return 0.50 - 0.06 * np.exp(-((s - 0.04) / 0.07) ** 2) + 0.03 * np.exp(-((s - 0.5) / 0.2) ** 2)


def spine(s, t):
    """vertical offset and slope of the spine at s, time t in [0,1)."""
    amp = 0.006 + TAIL_AMP * np.clip((s - 0.35) / 0.65, 0, 1) ** 2.2
    k = 2 * np.pi * 0.85                       # wave number along the body
    ph = 2 * np.pi * BEATS_PER_LOOP * t
    y = amp * np.sin(ph - k * s)
    ds = 1e-3
    amp2 = 0.006 + TAIL_AMP * np.clip((s + ds - 0.35) / 0.65, 0, 1) ** 2.2
    y2 = amp2 * np.sin(ph - k * (s + ds))
    return y, (y2 - y) / ds


def frame_of(s, t):
    """position on spine (x,y) and unit normal pointing up."""
    y, dy = spine(s, t)
    n = np.stack([-dy, np.ones_like(dy)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.stack([s, y], -1), n


def point_in_poly(px, py, poly):
    x, y = poly[:, 0], poly[:, 1]
    inside = np.zeros(px.shape, bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        cond = ((y[i] > py) != (y[j] > py)) & (
            px < (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i] + 1e-12) + x[i])
        inside ^= cond
        j = i
    return inside


@dataclass
class Samples:
    x: np.ndarray
    y: np.ndarray
    mat: np.ndarray
    shade: np.ndarray
    layer: np.ndarray


LIGHT = np.array([-0.35, 0.75, 0.55])
LIGHT /= np.linalg.norm(LIGHT)


def body_samples(t) -> Samples:
    s = np.linspace(0.0, 0.97, 1400)
    q = np.linspace(-1, 1, 320)
    S, Q = np.meshgrid(s, q, indexing="ij")
    d = depth(S)
    f = split(S)
    up, dn = d * f, d * (1 - f)
    v = np.where(Q >= 0, Q * up, Q * dn)
    pos, n = frame_of(S, t)
    X = pos[..., 0] + n[..., 0] * v
    Y = pos[..., 1] + n[..., 1] * v

    # pseudo-3D normal: elliptical cross-section + profile slope near ends
    nz = np.sqrt(np.clip(1 - Q ** 2, 0, 1))
    dd = np.gradient(depth(s))[:, None] / (s[1] - s[0])
    nx = -0.35 * np.clip(dd, -2, 2) * nz
    N = np.stack([nx, Q, nz], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    lam = np.clip(N @ LIGHT, 0, 1)
    spec = np.clip(N @ (LIGHT + np.array([0, 0, 1])) / np.linalg.norm(LIGHT + np.array([0, 0, 1])), 0, 1) ** 14
    shade = np.clip(0.08 + 0.62 * lam + 0.55 * spec, 0, 1)

    mat = np.full(S.shape, BLACK)
    # white lower jaw / throat / belly
    belly_edge = np.interp(S, [0.0, 0.03, 0.12, 0.30, 0.45, 0.55],
                              [-0.40, -0.15, -0.42, -0.68, -0.82, -0.98])
    mat[(Q < belly_edge) & (S < 0.55)] = WHITE
    # flank patch sweeping up behind the dorsal fin, pointing to the tail
    flank_top = -0.75 + 0.95 * np.exp(-((S - 0.60) / 0.055) ** 2) \
        + 0.35 * np.clip((S - 0.60) / 0.10, 0, 1) * np.exp(-((S - 0.66) / 0.05) ** 2)
    flank_bot = -1.1 + 0.6 * np.clip((S - 0.70) / 0.10, 0, 1)
    flank = (S > 0.48) & (S < 0.79) & (Q < flank_top) & (Q > flank_bot)
    mat[flank] = WHITE
    # eye patch: white oval above and behind the eye
    ex = (S - 0.150) / 0.042
    ey = (Q - 0.36) / 0.15
    rot = -0.35
    exr = ex * math.cos(rot) - ey * math.sin(rot)
    eyr = ex * math.sin(rot) + ey * math.cos(rot)
    mat[exr ** 2 + eyr ** 2 < 1] = WHITE
    # grey saddle behind the dorsal fin
    mat[(S > 0.45) & (S < 0.60) & (Q > 0.62 - 0.4 * np.exp(-((S - 0.52) / 0.04) ** 2) * 0.5)] = GREY
    # eye
    mat[((S - 0.085) / 0.008) ** 2 + ((Q - 0.02) / 0.07) ** 2 < 1] = EYE

    return Samples(X.ravel(), Y.ravel(), mat.ravel(), shade.ravel(),
                   np.full(X.size, 1))


def dorsal_samples(t) -> Samples:
    # fin profile in (s, height above back) -- straight leading edge,
    # slightly concave trailing edge, rounded-ish tip
    tip = (0.425, DORSAL_H)
    lead = [(0.355, 0.0), (0.38, 0.06), (0.40, 0.12), tip]
    trail = [(0.432, DORSAL_H * 0.96), (0.44, 0.13), (0.46, 0.06), (0.50, 0.012), (0.52, 0.0)]
    poly = np.array(lead + trail + [(0.50, -0.02), (0.36, -0.02)])
    s = np.linspace(0.34, 0.53, 260)
    h = np.linspace(-0.02, DORSAL_H, 260)
    S, H = np.meshgrid(s, h, indexing="ij")
    keep = point_in_poly(S, H, poly)
    S, H = S[keep], H[keep]
    pos, n = frame_of(S, t)
    top = depth(S) * split(S)
    v = top + H
    X = pos[:, 0] + n[:, 0] * v
    Y = pos[:, 1] + n[:, 1] * v
    # lit along the leading edge, darker toward the trailing edge
    rel = (S - 0.355 - H * (0.07 / DORSAL_H)) / 0.12
    shade = np.clip(0.55 - 0.35 * rel + 0.15 * (H / DORSAL_H), 0.05, 1)
    mat = np.full(S.shape, BLACK)
    return Samples(X, Y, mat, shade, np.full(S.size, 1))


def pectoral_samples(t) -> Samples:
    # paddle-shaped fin on the near side, pointing down and back
    root_s = 0.205
    pos, n = frame_of(np.array([root_s]), t)
    root = pos[0] - n[0] * depth(root_s) * (1 - split(root_s)) * 0.55
    ang = math.radians(-58)
    dirv = np.array([math.cos(ang), math.sin(ang)])
    side = np.array([-dirv[1], dirv[0]])
    length, width = 0.16, 0.058
    a = np.linspace(0, 1, 220)
    b = np.linspace(-1, 1, 120)
    A, B = np.meshgrid(a, b, indexing="ij")
    # paddle: narrow at the root, broad and round at the tip
    half = width * (0.35 + 0.65 * np.sin(np.pi * np.clip(A * 0.62 + 0.18, 0, 1)) ** 0.8) * np.sqrt(np.clip(1 - A ** 6, 0, 1))
    P = root + A[..., None] * length * dirv + (B * half)[..., None] * side
    shade = np.clip(0.35 + 0.35 * (-B) + 0.1 * (1 - A), 0.05, 1)
    return Samples(P[..., 0].ravel(), P[..., 1].ravel(), np.full(A.size, BLACK),
                   shade.ravel(), np.full(A.size, 2))


def fluke_samples(t) -> Samples:
    pos, _ = frame_of(np.array([FLUKE_ROOT]), t)
    _, slope = spine(np.array([FLUKE_ROOT]), t)
    ang = math.atan(slope[0]) * FLUKE_FOLLOW     # flukes lag/overshoot the spine
    tdir = np.array([math.cos(ang), math.sin(ang)])
    z = np.linspace(-FLUKE_SPAN, FLUKE_SPAN, 320)
    c = np.linspace(0, 1, 160)
    Z, C = np.meshgrid(z, c, indexing="ij")
    az = np.abs(Z) / FLUKE_SPAN
    # planform seen from above: the leading edge leaves the peduncle almost
    # square, then sweeps back into a pointed tip; the trailing edge runs from
    # the tip forward to a deep median notch
    lead = FLUKE_LEAD0 + (FLUKE_TIP - FLUKE_LEAD0) * az ** FLUKE_LEAD_CURVE
    trail = FLUKE_NOTCH + (FLUKE_TIP - FLUKE_NOTCH) * az ** FLUKE_TRAIL_CURVE
    chord = trail - lead
    along = lead + C * chord
    across = Z * math.sin(CAM_ELEV)                      # foreshortened span
    # flukes pitch about their span axis: only the chord turns with the spine,
    # the span stays vertical on screen, so the lobes do not shear when they beat
    P = pos[0] + along[..., None] * tdir + across[..., None] * np.array([0.0, 1.0])
    far = Z > 0
    # leading edge catches the light (a bright rim on the near lobe separates
    # it from the peduncle); tips and the far lobe sit darker
    shade = np.where(far, FLUKE_FAR_SHADE, FLUKE_NEAR_SHADE) + 0.22 * (1 - C) - 0.10 * az
    shade = shade + np.where(far, 0.0, FLUKE_RIM * np.exp(-(C / 0.25) ** 2))
    layer = np.where(far, 0, 2)
    return Samples(P[..., 0].ravel(), P[..., 1].ravel(), np.full(Z.size, BLACK),
                   np.clip(shade, 0, 1).ravel(), layer.ravel())


# --------------------------------------------------------------------------
# Rasterise to a character grid
# --------------------------------------------------------------------------
GW, GH = COLS * SS, int(ROWS * SS * CELL_ASPECT)
SCALE = ORCA_COLS * SS / 1.10                          # subpixels per body length
OX = (COLS - ORCA_COLS) / 2 * SS + 0.07 * SCALE        # snout x in subpixels
OY = GH * 0.50                                         # spine y in subpixels


def rasterise(t):
    parts = [fluke_samples(t), body_samples(t), dorsal_samples(t), pectoral_samples(t)]
    x = np.concatenate([p.x for p in parts])
    y = np.concatenate([p.y for p in parts])
    mat = np.concatenate([p.mat for p in parts])
    shade = np.concatenate([p.shade for p in parts])
    layer = np.concatenate([p.layer for p in parts])

    bob = 0.006 * math.sin(2 * math.pi * BEATS_PER_LOOP * t + 1.2)
    px = np.floor(OX + x * SCALE).astype(int)
    py = np.floor(OY - (y + bob) * SCALE).astype(int)
    ok = (px >= 0) & (px < GW) & (py >= 0) & (py < GH)
    px, py, mat, shade, layer = px[ok], py[ok], mat[ok], shade[ok], layer[ok]
    order = np.lexsort((mat == EYE, layer))             # later = on top
    px, py, mat, shade = px[order], py[order], mat[order], shade[order]

    M = np.full((GH, GW), EMPTY, int)
    SH = np.zeros((GH, GW))
    M[py, px] = mat
    SH[py, px] = shade

    ch = int(SS * CELL_ASPECT)
    Mc = M.reshape(ROWS, ch, COLS, SS).transpose(0, 2, 1, 3).reshape(ROWS, COLS, -1)
    Sc = SH.reshape(ROWS, ch, COLS, SS).transpose(0, 2, 1, 3).reshape(ROWS, COLS, -1)
    covered = Mc != EMPTY
    cov = covered.mean(-1)
    shade_c = np.where(covered, Sc, 0).sum(-1) / np.maximum(covered.sum(-1), 1)
    counts = np.stack([(Mc == m).sum(-1) for m in (BLACK, WHITE, GREY, EYE)], -1)
    mat_c = np.argmax(counts, -1)
    mat_c[counts[..., 3] > 0] = EYE
    mat_c[cov == 0] = EMPTY
    return mat_c, shade_c, cov


# --------------------------------------------------------------------------
# Glyphs
# --------------------------------------------------------------------------
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0
# glyph families: density stays similar, colour carries black/white/grey and
# shading; the mix of letters and punctuation gives the texture
BODY = "#%&@$8BWMNQ#%&@$"
SKIN = "#%*&+@=$x8?"
LIGHT_SET = ".:'`,-"
STYLES = ["b0", "b1", "b2", "b3", "w0", "w1", "g0", "e0", "h0", "p0", "p1", "f0"]


def cell_hash(x, y, k=0):
    h = (x * 374761393 + y * 668265263 + k * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return h ^ (h >> 16)


def hash01(x, y, k):
    return cell_hash(x, y, k) / 0xFFFFFFFF


def glyphs(mat, shade, cov):
    """return grid of (char, style) or None.

    Every choice is a fixed function of the cell, never of the frame: a cell
    only changes when the geometry under it changes, so the texture holds
    still and all motion comes from the model."""
    out = [[None] * COLS for _ in range(ROWS)]
    # body position s of each column; dither eases off toward the tail
    s_col = ((np.arange(COLS) + 0.5) * SS - OX) / SCALE
    ramp = np.clip((s_col - TAIL_DITHER_S[0]) / (TAIL_DITHER_S[1] - TAIL_DITHER_S[0]), 0, 1)
    dither = EDGE_DITHER + (TAIL_DITHER - EDGE_DITHER) * ramp
    for yy in range(ROWS):
        for xx in range(COLS):
            m = mat[yy, xx]
            c = cov[yy, xx]
            if m == EMPTY:
                continue
            h = cell_hash(xx, yy)
            thr = BAYER[yy % 4, xx % 4] * dither[xx]
            if c < 0.999 and c < thr + 0.10:
                # dissolving edge: sparse light glyph or nothing
                if hash01(xx, yy, 1) < EDGE_GLYPH:
                    out[yy][xx] = (LIGHT_SET[h % len(LIGHT_SET)], "h0")
                continue
            s = shade[yy, xx]
            if m == EYE:
                out[yy][xx] = ("@", "e0")
                continue
            if m == BLACK:
                lvl = 0 if s < 0.40 else 1 if s < 0.64 else 2 if s < 0.95 else 3
                fam = SKIN if lvl == 3 else BODY
                cell = (fam[h % len(fam)], f"b{lvl}")
            elif m == WHITE:
                cell = (BODY[h % len(BODY)], "w1" if s > 0.4 else "w0")
            else:
                cell = (SKIN[h % len(SKIN)], "g0")
            out[yy][xx] = cell
    # faint halo
    body = mat != EMPTY
    near = np.zeros_like(body)
    for dy in (-1, 0, 1):
        for dx in (-2, -1, 0, 1, 2):
            near |= np.roll(np.roll(body, dy, 0), dx, 1)
    for yy, xx in zip(*np.nonzero(near & ~body)):
        if out[yy][xx] is None and hash01(xx, yy, 2) < HALO:
            out[yy][xx] = (".", "h0")
    return out


# ambient water: bubbles rising from the blowhole, drifting particles, a small
# milling bait ball ahead of the orca. Everything is periodic in t so the loop
# is seamless.
N_PART = 46
PART = dict(
    x=rng.uniform(0, COLS, N_PART), y=rng.uniform(0, ROWS, N_PART),
    ph=rng.uniform(0, 1, N_PART), amp=rng.uniform(0.6, 2.2, N_PART),
    ch=rng.choice(list(".'`,·"), N_PART),
)
N_BUB = 9
BUB = dict(ph=rng.uniform(0, 1, N_BUB), dx=rng.uniform(-3, 3, N_BUB),
           ch=rng.choice(list("oO°.o"), N_BUB))
N_FISH = 14
FISH = dict(ph=rng.uniform(0, 1, N_FISH), rx=rng.uniform(3, 7, N_FISH),
            ry=rng.uniform(1.0, 2.6, N_FISH), cx=rng.normal(0, 0.8, N_FISH))


def ambient(grid, t):
    def put(x, y, ch, st):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < COLS and 0 <= yi < ROWS and grid[yi][xi] is None:
            grid[yi][xi] = (ch, st)

    for i in range(N_PART):
        a = 2 * np.pi * (t + PART["ph"][i])
        alpha = 0.5 + 0.5 * math.sin(a)
        if alpha > 0.45:
            put(PART["x"][i] + PART["amp"][i] * math.cos(a), PART["y"][i] + 0.4 * math.sin(2 * a),
                PART["ch"][i], "p1" if alpha > 0.85 else "p0")
    # blowhole is just behind the melon on the back
    bx = (OX + 0.20 * SCALE) / SS
    by = (OY - 0.11 * SCALE) / (SS * CELL_ASPECT)
    for i in range(N_BUB):
        p = (t + BUB["ph"][i]) % 1.0
        y = by - p * (by + 1)
        x = bx + BUB["dx"][i] * p + 0.8 * math.sin(2 * np.pi * (p * 2 + i * 0.3))
        put(x, y, BUB["ch"][i] if p > 0.15 else ".", "p1" if p < 0.6 else "p0")
    # bait ball ahead of the snout, milling in a loop
    fx = (OX - 0.04 * SCALE) / SS - 11
    fy = ROWS * 0.70
    for i in range(N_FISH):
        a = 2 * np.pi * (t + FISH["ph"][i])
        x = fx + FISH["cx"][i] + FISH["rx"][i] * math.cos(a)
        y = fy + FISH["ry"][i] * math.sin(a)
        put(x, y, "<" if math.sin(a) > 0 else ">", "f0")
    return grid


def build_frames():
    frames = []
    for fi in range(FRAMES):
        t = fi / FRAMES
        mat, shade, cov = rasterise(t)
        g = glyphs(mat, shade, cov)
        frames.append(ambient(g, t))
    return frames


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
THEMES = {
    "dark": {
        "b0": "#344255", "b1": "#45566c", "b2": "#5b6f88", "b3": "#93a8c0",
        "w0": "#aebccb", "w1": "#e6edf3", "g0": "#58687c", "e0": "#e6edf3",
        "h0": "#2c3744", "p0": "#24344a", "p1": "#3f5a78", "f0": "#4f7fa3",
    },
    "light": {
        "b0": "#0d1117", "b1": "#1f2328", "b2": "#353c45", "b3": "#5b6570",
        "w0": "#d4dae0", "w1": "#e8ecf0", "g0": "#8c959f", "e0": "#f6f8fa",
        "h0": "#c9d1d9", "p0": "#d5dee8", "p1": "#a9bccf", "f0": "#6a8aa6",
    },
}
CW, CH = 6.0, 12.0      # px per cell in the SVG; glyphs are placed per column


def esc(c):
    return {"<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&apos;"}.get(c, c)


def to_svg(frames, theme):
    th = THEMES[theme]
    W, H = COLS * CW, ROWS * CH
    dur = LOOP_SECONDS
    n = len(frames)
    css = [
        "text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'DejaVu Sans Mono',monospace;"
        f"font-size:{CH * 0.82:.1f}px;white-space:pre}}",
        ".f{opacity:0;animation:k %.6fs steps(1,end) infinite}.fr0{opacity:1}" % dur,
        # each frame owns exactly 1/n of the loop plus a hair of overlap, so the
        # hand-over never leaves a frame-less (blank) instant. Windows are
        # shifted early (see the delays below) so that at t=0, where browsers
        # hold animations while they start up, frame 0 is the only one showing.
        "@keyframes k{0%%{opacity:1}%.6f%%{opacity:0}100%%{opacity:0}}"
        % (100 / n + 100 * FRAME_OVERLAP_MS / (dur * 1000)),
        "@media (prefers-reduced-motion:reduce){.f{animation:none}}",
    ]
    for st in STYLES:
        css.append(f".{st}{{fill:{th[st]}}}")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
           f'width="{W:.0f}" height="{H:.0f}" role="img" aria-label="ASCII orca swimming">',
           "<style>" + "".join(css) + "</style>"]
    for fi, g in enumerate(frames):
        cls = "f fr0" if fi == 0 else "f"
        # window of frame fi: [fi*slot - 2e, (fi+1)*slot - e): neighbours overlap
        # by e and no window edge lands exactly on a slot boundary (or t=0)
        delay = fi * dur / n - dur - 2 * FRAME_OVERLAP_MS / 1000
        out.append(f'<g class="{cls}" style="animation-delay:{delay:.7f}s">')
        for yy, row in enumerate(g):
            # contiguous same-style glyphs become one <text>; textLength pins
            # the run to the grid so columns line up in any monospace font
            ybase = (yy + 0.78) * CH
            xx = 0
            while xx < COLS:
                cell = row[xx]
                if not cell:
                    xx += 1
                    continue
                st, x0, buf = cell[1], xx, []
                while xx < COLS and row[xx] and row[xx][1] == st:
                    buf.append(esc(row[xx][0]))
                    xx += 1
                n_ch = xx - x0
                attrs = f'class="{st}" x="{x0 * CW:g}" y="{ybase:.0f}"'
                if n_ch > 1:
                    # lengthAdjust defaults to "spacing": only gaps stretch
                    attrs += f' textLength="{n_ch * CW - 1:g}"'
                out.append(f"<text {attrs}>{''.join(buf)}</text>")
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


def to_txt(g):
    return "\n".join("".join(c[0] if c else " " for c in row).rstrip() for row in g) + "\n"


PREVIEW_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/Library/Fonts/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


def preview_font(size):
    from PIL import ImageFont
    for path in PREVIEW_FONTS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def preview_png(g, theme, path, scale=1.0):
    from PIL import Image, ImageDraw
    th = THEMES[theme]
    bg = (13, 17, 23) if theme == "dark" else (255, 255, 255)
    cw, chh = int(CW * scale), int(CH * scale)
    img = Image.new("RGB", (COLS * cw, ROWS * chh), bg)
    d = ImageDraw.Draw(img)
    font = preview_font(int(chh * 0.82))
    for yy, row in enumerate(g):
        for xx, cell in enumerate(row):
            if cell:
                d.text((xx * cw, yy * chh), cell[0], fill=th[cell[1]], font=font)
    if path:
        img.save(path)
    return img


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frames = build_frames()
    for theme in THEMES:
        with open(os.path.join(root, "assets", f"orca-{theme}.svg"), "w") as fh:
            fh.write(to_svg(frames, theme))
    with open(os.path.join(root, "assets", "orca.txt"), "w") as fh:
        fh.write(to_txt(frames[0]))
    want_preview = "--no-preview" not in sys.argv
    if want_preview:
        try:
            import PIL  # noqa: F401  (previews only)
        except ImportError:
            print("Pillow not installed: skipping preview/")
            want_preview = False
    if want_preview:
        pv = os.path.join(root, "preview")
        os.makedirs(pv, exist_ok=True)
        preview_png(frames[0], "light", os.path.join(pv, "still-light.png"))
        imgs = [preview_png(f, "dark", None) for f in frames]
        imgs[0].save(os.path.join(pv, "orca-dark.gif"), save_all=True, append_images=imgs[1:],
                     duration=int(LOOP_SECONDS * 1000 / FRAMES), loop=0)
    for theme in THEMES:
        p = os.path.join(root, "assets", f"orca-{theme}.svg")
        print(theme, f"{os.path.getsize(p) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
