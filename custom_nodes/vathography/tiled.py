# SPDX-License-Identifier: GPL-3.0-or-later
"""
Vathography Tiled Depth: a Marigold V2 depth map at the photo's own resolution.

One global pass sees the whole photo (at most `global_long_edge` px, never
enlarged) and fixes the overall shape. Overlapping tiles cut from the photo at
full resolution add the detail. Each tile's prediction is matched to the global
one by a robust scale-and-offset fit (Marigold's depth is only known up to
that), the tiles are blended with raised-cosine weights across their overlaps,
and the result keeps the global map's large-scale shape with the tiles' fine
detail on top.

Every model input is a multiple of 16 px by construction (no VAE crop, no
padding by the model). Nothing is enlarged and nothing is padded: a side
shorter than the tile size gets tiles of that side's length (rounded down to a
multiple of 16), and the last tile on each axis sits flush with the edge.

Inference is the official Marigold V2 template's graph, run per image: VAE
encode, one euler step over sigmas 0.5 -> 0 with no noise and the basic guider,
VAE decode, channel mean. Our own code, written for this project.
"""

import math
import os

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.samplers
import comfy.utils
import nodes
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_EmptyNoise, SamplerCustomAdvanced

MULTIPLE = 16
SIGMAS = (0.5, 0.0)
CLAMP_EPS = 0.002        # decoded values within this of 0 or 1 hit the VAE's clamp: left out of the fit
FIT_MARGIN = 1 / 16      # each tile's border strip left out of the fit (the model is least sure there)
FIT_STEP = 4             # fit on every 4th pixel each way: plenty, and fast
DETAIL_SIGMA = 64.0      # shapes larger than ~this many global pixels come from the global map, smaller from
                         # the tiles. Small values (4) leave rings round things a tile places at another depth
                         # than the global pass (fish in a dark aquarium); 64 doesn't, and shows no seams.


def say(*a):
    print(*a, flush=True)


def floor16(v):
    return max(MULTIPLE, int(v) // MULTIPLE * MULTIPLE)


# ----------------------------------------------------------------------------- plan

def global_size(w, h, long_edge):
    """The global pass's input size: the photo's shape at most `long_edge` px long, never larger
    than the photo, each side a multiple of 16."""
    s = min(1.0, long_edge / max(w, h))
    return min(floor16(w), floor16(round(w * s / MULTIPLE) * MULTIPLE)), min(floor16(h), floor16(round(h * s / MULTIPLE) * MULTIPLE))


def axis_plan(length, tile, overlap):
    """(tile length, start positions) along one axis. A side shorter than `tile` gets one tile of
    its own length (rounded down to 16) or, if that leaves rows over, two flush with each end."""
    t = min(tile, floor16(length))
    if length <= t:
        return t, [0]
    n = max(2, math.ceil((length - overlap) / (t - overlap)))
    return t, [round(i * (length - t) / (n - 1)) for i in range(n)]


def plan(w, h, tile, overlap, global_long_edge):
    gw, gh = global_size(w, h, global_long_edge)
    tiles = []
    if max(w, h) > global_long_edge:                     # otherwise the global pass is already full size
        tw, xs = axis_plan(w, tile, overlap)
        th, ys = axis_plan(h, tile, overlap)
        tiles = [(x, y, tw, th) for y in ys for x in xs]
    return (gw, gh), tiles


# ----------------------------------------------------------------------------- inference

def predict(model, vae, conditioning, image):
    """One Marigold V2 depth prediction: image [1, h, w, 3] (0-1, h and w multiples of 16) ->
    [h, w] float32 on the CPU, the decoded channel mean (low = near). Out of GPU memory, it frees
    ComfyUI's cache and tries once more (memory left over from the previous job is the usual cause)."""
    try:
        return _predict(model, vae, conditioning, image)
    except comfy.model_management.OOM_EXCEPTION:
        say(f"[vathography tiled] out of GPU memory at {image.shape[2]}x{image.shape[1]}: freeing the cache, trying again")
        comfy.model_management.soft_empty_cache(True)
        try:
            return _predict(model, vae, conditioning, image)
        except comfy.model_management.OOM_EXCEPTION as e:
            comfy.model_management.soft_empty_cache(True)
            raise RuntimeError(f"out of GPU memory on a {image.shape[2]}x{image.shape[1]} pass. Lower global_long_edge "
                               f"(and tile) on this node, or use a GPU with more memory") from e


def _predict(model, vae, conditioning, image):
    latent = nodes.VAEEncode().encode(vae, image)[0]
    guider = Guider_Basic(model)
    guider.set_conds(conditioning)
    out = SamplerCustomAdvanced.execute(Noise_EmptyNoise(), guider, comfy.samplers.sampler_object("euler"),
                                        torch.tensor(SIGMAS), latent).result[0]
    dec = nodes.VAEDecode().decode(vae, out)[0]
    if dec.ndim == 5:
        dec = dec.reshape(-1, *dec.shape[-3:])
    return dec[0].float().mean(dim=-1).cpu()


# ----------------------------------------------------------------------------- merge

def resize(x, h, w, mode):
    """[h0, w0] float tensor -> [h, w]; area for shrinking, bicubic for growing."""
    if x.shape == (h, w):
        return x
    return F.interpolate(x[None, None], size=(h, w), mode=mode, **({} if mode == "area" else {"align_corners": False}))[0, 0]


def fit_affine(p, g, iters=8):
    """Robust a, b with a * p + b ~ g (Huber IRLS) over pixels that aren't clamped, on a
    subsample away from the tile's border. Returns None when there's too little to fit."""
    h, w = p.shape
    my, mx = int(h * FIT_MARGIN), int(w * FIT_MARGIN)
    p = p[my:h - my:FIT_STEP, mx:w - mx:FIT_STEP].reshape(-1).double()
    g = g[my:h - my:FIT_STEP, mx:w - mx:FIT_STEP].reshape(-1).double()
    ok = (p > CLAMP_EPS) & (p < 1 - CLAMP_EPS) & (g > CLAMP_EPS) & (g < 1 - CLAMP_EPS)
    p, g = p[ok], g[ok]
    if p.numel() < 1000 or float(p.std()) < 1e-4:
        return None
    wts = torch.ones_like(p)
    a = b = None
    for _ in range(iters):
        sw = wts.sum()
        mp, mg = (wts * p).sum() / sw, (wts * g).sum() / sw
        var = (wts * (p - mp) ** 2).sum()
        a = float((wts * (p - mp) * (g - mg)).sum() / var)
        b = float(mg - a * mp)
        r = (a * p + b - g).abs()
        k = 1.345 * max(float(r.median()) / 0.6745, 1e-6)    # Huber threshold from a robust scale
        wts = torch.where(r <= k, torch.ones_like(r), k / r)
    return (a, b) if a > 0 else None


def ramp(n, lo, hi):
    """Weights along one tile axis: 1 inside, a raised-cosine rise over `lo` px at the start and
    a fall over `hi` px at the end (0 where there's no neighbour on that side)."""
    w = torch.ones(n)
    for k, length in ((0, lo), (1, hi)):
        if length > 0:
            t = (torch.arange(length) + 0.5) / length
            r = 0.5 - 0.5 * torch.cos(math.pi * t)
            if k == 0:
                w[:length] = torch.minimum(w[:length], r)
            else:
                w[n - length:] = torch.minimum(w[n - length:], r.flip(0))
    return w


def gaussian_blur(x, sigma):
    if sigma <= 0.3:
        return x
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / sigma) ** 2)
    k = (k / k.sum()).to(x)
    y = F.pad(x[None, None], (r, r, 0, 0), mode="replicate")
    y = F.conv2d(y, k.view(1, 1, 1, -1))
    y = F.pad(y, (0, 0, r, r), mode="replicate")
    return F.conv2d(y, k.view(1, 1, -1, 1))[0, 0]


def merge(gpred, tiles, preds, w, h, detail_sigma=DETAIL_SIGMA, log=say):
    """The global prediction and the tiles' -> one [h, w] map (same scale as the global one)."""
    G = resize(gpred, h, w, "bicubic")
    if not tiles:
        return G, []
    acc, wsum, stats = torch.zeros(h, w), torch.zeros(h, w), []
    xs = sorted({x for x, _, _, _ in tiles})
    ys = sorted({y for _, y, _, _ in tiles})
    for (x, y, tw, th), p in zip(tiles, preds):
        ab = fit_affine(p, G[y:y + th, x:x + tw])
        stats.append(ab)
        log(f"[vathography tiled] tile at {x},{y}: " + ("no fit" if ab is None else f"scale {ab[0]:.3f} offset {ab[1]:+.3f}"))
        if ab is None:
            log(f"[vathography tiled] tile at {x},{y}: nothing to fit (flat or clamped); the global map is used there")
            continue
        i, j = xs.index(x), ys.index(y)
        ov = lambda pos, k, size: max(0, pos[k] + size - pos[k + 1]) if k + 1 < len(pos) else 0
        wx = ramp(tw, ov(xs, i - 1, tw) if i else 0, ov(xs, i, tw))
        wy = ramp(th, ov(ys, j - 1, th) if j else 0, ov(ys, j, th))
        wt = wy[:, None] * wx[None, :]
        acc[y:y + th, x:x + tw] += wt * (ab[0] * p + ab[1])
        wsum[y:y + th, x:x + tw] += wt
    M = torch.where(wsum > 1e-6, acc / wsum.clamp(min=1e-6), G)
    del acc, wsum
    sigma = detail_sigma * max(w, h) / max(gpred.shape[1], gpred.shape[0])
    return M - gaussian_blur(M, sigma) + gaussian_blur(G, sigma), stats


# ----------------------------------------------------------------------------- the node

def run(image, model, vae, conditioning, mode="full", tile=2048, overlap=512, global_long_edge=2496,
        detail_sigma=DETAIL_SIGMA, log=say):
    """image [1, H, W, C] -> (depth [H, W], 0-1 with near = bright, info dict)."""
    img = image[:1, :, :, :3]
    h, w = img.shape[1], img.shape[2]
    (gw, gh), tiles = plan(w, h, tile, overlap, global_long_edge)
    if mode == "draft":
        tiles = []
    log(f"[vathography tiled] photo {w}x{h} | global {gw}x{gh} | {len(tiles)} tile(s)"
        + (f" of {tiles[0][2]}x{tiles[0][3]}" if tiles else "") + f" | mode {mode}")
    pbar = comfy.utils.ProgressBar(1 + len(tiles))
    chw = img.movedim(-1, 1)
    gin = (F.interpolate(chw, size=(gh, gw), mode="area") if (gh, gw) != (h, w) else chw).movedim(1, -1)
    gpred = predict(model, vae, conditioning, gin)
    pbar.update(1)
    preds = []
    for x, y, tw, th in tiles:
        comfy.model_management.throw_exception_if_processing_interrupted()
        preds.append(predict(model, vae, conditioning, img[:, y:y + th, x:x + tw, :]))
        pbar.update(1)
    dump = os.environ.get("VATHOGRAPHY_TILED_DUMP")         # a folder: keep the raw predictions, to tune the merge offline
    if dump:
        os.makedirs(dump, exist_ok=True)
        path = os.path.join(dump, f"tiled_{w}x{h}_{len(tiles)}t_{tile}.pt")
        torch.save({"global": gpred, "tiles": tiles, "preds": preds, "w": w, "h": h}, path)
        log(f"[vathography tiled] raw predictions saved to {path}")
    raw, stats = merge(gpred, tiles, preds, w, h, detail_sigma, log)
    lo, hi = float(raw.min()), float(raw.max())
    depth = ((hi - raw) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)          # near = bright, like MarigoldV2PostProcess
    fits = [s for s in stats if s is not None]
    log(f"[vathography tiled] merged; tile scale {min(a for a, _ in fits):.3f}-{max(a for a, _ in fits):.3f}" if fits
        else "[vathography tiled] global pass only")
    return depth, {"global": [gw, gh], "tiles": len(tiles), "unfitted": sum(s is None for s in stats)}


class VathographyTiledDepth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The photo, at its own size (one image)."}),
                "model": ("MODEL", {"tooltip": "Marigold V2 depth model after ModelSamplingAuraFlow (shift 1.73, img_to_img_velocity), as in the template."}),
                "vae": ("VAE",),
                "conditioning": ("CONDITIONING", {"tooltip": "The Marigold V2 depth conditioning (ConditioningLoader)."}),
                "mode": (["full", "draft"], {"tooltip": "full: global pass + full-resolution tiles. draft: the global pass alone (fast)."}),
            },
            "optional": {
                "tile": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": MULTIPLE, "advanced": True,
                                 "tooltip": "Tile size in px (a multiple of 16)."}),
                "overlap": ("INT", {"default": 512, "min": 64, "max": 2048, "step": MULTIPLE, "advanced": True,
                                    "tooltip": "Least overlap between neighbouring tiles, px."}),
                "global_long_edge": ("INT", {"default": 2496, "min": 512, "max": 4096, "step": MULTIPLE, "advanced": True,
                                             "tooltip": "The global pass's long edge, px (never above the photo's own)."}),
                "detail_sigma": ("FLOAT", {"default": DETAIL_SIGMA, "min": 0.0, "max": 64.0, "step": 0.5, "advanced": True,
                                           "tooltip": "Shapes larger than about this many global-pass pixels come from the global map, finer ones from the tiles."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "estimate"
    CATEGORY = "vathography"
    DESCRIPTION = ("Marigold V2 depth at the photo's own resolution: a global pass plus full-resolution tiles. "
                   "Near = bright, 0-1. Wire the depth template's MODEL, VAE and CONDITIONING.")

    def estimate(self, image, model, vae, conditioning, mode, tile=2048, overlap=512, global_long_edge=2496,
                 detail_sigma=DETAIL_SIGMA):
        if overlap >= tile:
            raise ValueError("overlap must be smaller than the tile size")
        depth, _ = run(image, model, vae, conditioning, mode, tile, overlap, global_long_edge, detail_sigma)
        return (depth[None, :, :, None].repeat(1, 1, 1, 3).to(comfy.model_management.intermediate_device()),)


NODE_CLASS_MAPPINGS = {"VathographyTiledDepth": VathographyTiledDepth}
NODE_DISPLAY_NAME_MAPPINGS = {"VathographyTiledDepth": "Vathography Tiled Depth"}
