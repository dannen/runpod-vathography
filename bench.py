#!/usr/bin/env python3
"""
Phase 1 benchmark: run Marigold V2 depth on the pod at one or more sizes and
record wall time and peak VRAM (vathography.md §14.4). Python 3 stdlib only.

    python3 bench.py --url https://<pod-id>-8188.proxy.runpod.net \\
        --image photo.jpg --long-edge 1280 2500 3750 --repeat 2 --out bench-out/

Sizes are the long edge in pixels; the photo keeps its shape, with each
side rounded to the nearest multiple of 16 (a 3:2 photo at 2500 runs at
2496 x 1664), so the saved map is exactly that size.

For each size it submits the same graph as workflows/vathography_depth.json
(flattened from the official template's subgraph: same models, sampler,
sigmas and post-process), polls /system_stats while the job runs, and saves
the EXR to --out. One JSON line per run goes to stdout.

Peak VRAM is total minus the lowest free memory /system_stats reported while
the job ran, sampled every --poll seconds (default 0.2). That's device-level (it includes
the loaded model and PyTorch's cache), so it's what decides whether a size
fits. It can miss a short spike between samples, so treat it as a lower
bound. The first run at each session includes loading the 20.5GB backbone
from the volume; --repeat 2 gives a warm number.

Cloudflare in front of RunPod's proxy rejects urllib's default User-Agent
(error 1010), so every request sets one.
"""

import argparse
import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

UA = {"User-Agent": "runpod-vathography-bench/1.0"}

PREDICTIONS = {
    "depth": ("marigold_v2_depth_log_stage2.safetensors", "marigold_v2_depth_log_stage2_vae.safetensors",
              "marigold_v2_depth_conditioning.safetensors", "linear"),
    "normals": ("marigold_v2_normals.safetensors", "marigold_v2_normals_vae.safetensors",
                "marigold_v2_normals_conditioning.safetensors", "linear"),
    "albedo": ("marigold_v2_albedo.safetensors", "marigold_v2_albedo_vae.safetensors",
               "marigold_v2_albedo_conditioning.safetensors", "sRGB"),
}


# a = width, b = height, c = long edge (same expression as make_workflows.py).
SIDE_EXPR = "max(16, round({side} * min(c, max(a, b)) / max(a, b) / 16) * 16)"


def request(url, data=None, headers=None, timeout=60):
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def get_json(base, path):
    return json.loads(request(base + path))


def upload_image(base, path: Path) -> str:
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + path.read_bytes() + (
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n"
    ).encode()
    out = json.loads(request(base + "/upload/image", body,
                             {"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=300))
    return out["name"]


def build_prompt(image_name, long_edge, prediction, prefix, run=0):
    lora, vae, cond, color_space = PREDICTIONS[prediction]
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        # Long edge -> each side rounded to a multiple of 16 (never above the
        # photo's own long edge), same as make_workflows.py: no VAE crop and no
        # circular padding, so the map is exactly the scaled photo's size.
        "3": {"class_type": "GetImageSize", "inputs": {"image": ["1", 0]}},
        "4": {"class_type": "ComfyMathExpression",
              "inputs": {"expression": SIDE_EXPR.format(side="a"), "values.a": ["3", 0], "values.b": ["3", 1], "values.c": long_edge}},
        "5": {"class_type": "ComfyMathExpression",
              "inputs": {"expression": SIDE_EXPR.format(side="b"), "values.a": ["3", 0], "values.b": ["3", 1], "values.c": long_edge}},
        "2": {"class_type": "ImageScale",
              "inputs": {"image": ["1", 0], "upscale_method": "lanczos", "width": ["4", 1], "height": ["5", 1], "crop": "disabled"}},
        "10": {"class_type": "UNETLoader",
               "inputs": {"unet_name": "qwen_image_edit_2509_int8_convrot.safetensors", "weight_dtype": "default"}},
        "11": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10", 0], "lora_name": lora, "strength_model": 1}},
        "12": {"class_type": "ModelSamplingAuraFlow",
               "inputs": {"model": ["11", 0], "shift": 1.73, "sampling": "img_to_img_velocity"}},
        "13": {"class_type": "ConditioningLoader", "inputs": {"conditioning_name": cond}},
        "14": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "15": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["14", 0]}},
        "16": {"class_type": "DisableNoise", "inputs": {}},
        "17": {"class_type": "BasicGuider", "inputs": {"model": ["12", 0], "conditioning": ["13", 0]}},
        "18": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        # Trailing spaces parse to the same sigmas but change the cache key, so
        # repeat runs really sample and decode again (the model stays loaded).
        "19": {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.5, 0" + " " * run}},
        "20": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["16", 0], "guider": ["17", 0], "sampler": ["18", 0],
                          "sigmas": ["19", 0], "latent_image": ["15", 0]}},
        "21": {"class_type": "VAEDecode", "inputs": {"samples": ["20", 0], "vae": ["14", 0]}},
        "22": {"class_type": "MarigoldV2PostProcess", "inputs": {"image": ["21", 0], "prediction": prediction}},
        "23": {"class_type": "SaveImageAdvanced",
               "inputs": {"images": ["22", 0], "filename_prefix": prefix, "format": "exr",
                          "format.bit_depth": "32-bit float", "format.input_color_space": color_space}},
    }


def vram(base):
    dev = get_json(base, "/system_stats")["devices"][0]
    return dev["vram_total"], dev["vram_free"]


def run_one(base, image_name, long_edge, prediction, poll, out_dir: Path, run=0):
    total, free_before = vram(base)
    prefix = f"vathography/bench_{prediction}_{long_edge}px"
    body = json.dumps({"prompt": build_prompt(image_name, long_edge, prediction, prefix, run)}).encode()
    t0 = time.time()
    try:
        resp = json.loads(request(base + "/prompt", body, {"Content-Type": "application/json"}))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"prompt rejected: HTTP {e.code}: {e.read().decode(errors='replace')}")
    pid = resp["prompt_id"]
    min_free = free_before
    while True:
        try:
            min_free = min(min_free, vram(base)[1])
            hist = get_json(base, f"/history/{pid}")
        except (urllib.error.URLError, TimeoutError):
            time.sleep(poll)  # proxy hiccup: keep waiting, the job runs on regardless
            continue
        if pid in hist and hist[pid].get("status", {}).get("completed") is not None:
            entry = hist[pid]
            break
        time.sleep(poll)
    wall = time.time() - t0
    status = entry["status"]
    result = {
        "prediction": prediction, "long_edge": long_edge, "status": status.get("status_str"),
        "wall_s": round(wall, 1),
        "peak_vram_gib": round((total - min_free) / 2**30, 2),
        "idle_vram_gib": round((total - free_before) / 2**30, 2),
        "total_vram_gib": round(total / 2**30, 2),
    }
    starts = {m[0]: m[1].get("timestamp") for m in status.get("messages", []) if m[0] in ("execution_start", "execution_success")}
    if starts.get("execution_start") and starts.get("execution_success"):
        result["exec_s"] = round((starts["execution_success"] - starts["execution_start"]) / 1000, 1)
    if status.get("status_str") != "success":
        result["error"] = [m for m in status.get("messages", []) if m[0] == "execution_error"]
        return result
    for node_out in entry.get("outputs", {}).values():
        for img in node_out.get("images", []):
            q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                        "type": img.get("type", "output")})
            data = request(base + "/view?" + q, timeout=300)
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / img["filename"]
            dest.write_bytes(data)
            result["file"] = str(dest)
    return result


def main():
    p = argparse.ArgumentParser(description="Benchmark Marigold V2 on the pod (time and peak VRAM per size).")
    p.add_argument("--url", required=True, help="Pod proxy URL, e.g. https://<pod-id>-8188.proxy.runpod.net")
    p.add_argument("--image", required=True, type=Path, help="Input photo")
    p.add_argument("--long-edge", type=int, nargs="+", default=[1280], help="Inference sizes: long edge in pixels (default 1280)")
    p.add_argument("--prediction", choices=list(PREDICTIONS), default="depth")
    p.add_argument("--repeat", type=int, default=1, help="Runs per size (2 gives a warm number)")
    p.add_argument("--poll", type=float, default=0.2, help="VRAM sampling interval in seconds (default 0.2)")
    p.add_argument("--out", type=Path, default=Path("bench-out"), help="Where to save the EXRs")
    args = p.parse_args()
    base = args.url.rstrip("/")

    name = upload_image(base, args.image)
    for size in args.long_edge:
        for i in range(args.repeat):
            r = run_one(base, name, size, args.prediction, args.poll, args.out, run=int(time.time()) % 1000)
            r["run"] = i + 1
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
