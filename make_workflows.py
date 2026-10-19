#!/usr/bin/env python3
"""
Generate the vathography_*.json workflows from the official Comfy-Org
Marigold V2 templates in workflows/. Change this script and rerun it; don't
hand-edit the output.

    python3 make_workflows.py

Each official template is LoadImage -> Marigold V2 subgraph -> 8-bit PNG
SaveImageAdvanced -> ImageCompare. Changes per template:

1. A resize between LoadImage and the subgraph, set by one number: the
   "Inference size (long edge, px)" node (default 1280). The templates run
   inference at the photo's own size, which for a camera photo (6000 px and
   up) runs out of memory and is far above the ~1250 x 835 the Qwen-Image
   backbone was trained around.
   Each side is rounded to the nearest multiple of 16, never above the
   photo's own long edge: Get Image Size -> two Math Expressions
   (round(side * L / long_side / 16) * 16) -> Image Scale (lanczos, no
   crop). Otherwise VAEEncode center-crops each side to a multiple of 8,
   and the DiT pads an odd latent side with circular padding (the opposite
   edge wraps in), so the depth map would be up to 7 px smaller than the
   photo and not cover the full frame. With multiples of 16 the depth map
   is exactly the scaled photo's size, so scaling it back to the original's
   dimensions lines up. Rounding both sides stretches the shape by at most
   8 px (~0.5% at 1280), which scaling back undoes.
2. Save as 32-bit float EXR instead of 8-bit PNG, so depth isn't quantized
   to 256 steps before vathopy/Blender see it. Files are named after the
   photo, <photo>_<prediction>_00001.exr, by our Vathography Output Name
   node (custom_nodes/vathography) feeding the save node's filename_prefix. Depth and normals are saved
   with input_color_space "linear" (values written unchanged). Albedo is
   sRGB-encoded by the post-process node, so it's saved as "sRGB" and the
   EXR holds scene-linear albedo.
3. A Preview Image node on the model's output, and no Image Compare.
   Browsers can't display EXR, so the save node shows nothing, and Image
   Compare starts on the "before" image until the mouse is dragged across
   it, so it looked blank. The preview shows the result straight away (an
   8-bit PNG in ComfyUI's temp folder, not saved).
4. Every node (and group) on the main canvas is pinned, so it can't be
   dragged out of place by accident. Right-click > Unpin to move one.

Every edit asserts what it expects to find, so a changed upstream template
fails loudly instead of producing a broken graph.
"""

import json
import uuid
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parent / "workflows"

# official template -> (output name, prediction, EXR input_color_space)
TEMPLATES = {
    "image_marigold_v2_depth_estimation.json": ("vathography_depth.json", "depth", "linear"),
    "image_marigold_v2_surface_normal_estimation.json": ("vathography_normals.json", "normals", "linear"),
    "image_marigold_v2_albedo_estimation.json": ("vathography_albedo.json", "albedo", "sRGB"),
}

LOAD_ID, COMPARE_ID, SAVE_ID, MARIGOLD_ID = 12, 21, 23, 24
SCALE_ID = 26
PREVIEW_ID = 27
SIZE_ID, LONG_EDGE_ID, WIDTH_ID, HEIGHT_ID = 28, 29, 30, 31
NAME_ID = 32
NEW_IDS = (SCALE_ID, PREVIEW_ID, SIZE_ID, LONG_EDGE_ID, WIDTH_ID, HEIGHT_ID, NAME_ID)
DEFAULT_LONG_EDGE = 1280
MULTIPLE = 16
# a = width, b = height, c = long edge setting (never above the photo's own).
SIDE_EXPR = "max({m}, round({side} * min(c, max(a, b)) / max(a, b) / {m}) * {m})"


def node(graph: dict, node_id: int) -> dict:
    found = [n for n in graph["nodes"] if n["id"] == node_id]
    assert len(found) == 1, f"expected one node {node_id}, found {len(found)}"
    return found[0]


def link(graph: dict, link_id: int) -> list:
    found = [lk for lk in graph["links"] if lk[0] == link_id]
    assert len(found) == 1, f"expected one link {link_id}, found {len(found)}"
    return found[0]


def add_link(graph: dict, origin: int, origin_slot: int, target: int, target_slot: int, kind: str) -> int:
    lid = graph["last_link_id"] + 1
    graph["links"].append([lid, origin, origin_slot, target, target_slot, kind])
    graph["last_link_id"] = lid
    return lid


def widget_input(name: str, kind: str, link_id=None) -> dict:
    return {"localized_name": name, "name": name, "type": kind, "widget": {"name": name}, "link": link_id}


def math_node(node_id: int, pos: list, side: str, links: tuple, out_link: int) -> dict:
    expr = SIDE_EXPR.format(m=MULTIPLE, side=side)
    a, b, c = links
    return {
        "id": node_id,
        "type": "ComfyMathExpression",
        "title": f"{'Width' if side == 'a' else 'Height'} (multiple of {MULTIPLE})",
        "pos": pos,
        "size": [400, 200],
        "flags": {"collapsed": True},
        "order": 2,
        "mode": 0,
        "inputs": [
            {"label": "a", "localized_name": "values.a", "name": "values.a", "type": "FLOAT,INT", "link": a},
            {"label": "b", "localized_name": "values.b", "name": "values.b", "shape": 7, "type": "FLOAT,INT", "link": b},
            {"label": "c", "localized_name": "values.c", "name": "values.c", "shape": 7, "type": "FLOAT,INT", "link": c},
            {"label": "d", "localized_name": "values.d", "name": "values.d", "shape": 7, "type": "FLOAT,INT", "link": None},
            widget_input("expression", "STRING"),
        ],
        "outputs": [
            {"localized_name": "FLOAT", "name": "FLOAT", "type": "FLOAT", "links": None},
            {"localized_name": "INT", "name": "INT", "type": "INT", "links": [out_link]},
            {"localized_name": "BOOL", "name": "BOOL", "type": "BOOLEAN", "links": None},
        ],
        "properties": {"cnr_id": "comfy-core", "Node name for S&R": "ComfyMathExpression"},
        "widgets_values": [expr],
    }


def adapt(graph: dict, prediction: str, color_space: str, name: str) -> dict:
    load = node(graph, LOAD_ID)
    compare = node(graph, COMPARE_ID)
    save = node(graph, SAVE_ID)
    marigold = node(graph, MARIGOLD_ID)
    assert load["type"] == "LoadImage", load["type"]
    assert compare["type"] == "ImageCompare", compare["type"]
    assert save["type"] == "SaveImageAdvanced", save["type"]
    assert marigold["widgets_values"][2] == prediction, marigold["widgets_values"]
    assert save["widgets_values"] == ["ComfyUI", "png", "8-bit", "sRGB"], save["widgets_values"]
    taken = [n["id"] for n in graph["nodes"] if n["id"] in NEW_IDS]
    assert not taken, f"node ids {taken} already taken"

    # LoadImage feeds ImageCompare (link a) and the subgraph (link b); the
    # save node's pass-through output feeds ImageCompare too. Drop ImageCompare
    # and both of its links (3. above).
    load_out = load["outputs"][0]
    assert load_out["name"] == "IMAGE" and len(load_out["links"]) == 2, load_out
    to_compare, to_marigold = sorted(load_out["links"])
    assert link(graph, to_compare)[3] == COMPARE_ID and link(graph, to_marigold)[3] == MARIGOLD_ID
    save_out = save["outputs"][0]
    assert len(save_out["links"]) == 1 and link(graph, save_out["links"][0])[3] == COMPARE_ID, save_out
    dropped = {to_compare, save_out["links"][0]}
    graph["links"] = [lk for lk in graph["links"] if lk[0] not in dropped]
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] != COMPARE_ID]
    load_out["links"] = [to_marigold]
    save_out["links"] = []

    # 1. The resize: LoadImage -> Image Scale -> {subgraph, ImageCompare}, with
    #    Image Scale's width/height computed from the photo's size and the
    #    long-edge setting, each rounded to a multiple of 16.
    to_scale = add_link(graph, LOAD_ID, 0, SCALE_ID, 0, "IMAGE")
    to_size = add_link(graph, LOAD_ID, 0, SIZE_ID, 0, "IMAGE")
    link(graph, to_marigold)[1] = SCALE_ID
    load_out["links"] = [to_scale, to_size]
    w_a, h_a = add_link(graph, SIZE_ID, 0, WIDTH_ID, 0, "INT"), add_link(graph, SIZE_ID, 0, HEIGHT_ID, 0, "INT")
    w_b, h_b = add_link(graph, SIZE_ID, 1, WIDTH_ID, 1, "INT"), add_link(graph, SIZE_ID, 1, HEIGHT_ID, 1, "INT")
    w_c, h_c = add_link(graph, LONG_EDGE_ID, 0, WIDTH_ID, 2, "INT"), add_link(graph, LONG_EDGE_ID, 0, HEIGHT_ID, 2, "INT")
    to_width = add_link(graph, WIDTH_ID, 1, SCALE_ID, 2, "INT")
    to_height = add_link(graph, HEIGHT_ID, 1, SCALE_ID, 3, "INT")

    x, y = load["pos"]
    below = y + load["size"][1] + 60
    graph["nodes"] += [
        {
            "id": LONG_EDGE_ID,
            "type": "PrimitiveInt",
            "title": "Inference size (long edge, px)",
            "pos": [x, below],
            "size": [290, 82],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [widget_input("value", "INT")],
            "outputs": [{"localized_name": "INT", "name": "INT", "type": "INT", "links": [w_c, h_c]}],
            "properties": {"cnr_id": "comfy-core", "Node name for S&R": "PrimitiveInt"},
            "widgets_values": [DEFAULT_LONG_EDGE, "fixed"],
        },
        {
            "id": SIZE_ID,
            "type": "GetImageSize",
            "pos": [x, below + 130],
            "size": [290, 120],
            "flags": {"collapsed": True},
            "order": 1,
            "mode": 0,
            "inputs": [{"localized_name": "image", "name": "image", "type": "IMAGE", "link": to_size}],
            "outputs": [
                {"localized_name": "width", "name": "width", "type": "INT", "links": [w_a, h_a]},
                {"localized_name": "height", "name": "height", "type": "INT", "links": [w_b, h_b]},
                {"localized_name": "batch_size", "name": "batch_size", "type": "INT", "links": None},
            ],
            "properties": {"cnr_id": "comfy-core", "Node name for S&R": "GetImageSize"},
            "widgets_values": [],
        },
        math_node(WIDTH_ID, [x, below + 180], "a", (w_a, w_b, w_c), to_width),
        math_node(HEIGHT_ID, [x, below + 230], "b", (h_a, h_b, h_c), to_height),
        {
            "id": SCALE_ID,
            "type": "ImageScale",
            "title": f"Resize (multiples of {MULTIPLE})",
            "pos": [x, below + 290],
            "size": [290, 160],
            "flags": {"collapsed": True},
            "order": 3,
            "mode": 0,
            "inputs": [
                {"localized_name": "image", "name": "image", "type": "IMAGE", "link": to_scale},
                widget_input("upscale_method", "COMBO"),
                widget_input("width", "INT", to_width),
                widget_input("height", "INT", to_height),
                widget_input("crop", "COMBO"),
            ],
            "outputs": [{"localized_name": "IMAGE", "name": "IMAGE", "type": "IMAGE", "links": [to_marigold]}],
            "properties": {"cnr_id": "comfy-core", "Node name for S&R": "ImageScale"},
            "widgets_values": ["lanczos", DEFAULT_LONG_EDGE, DEFAULT_LONG_EDGE, "disabled"],
        },
    ]
    graph["last_node_id"] = max(graph["last_node_id"], *NEW_IDS)

    # 3. Preview Image on the subgraph's output, next to the save node.
    marigold_out = marigold["outputs"][0]
    assert marigold_out["type"] == "IMAGE" and len(marigold_out["links"]) == 1, marigold_out
    preview_link = add_link(graph, MARIGOLD_ID, 0, PREVIEW_ID, 0, "IMAGE")
    marigold_out["links"].append(preview_link)
    sx, sy = save["pos"]
    graph["nodes"].append({
        "id": PREVIEW_ID,
        "type": "PreviewImage",
        "title": "Result preview",
        "pos": [sx, sy + save["size"][1] + 60],
        "size": [440, 360],
        "flags": {},
        "order": 4,
        "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": preview_link}],
        "outputs": [],
        "properties": {"cnr_id": "comfy-core", "Node name for S&R": "PreviewImage"},
        "widgets_values": [],
    })

    # 2. Float EXR output, named after the photo: Vathography Output Name
    #    (custom_nodes/vathography, ours) -> the save node's filename_prefix,
    #    giving vathography/<photo>_<prediction>_00001.exr.
    to_name = add_link(graph, LOAD_ID, 0, NAME_ID, 0, "IMAGE")
    load_out["links"].append(to_name)
    to_prefix = add_link(graph, NAME_ID, 0, SAVE_ID, 1, "STRING")
    assert [i["name"] for i in save["inputs"]] == ["images"], save["inputs"]
    save["inputs"].append(widget_input("filename_prefix", "STRING", to_prefix))
    graph["nodes"].append({
        "id": NAME_ID,
        "type": "VathographyOutputName",
        "title": "Output name (from the photo)",
        "pos": [sx, sy - 170],
        "size": [440, 130],
        "flags": {},
        "order": 2,
        "mode": 0,
        "inputs": [
            {"localized_name": "image", "name": "image", "type": "IMAGE", "link": to_name},
            widget_input("suffix", "STRING"),
            widget_input("folder", "STRING"),
        ],
        "outputs": [{"localized_name": "filename_prefix", "name": "filename_prefix", "type": "STRING", "links": [to_prefix]}],
        "properties": {"Node name for S&R": "VathographyOutputName"},
        "widgets_values": [prediction, "vathography"],
    })
    prefix = f"vathography/{prediction}"  # placeholder; the linked input overrides it
    save["widgets_values"] = [prefix, "exr", "32-bit float", color_space]
    save["widgets_values_named"] = {
        "filename_prefix": prefix,
        "format": "exr",
        "format.bit_depth": "32-bit float",
        "format.input_color_space": color_space,
    }

    # 4. Pin everything on the main canvas (not inside the subgraph).
    for item in graph["nodes"] + graph.get("groups", []):
        item.setdefault("flags", {})["pinned"] = True

    # A distinct workflow id, so the frontend doesn't treat it as the official one.
    graph["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"runpod-vathography/{name}"))
    return graph


def main():
    for src, (dst, prediction, color_space) in TEMPLATES.items():
        graph = json.loads((WORKFLOWS / src).read_text())
        out = adapt(graph, prediction, color_space, dst)
        (WORKFLOWS / dst).write_text(json.dumps(out, indent=2) + "\n")
        print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()
