# SPDX-License-Identifier: GPL-3.0-or-later
"""
Vathography See-through Model: the Marigold V2 see-through depth model (depth
behind glass) on its 4-bit nf4 Qwen-Image-Edit-2509 backbone.

The backbone file is a plain bitsandbytes nf4 state dict in safetensors (per
quantised Linear: the packed weight, its absmax and quant_map, and the
packed quant_state), with the model's settings in the file's metadata
(`unet_config`), since ComfyUI can't detect the model from 4-bit shapes. The
weights stay 4-bit in memory and bitsandbytes dequantises each layer's weight
when it runs (bf16). The LoRA (PEFT lora_A / lora_B, scale 1) is added
unmerged, since it can't be merged into 4-bit weights.

Our own code, written for this project. Needs bitsandbytes (Dockerfile).
"""

import json

import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths

QUANT_SUFFIXES = ("weight.absmax", "weight.quant_map", "weight.quant_state.bitsandbytes__nf4")


class Nf4Linear(torch.nn.Module):
    """A Linear whose weight is either bitsandbytes nf4 (dequantised per call) or a plain tensor,
    with an optional unmerged LoRA. Tensors are parameters, so ComfyUI counts and moves them."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = self.bias = self.absmax = self.quant_map = None
        self.lora_A = self.lora_B = None
        self.qs = None                                   # quant_state fields without tensors

    def _load_from_state_dict(self, sd, prefix, local_metadata, strict, missing, unexpected, errors):
        frozen = lambda t: torch.nn.Parameter(t, requires_grad=False)
        if prefix + "weight.absmax" in sd:
            import bitsandbytes.functional as BF
            qd = {k: sd.pop(prefix + k) for k in QUANT_SUFFIXES}
            qs = BF.QuantState.from_dict(qd, device=torch.device("cpu"))
            if qs.quant_type != "nf4" or qs.state2 is not None:
                errors.append(f"{prefix}weight: expected plain nf4 (no double quantisation)")
                return
            self.qs = {"blocksize": qs.blocksize, "shape": qs.shape, "dtype": qs.dtype}
            self.absmax, self.quant_map = frozen(qs.absmax), frozen(qs.code)
        if prefix + "weight" in sd:
            self.weight = frozen(sd.pop(prefix + "weight"))
        else:
            missing.append(prefix + "weight")
        if prefix + "bias" in sd:
            self.bias = frozen(sd.pop(prefix + "bias"))

    def dense_weight(self, device):
        if self.qs is None:
            return self.weight.to(device, torch.bfloat16)
        import bitsandbytes.functional as BF
        state = BF.QuantState(absmax=self.absmax.to(device), code=self.quant_map.to(device), quant_type="nf4",
                              **self.qs)
        return BF.dequantize_4bit(self.weight.to(device), state).to(torch.bfloat16)

    def forward(self, x):
        xb = x.to(torch.bfloat16)
        y = F.linear(xb, self.dense_weight(x.device), None if self.bias is None else self.bias.to(x.device, torch.bfloat16))
        if self.lora_A is not None:
            a, b = self.lora_A.to(x.device, torch.float32), self.lora_B.to(x.device, torch.float32)
            y = y + (xb.float() @ a.t() @ b.t()).to(y.dtype)
        return y.to(x.dtype)


class VathographySeeThroughModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "backbone": (folder_paths.get_filename_list("diffusion_models"),
                         {"tooltip": "marigold_v2_see_through_qwen_image_edit_2509_nf4.safetensors"}),
            "lora": (folder_paths.get_filename_list("loras"), {"tooltip": "marigold_v2_depth_see_through.safetensors"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "vathography"
    DESCRIPTION = ("Loads the Marigold V2 see-through depth model (depth behind glass): the nf4 backbone plus its LoRA. "
                   "Use instead of Load Diffusion Model + LoraLoaderModelOnly, then ModelSamplingAuraFlow as usual.")

    def load(self, backbone, lora):
        path = folder_paths.get_full_path_or_raise("diffusion_models", backbone)
        with safe_open(path, "pt") as f:
            meta = f.metadata() or {}
        if meta.get("quant_type") != "nf4" or "unet_config" not in meta:
            raise ValueError(f"{backbone}: not an nf4 backbone with its unet_config in the metadata")
        model_config = comfy.model_detection.model_config_from_unet_config(json.loads(meta["unet_config"]))
        if model_config is None:
            raise RuntimeError(f"this ComfyUI doesn't know the model in {backbone}")
        load_device = comfy.model_management.get_torch_device()
        cast = comfy.model_management.unet_manual_cast(torch.bfloat16, load_device, model_config.supported_inference_dtypes)
        model_config.set_inference_dtype(torch.bfloat16, cast, device=load_device)
        base = comfy.ops.pick_operations(torch.bfloat16, cast, load_device=load_device, model_config=model_config)
        model_config.custom_operations = type("Nf4Ops", (base,), {"Linear": Nf4Linear})

        sd = comfy.utils.load_torch_file(path, safe_load=True)
        model = model_config.get_model(sd, "")
        model.load_model_weights(sd, "")
        layers = dict(model.diffusion_model.named_modules())

        lsd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora), safe_load=True)
        n = 0
        for key in lsd:
            if not key.endswith(".lora_A.weight"):
                continue
            name = key.removeprefix("diffusion_model.").removesuffix(".lora_A.weight")
            layer = layers.get(name)
            if not isinstance(layer, Nf4Linear):
                raise ValueError(f"{lora}: targets {name!r}, not a Linear layer of this backbone")
            layer.lora_A = torch.nn.Parameter(lsd[key], requires_grad=False)
            layer.lora_B = torch.nn.Parameter(lsd[key.replace(".lora_A.", ".lora_B.")], requires_grad=False)
            n += 1
        empty = [k for k, m in layers.items() if isinstance(m, Nf4Linear) and m.weight is None]
        if empty:
            raise ValueError(f"{backbone}: no weights for {len(empty)} Linear layers, e.g. {empty[0]}")
        quant = sum(isinstance(m, Nf4Linear) and m.qs is not None for m in layers.values())
        print(f"[vathography see-through] {quant} nf4 layers, LoRA on {n}", flush=True)
        return (comfy.model_patcher.ModelPatcher(model, load_device=load_device,
                                                 offload_device=comfy.model_management.unet_offload_device()),)


NODE_CLASS_MAPPINGS = {"VathographySeeThroughModel": VathographySeeThroughModel}
NODE_DISPLAY_NAME_MAPPINGS = {"VathographySeeThroughModel": "Vathography See-through Model"}
