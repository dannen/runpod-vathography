# ComfyUI image for the Vathography (Marigold V2) persistent RunPod Pod.
#
# A lightweight image (OS + venv + ComfyUI core + pinned torch stack) with
# every model on the mounted network volume (see start.sh / models.txt).
# Marigold V2 is native in ComfyUI core since Comfy-Org/ComfyUI #16232
# (2026-09-12, first released in v0.36.0: MarigoldV2PostProcess), and the
# three official templates in workflows/ use core nodes only. The custom
# nodes are ours (custom_nodes/vathography): output names, tiled depth and the
# see-through model loader. No third-party custom nodes.
#
# Two builds (see build.md): the default cu130, and cu128 for hosts with older
# drivers (CUDA 12.8+):
#   --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 --build-arg CUDA_TAG=cu128
ARG BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

# v0.41.3 = newest release as of 2026-10-18. Marigold V2 needs >= v0.36.0.
ARG COMFYUI_REF=v0.41.3
# cu130 gets comfy-kitchen's CUDA kernels for the int8_convrot backbone
# (comfy/quant_ops.py disables that backend when torch.version.cuda < 13), but
# needs host driver >= 580, so create-pod.py passes allowedCudaVersions=["13.0"].
# cu128 runs the same backbone on comfy-kitchen's eager (plain PyTorch) backend:
# same depth maps (within ~1%, the same as resizing the input differently), and
# it runs on far more hosts. Measured on a 16GB RTX 4060 Ti, which needs
# ComfyUI's dynamic VRAM loading for the 20GB backbone: 2496 px in 24 s.
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG CUDA_TAG=cu130

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONNOUSERSITE=1

# ───────────────────── System packages ─────────────────────
# nginx + openssl: the password in front of ComfyUI (start.sh), which has none of its own.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git ca-certificates rsync util-linux \
        python3 python3-venv python3-pip \
        nginx openssl \
    && rm -f /etc/nginx/sites-enabled/default \
    && rm -rf /var/lib/apt/lists/*

# ───────────────────── ComfyUI core ─────────────────────
RUN git clone --branch "${COMFYUI_REF}" --depth 1 \
        https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
    && test -f /opt/ComfyUI/comfy_extras/nodes_marigold.py

WORKDIR /opt/ComfyUI

RUN python3 -m venv venv \
    && ./venv/bin/pip install --no-input --upgrade pip setuptools wheel

# Torch first from the CUDA index, so ComfyUI's unpinned "torch" requirement
# is already satisfied and pip can't swap in a CPU/other-CUDA build.
RUN ./venv/bin/pip install --no-input \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
        "torch==${TORCH_VERSION}+${CUDA_TAG}" \
        "torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}" \
        "torchaudio==${TORCHAUDIO_VERSION}+${CUDA_TAG}" \
    && ./venv/bin/pip install --no-input --prefer-binary -r requirements.txt \
    && ./venv/bin/python -c "import torch; assert torch.__version__.startswith('${TORCH_VERSION}'), torch.__version__"

# ───────────────────── S3 client ─────────────────────
# For MODELS_S3 (start.sh): copy the models from a seeded network volume's S3
# gateway instead of Hugging Face, e.g. for a pod without a network volume.
RUN python3 -m venv /opt/awscli \
    && /opt/awscli/bin/pip install --no-input --prefer-binary awscli \
    && /opt/awscli/bin/aws --version

# ───────────────────── vathopy GUI ─────────────────────
# The vathopy web GUI (sliders, palettes, contours) on port 8770, started by
# start.sh next to ComfyUI and reading ComfyUI's output/vathography/ depth
# maps. Its own venv, so its numpy/OpenCV/OpenEXR pins can't touch ComfyUI's.
# vathopy-src/ comes from ./sync-vathopy.sh (run it before building); its
# VATHOPY_COMMIT records which vathopy commit it is.
COPY vathopy-src/ /opt/vathopy-src/
RUN python3 -m venv /opt/vathopy \
    && /opt/vathopy/bin/pip install --no-input --prefer-binary /opt/vathopy-src \
    && cp /opt/vathopy-src/VATHOPY_COMMIT /opt/vathopy/ \
    && /opt/vathopy/bin/vathopy --version

# ───────────────────── Our custom nodes ─────────────────────
# custom_nodes/vathography: output files named after the photo
# (<photo>_depth_00001.exr), Vathography Tiled Depth (tiled.py) and the
# Vathography See-through Model loader (see_through.py), which needs
# bitsandbytes to dequantise its nf4 backbone (0.48.2 wrote the file).
RUN ./venv/bin/pip install --no-input "bitsandbytes==0.48.2" \
    && ./venv/bin/python -c "import bitsandbytes.functional"
COPY custom_nodes/vathography/ /opt/ComfyUI/custom_nodes/vathography/

# ───────────────────── Workflows + entrypoint ─────────────────────
# start.sh seeds these into the volume's user/default/workflows on boot
# (without overwriting edited copies), so they show up in the Workflows
# sidebar.
COPY workflows/ /opt/vathography-workflows/
COPY start.sh models.txt models-*.txt ./
RUN chmod +x start.sh

EXPOSE 8188 8770

ENTRYPOINT ["/opt/ComfyUI/start.sh"]
