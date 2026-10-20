# Building and pushing the image

Build locally only. Don't add a GitHub Actions workflow: hosted runners run out of disk on images like this one.

Replace `YOUR_DOCKERHUB_USER` with your Docker Hub user.

```bash
python3 make_workflows.py   # if make_workflows.py or an official template changed
./sync-vathopy.sh           # copy vathopy (the GUI) from a vathopy checkout (default ../vathopy) into vathopy-src/
docker build --no-cache \
  -t YOUR_DOCKERHUB_USER/vathography-comfyui:latest \
  -t YOUR_DOCKERHUB_USER/vathography-comfyui:$(git rev-parse --short HEAD 2>/dev/null || echo local) \
  .
docker push YOUR_DOCKERHUB_USER/vathography-comfyui:latest
docker push YOUR_DOCKERHUB_USER/vathography-comfyui:$(git rev-parse --short HEAD 2>/dev/null || echo local)
```

### cu128 variant (hosts with older drivers)

The default image needs a host with CUDA 13 (driver >= 580), and those are scarce. The cu128 build runs on CUDA 12.8+ hosts: the int8 backbone runs on comfy-kitchen's eager (plain PyTorch) backend instead of its CUDA kernels. Its depth maps match the cu130 build's to within about 1%, about as much as resizing the input differently changes them. Tag it `-cu128`: `create-pod.py` then accepts CUDA 12.8, 12.9 and 13.0 hosts by default.

```bash
docker build --no-cache \
  --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 --build-arg CUDA_TAG=cu128 \
  -t YOUR_DOCKERHUB_USER/vathography-comfyui:$(git rev-parse --short HEAD)-cu128 .
docker push YOUR_DOCKERHUB_USER/vathography-comfyui:$(git rev-parse --short HEAD)-cu128
```

In a datacenter without a network volume, give the pod its own volume instead. The models then download from Hugging Face on first boot, about 26GB, and are lost with the pod:

```bash
./create-pod.sh --image YOUR_DOCKERHUB_USER/vathography-comfyui:<sha>-cu128 \
  --datacenter <DC> --gpu-type-id "NVIDIA RTX A5000" --pod-volume-gb 50 \
  --env "VATHOPY_GUI_AUTH=$(cat ~/.api/runpod/vathopy-gui.auth)"
```

Quick check without a GPU (import-level only):
```bash
docker run --rm --entrypoint ./venv/bin/python YOUR_DOCKERHUB_USER/vathography-comfyui:latest \
  -c "import torch, comfy_extras.nodes_marigold; print(torch.__version__)"
```

## Rolling out to a running pod

Point the pod at the new **git-sha tag**, not `:latest`:

```bash
./create-pod.sh --image YOUR_DOCKERHUB_USER/vathography-comfyui:$(git rev-parse --short HEAD) --update \
  --env "VATHOPY_GUI_AUTH=$(cat ~/.api/runpod/vathopy-gui.auth)"
```

**Always pass `--env VATHOPY_GUI_AUTH=...`:** `--update` replaces the pod's whole env list, so leaving it out would bring the vathopy GUI **and ComfyUI** back up with no password on a public URL.

Why not `:latest`: RunPod has been seen restarting workers on the image they already had after a new `:latest` was pushed. Changing the image reference forces a fresh pull.

`--update` PATCHes the image, env, ports and container disk. `--env` **replaces the pod's whole env list**, so pass every variable you want to keep. Changing the GPU type, datacenter or volume needs `--recreate` (delete, then create) plus `--gpu-type-id`, `--datacenter` and `--network-volume-id`.

## Upgrading ComfyUI

Change `COMFYUI_REF` in the Dockerfile and rebuild. Nothing on the volume depends on the ComfyUI version, since there are no custom nodes and no venv on the volume. After an upgrade, refresh the official templates from [Comfy-Org/workflow_templates](https://github.com/Comfy-Org/workflow_templates/tree/main/templates) (`image_marigold_v2_*.json`) and rerun `make_workflows.py`.
