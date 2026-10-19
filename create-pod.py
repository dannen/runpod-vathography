#!/usr/bin/env python3
"""
Create, update, or recreate the Vathography (Marigold V2) persistent RunPod Pod, and optionally
create the dedicated network volume it mounts.

Uses RunPod's REST API (https://rest.runpod.io/v1), not the GraphQL
podFindAndDeployOnDemand mutation — the REST /pods schema documents a
networkVolumeId field directly (confirmed against the live OpenAPI spec at
https://rest.runpod.io/v1/openapi.json), which is what's needed to attach a
pre-existing dedicated volume. GraphQL's pod mutations have shifted over
time; re-verify against that OpenAPI spec if this ever starts 400ing.

A running GPU Pod isn't safely "upsertable" the way a stateless serverless
template is, so unlike create-template.py this script does NOT silently
mutate an existing pod by default:
  - no matching pod found            -> create it
  - matching pod found, no flag       -> report it and exit (safe no-op)
  - matching pod found, --update      -> PATCH the fields that are safely
                                          patchable on a live pod (image,
                                          env, ports, container disk)
  - matching pod found, --recreate    -> DELETE + fresh POST (needed for
                                          anything PATCH can't touch: GPU
                                          type, datacenter, network volume)

Usage:
    # One-time: create the dedicated network volume
    python3 create-pod.py --create-volume --volume-name vathography-storage \\
        --volume-size 30 --datacenter <DC>

    # Create the pod (idempotent by name; reports existing pod if present)
    python3 create-pod.py --name vathography-comfyui \\
        --image user/vathography-comfyui:latest \\
        --gpu-type-id "NVIDIA RTX PRO 6000 Blackwell Server Edition" --datacenter <DC> \\
        --network-volume-id <id from --create-volume>

    # Roll a running pod onto a new image tag
    python3 create-pod.py --name vathography-comfyui --image user/vathography-comfyui:latest --update

    # Try several datacenters and GPU types in order, until one has stock. Each
    # datacenter's volume is found by name (--volume-name, as a prefix), so a
    # datacenter with no such volume is skipped. GraphQL is used where REST
    # rejects the datacenter (US-NE-1 in October 2026).
    python3 create-pod.py --image user/vathography-comfyui:<tag> \\
        --datacenter US-MO-1,US-NE-1,US-IL-1 \\
        --gpu-type-id "NVIDIA L40S,NVIDIA RTX A5000,NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb"

    # Change GPU type / datacenter / volume (requires delete + recreate)
    python3 create-pod.py --name vathography-comfyui --image user/vathography-comfyui:latest \\
        --gpu-type-id "NVIDIA H100 80GB HBM3" --datacenter <DC> \\
        --network-volume-id <id> --recreate
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
import urllib.parse

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
REST_BASE = "https://rest.runpod.io/v1"


def rest(method: str, path: str, body: dict = None, params: dict = None) -> dict:
    url = f"{REST_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "runpod-vathography-pod/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {detail}") from None


def find_pod_by_name(name: str) -> dict | None:
    """Return the existing pod dict if one with this name exists, else None."""
    pods = rest("GET", "/pods", params={"name": name})
    if isinstance(pods, list):
        for p in pods:
            if p.get("name") == name:
                return p
    return None


def get_network_volume(volume_id: str) -> dict:
    return rest("GET", f"/networkvolumes/{volume_id}")


def create_network_volume(name: str, size: int, datacenter: str) -> dict:
    """Create a new network volume. Returns {id, name, size, dataCenterId}.

    NOTE: volume encryption cannot be set via the API — enable it in the
    RunPod console (checkbox shown when attaching the volume to a pod).
    """
    return rest(
        "POST",
        "/networkvolumes",
        body={"name": name, "size": size, "dataCenterId": datacenter},
    )


def create_pod(
    name: str,
    image: str,
    gpu_type_id: str,
    datacenter: str,
    network_volume_id: str,
    container_disk_gb: int = 30,
    volume_mount_path: str = "/workspace",
    ports: list[str] = None,
    env: dict = None,
    allowed_cuda_versions: list[str] = None,
) -> dict:
    body = {
        "name": name,
        "imageName": image,
        "cloudType": "SECURE",  # required for network-volume support
        "gpuTypeIds": [gpu_type_id],
        "gpuCount": 1,
        "dataCenterIds": [datacenter],
        "dataCenterPriority": "custom",
        "networkVolumeId": network_volume_id,
        "volumeInGb": 0,  # persistence comes from the attached network volume, not a pod volume
        "volumeMountPath": volume_mount_path,
        "containerDiskInGb": container_disk_gb,
        "ports": ports or ["8188/http", "8770/http", "22/tcp"],
        "env": env or {},
    }
    if allowed_cuda_versions:
        body["allowedCudaVersions"] = allowed_cuda_versions
    return rest("POST", "/pods", body=body)


def create_pod_graphql(name, image, gpu_type_id, datacenter, network_volume_id, container_disk_gb,
                       volume_mount_path, ports, env, allowed_cuda_versions) -> dict:
    """The same pod through GraphQL podFindAndDeployOnDemand, for datacenters the REST
    schema doesn't list yet. One GPU type per call."""
    inp = {"cloudType": "SECURE", "gpuCount": 1, "gpuTypeId": gpu_type_id, "dataCenterId": datacenter,
           "networkVolumeId": network_volume_id, "volumeInGb": 0, "volumeMountPath": volume_mount_path,
           "containerDiskInGb": container_disk_gb, "imageName": image, "name": name,
           "ports": ",".join(ports or []), "env": [{"key": k, "value": v} for k, v in (env or {}).items()]}
    if allowed_cuda_versions:
        inp["allowedCudaVersions"] = allowed_cuda_versions
    body = json.dumps({"query": "mutation($input: PodFindAndDeployOnDemandInput) "
                                "{ podFindAndDeployOnDemand(input: $input) { id name } }",
                       "variables": {"input": inp}}).encode()
    req = urllib.request.Request("https://api.runpod.io/graphql", data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}",
        "User-Agent": "runpod-vathography-pod/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        r = json.loads(resp.read())
    pod = (r.get("data") or {}).get("podFindAndDeployOnDemand")
    if not pod or not pod.get("id"):
        raise RuntimeError("GraphQL: " + ((r.get("errors") or [{}])[0].get("message") or json.dumps(r)[:300]))
    return pod


def find_volume(datacenter: str, name_prefix: str) -> dict | None:
    """The network volume in this datacenter whose name starts with name_prefix (the
    exact name first), or None."""
    vols = [v for v in rest("GET", "/networkvolumes") if v.get("dataCenterId") == datacenter
            and v.get("name", "").startswith(name_prefix)]
    vols.sort(key=lambda v: v["name"] != name_prefix)
    return vols[0] if vols else None


def update_pod(pod_id: str, image: str = None, env: dict = None,
               ports: list[str] = None, container_disk_gb: int = None) -> dict:
    body = {}
    if image is not None:
        body["imageName"] = image
    if env is not None:
        body["env"] = env
    if ports is not None:
        body["ports"] = ports
    if container_disk_gb is not None:
        body["containerDiskInGb"] = container_disk_gb
    if not body:
        raise ValueError("update_pod: nothing to update")
    return rest("PATCH", f"/pods/{pod_id}", body=body)


def delete_pod(pod_id: str) -> None:
    rest("DELETE", f"/pods/{pod_id}")


def proxy_url(pod_id: str, port: int = 8188) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def main():
    parser = argparse.ArgumentParser(
        description="Create/update/recreate the Vathography (Marigold V2) persistent RunPod Pod, "
                     "and optionally create its dedicated network volume."
    )
    parser.add_argument("--name", default="vathography-comfyui", help="Pod name (default: vathography-comfyui)")
    parser.add_argument("--image", help="Docker image reference (e.g. user/vathography-comfyui:latest)")
    parser.add_argument("--gpu-type-id", help='GPU type, e.g. "NVIDIA RTX PRO 6000 Blackwell Server Edition" (see check-gpu-availability.py); '
                                              'a comma-separated list is tried in order')
    parser.add_argument("--datacenter", help="Datacenter ID, e.g. US-IL-1 — must match the network volume's datacenter; a comma-separated "
                                             "list is tried in order, each with its own volume (found by --volume-name)")
    parser.add_argument("--network-volume-id", help="Network volume ID to attach (one datacenter only; otherwise found by --volume-name)")
    parser.add_argument("--container-disk-gb", type=int, default=40, help="Container disk GB (default: 40)")
    parser.add_argument("--volume-mount-path", default="/workspace", help="Mount path for the network volume (default: /workspace)")
    parser.add_argument("--ports", default="8188/http,8770/http,22/tcp",
                        help="Comma-separated ports (default: 8188/http,8770/http,22/tcp: ComfyUI, the vathopy GUI, SSH)")
    parser.add_argument("--allowed-cuda-versions", default="13.0",
                         help="Comma-separated host CUDA versions to accept (default: 13.0 — the image's "
                              "cu130 torch needs driver >= 580). Pass '' to allow any.")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                         help="Pod environment variable (repeatable). Replaces the pod's full env list.")

    parser.add_argument("--update", action="store_true",
                         help="If a pod with this name exists, PATCH it (image/env/ports/container-disk only)")
    parser.add_argument("--recreate", action="store_true",
                         help="If a pod with this name exists, DELETE it and create a fresh one "
                              "(needed to change GPU type / datacenter / network volume)")

    parser.add_argument("--create-volume", action="store_true", help="Create a network volume and exit")
    parser.add_argument("--volume-name", default="vathography-storage",
                        help="Volume name (default: vathography-storage). Creating a pod without --network-volume-id uses the "
                             "datacenter's volume whose name starts with this (e.g. vathography-storage-ne)")
    parser.add_argument("--volume-size", type=int, default=30,
                        help="Volume size GB (default: 30; the models take 26.4GB, and a volume can grow later but not shrink)")

    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: RUNPOD_API_KEY environment variable not set.")
        sys.exit(1)

    if args.create_volume:
        if not args.datacenter:
            parser.error("--datacenter is required with --create-volume")
        print(f"Creating network volume '{args.volume_name}' ({args.volume_size}GB) in {args.datacenter}...",
              end=" ", flush=True)
        try:
            vol = create_network_volume(args.volume_name, args.volume_size, args.datacenter)
        except Exception as e:
            print(f"\nERROR: {e}")
            sys.exit(1)
        print("done.")
        print(f"  Volume ID:   {vol['id']}")
        print(f"  Datacenter:  {vol['dataCenterId']}")
        print(f"  Size:        {vol['size']}GB")
        print()
        print("  NOTE: encryption can't be set via the API — enable it in the RunPod")
        print("  console's 'Encrypt Volume' checkbox when attaching this volume.")
        return

    if args.update and args.recreate:
        parser.error("--update and --recreate are mutually exclusive")

    if not args.image:
        parser.error("--image is required")

    env_dict = {}
    for item in args.env:
        if "=" not in item:
            parser.error(f"--env value {item!r} must be KEY=VALUE")
        k, v = item.split("=", 1)
        env_dict[k] = v
    ports_list = [p.strip() for p in args.ports.split(",") if p.strip()]

    print(f"Looking up existing pod '{args.name}'...", end=" ", flush=True)
    existing = find_pod_by_name(args.name)

    if existing and not args.update and not args.recreate:
        print(f"found (id: {existing['id']}, status: {existing.get('desiredStatus')}).")
        print("Not modifying it — pass --update to patch image/env/ports, or --recreate")
        print("to delete and redeploy (required to change GPU type / datacenter / volume).")
        print(f"  Proxy URL: {proxy_url(existing['id'])}")
        return

    if existing and args.update:
        print(f"found (id: {existing['id']}). Updating (image/env/ports/container-disk).")
        try:
            pod = update_pod(
                existing["id"], image=args.image, env=env_dict,
                ports=ports_list, container_disk_gb=args.container_disk_gb,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        print(f"Updated pod: {pod.get('name', args.name)}")
        print(f"  Pod ID:    {pod.get('id', existing['id'])}")
        print(f"  Proxy URL: {proxy_url(pod.get('id', existing['id']))}")
        return

    dcs = [d.strip() for d in (args.datacenter or "").split(",") if d.strip()]
    gpus = [g.strip() for g in (args.gpu_type_id or "").split(",") if g.strip()]
    if args.network_volume_id and len(dcs) > 1:
        parser.error("--network-volume-id goes with a single --datacenter (with a list, each datacenter's volume is found by --volume-name)")

    if existing and args.recreate:
        print(f"found (id: {existing['id']}). Deleting for recreate.")
        if not gpus or not dcs:
            parser.error("--recreate requires --gpu-type-id and --datacenter")
        try:
            delete_pod(existing["id"])
        except Exception as e:
            print(f"ERROR deleting existing pod: {e}")
            sys.exit(1)
    else:
        print("not found. Creating.")
        if not gpus or not dcs:
            parser.error("--gpu-type-id and --datacenter are required to create a new pod")

    cuda = [v.strip() for v in args.allowed_cuda_versions.split(",") if v.strip()]
    pod = None
    for dc in dcs:
        # Fail fast rather than deploying a pod whose /workspace never mounts.
        try:
            if args.network_volume_id:
                vol = get_network_volume(args.network_volume_id)
                if vol.get("dataCenterId") != dc:
                    print(f"ERROR: network volume {args.network_volume_id} is in datacenter "
                          f"{vol.get('dataCenterId')!r}, but --datacenter is {dc!r}. "
                          "A pod and its network volume must be in the same datacenter or the volume won't mount.")
                    sys.exit(1)
            else:
                vol = find_volume(dc, args.volume_name)
                if not vol:
                    print(f"  {dc}: no volume named {args.volume_name}* there, skipping")
                    continue
        except Exception as e:
            print(f"ERROR: could not look up the network volume for {dc}: {e}")
            sys.exit(1)
        common = dict(name=args.name, image=args.image, datacenter=dc, network_volume_id=vol["id"],
                      container_disk_gb=args.container_disk_gb, volume_mount_path=args.volume_mount_path,
                      ports=ports_list, env=env_dict, allowed_cuda_versions=cuda)
        for gpu in gpus:
            try:
                pod = create_pod(gpu_type_id=gpu, **common)
            except Exception as e:
                msg = str(e)
                if "HTTP 400" in msg and "enum" in msg:           # REST's schema doesn't list this datacenter or GPU type yet
                    try:
                        pod = create_pod_graphql(gpu_type_id=gpu, **common)
                    except Exception as e2:
                        msg = str(e2)
                if not pod:
                    print(f"  {dc} / {gpu}: {msg.strip()[:160]}")
                    continue
            print(f"  {dc} / {gpu}: created (volume {vol.get('name')} {vol['id']})")
            break
        if pod:
            break
    if not pod:
        print("ERROR: no pod could be created in " + ", ".join(dcs) + " with " + ", ".join(gpus))
        sys.exit(1)

    print(f"Created pod: {pod.get('name', args.name)}")
    print(f"  Pod ID:     {pod['id']}")
    print(f"  GPU:        {gpu}")
    print(f"  Datacenter: {dc}")
    print(f"  Proxy URL:  {proxy_url(pod['id'])}")
    print()
    print("  If the volume wasn't pre-seeded (populate-volume.sh --s3), first boot")
    print("  downloads any missing models (~26GB) before ComfyUI comes up —")
    print("  watch progress via the RunPod console or SSH.")


if __name__ == "__main__":
    main()
