#!/usr/bin/env python3
"""
RunPod GPU availability checker.
Finds GPUs with >=MIN_VRAM on secure cloud (network volume capable),
cross-references with storage-enabled datacenters, and ranks by value.

Usage:
    python3 check-gpu-availability.py          # defaults: >=32GB VRAM, US only
    python3 check-gpu-availability.py --vram 48 --all-regions
    python3 check-gpu-availability.py --vram 32 --show-no-stock
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
from datetime import datetime

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
GRAPHQL_URL = "https://api.runpod.io/graphql"

STOCK_ORDER = {"High": 0, "Medium": 1, "Low": 2, None: 3}
STOCK_LABEL = {"High": "HIGH ", "Medium": "MED  ", "Low": "LOW  ", None: "-----"}


def graphql(query: str) -> dict:
    body = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "runpod-gpu-check/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def fetch_data():
    result = graphql("""
    {
      gpuTypes {
        id
        displayName
        memoryInGb
        secureCloud
        securePrice
        lowestPrice(input: { gpuCount: 1 }) {
          minimumBidPrice
          uninterruptablePrice
        }
      }
      dataCenters {
        id
        name
        location
        storageSupport
        gpuAvailability {
          gpuTypeId
          stockStatus
        }
      }
    }
    """)
    if "errors" in result:
        raise RuntimeError(f"GraphQL error: {result['errors'][0]['message']}")
    return result["data"]


def analyze(data, min_vram: int, us_only: bool, show_no_stock: bool):
    gpu_types = {g["id"]: g for g in data["gpuTypes"]}
    datacenters = data["dataCenters"]

    # Build map: gpu_id -> list of (dc_id, location, stock)
    # Only for storage-enabled DCs
    gpu_dc_map: dict[str, list] = {}
    for dc in datacenters:
        if not dc.get("storageSupport"):
            continue
        if us_only and not (dc["id"].startswith("US-") or dc["id"].startswith("CA-")):
            continue
        for ga in (dc.get("gpuAvailability") or []):
            gid = ga["gpuTypeId"]
            stock = ga.get("stockStatus")
            if gid not in gpu_dc_map:
                gpu_dc_map[gid] = []
            gpu_dc_map[gid].append((dc["id"], dc["location"], stock))

    # Filter GPUs: secure cloud, >=min_vram, has at least one storage-enabled DC
    candidates = []
    for gid, gpu in gpu_types.items():
        if not gpu.get("secureCloud"):
            continue
        vram = gpu.get("memoryInGb", 0)
        if vram < min_vram:
            continue
        secure_price = gpu.get("securePrice")
        if not secure_price:
            continue
        dcs = gpu_dc_map.get(gid, [])
        if not dcs and not show_no_stock:
            continue
        lp = gpu.get("lowestPrice") or {}
        spot = lp.get("minimumBidPrice")
        ondemand = lp.get("uninterruptablePrice")
        value = secure_price / vram  # $/GB VRAM/hr — lower is better
        candidates.append({
            "id": gid,
            "name": gpu["displayName"],
            "vram": vram,
            "secure": secure_price,
            "spot": spot,
            "ondemand": ondemand,
            "value": value,
            "dcs": sorted(dcs, key=lambda x: (STOCK_ORDER.get(x[2], 3), x[0])),
        })

    candidates.sort(key=lambda g: (g["value"], g["secure"]))
    return candidates


def print_report(candidates, min_vram, us_only):
    region_label = "US/CA only" if us_only else "all regions"
    print(f"\n{'='*90}")
    print(f"  RunPod GPU Availability — ≥{min_vram}GB VRAM, Secure Cloud, Network Volume Support")
    print(f"  Regions: {region_label}   |   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*90}")
    print(f"\n  {'GPU':<35} {'VRAM':>5}  {'$/hr':>6}  {'$/GB':>5}  {'Spot':>6}  AVAILABLE DCs (stock)")
    print(f"  {'-'*35} {'-'*5}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*35}")

    for g in candidates:
        spot_s = f"${g['spot']:.2f}" if g["spot"] else "  N/A"
        dc_parts = []
        for dc_id, loc, stock in g["dcs"]:
            label = STOCK_LABEL.get(stock, "?????")
            dc_parts.append(f"{dc_id}[{label.strip()}]")
        dc_str = "  ".join(dc_parts) if dc_parts else "— none in region with storage"

        print(
            f"  {g['name']:<35} {g['vram']:>4}GB"
            f"  ${g['secure']:>5.2f}"
            f"  {g['value']:>5.3f}"
            f"  {spot_s:>6}"
            f"  {dc_str}"
        )

    print()
    if candidates:
        top = candidates[0]
        top_dcs_with_stock = [dc for dc in top["dcs"] if dc[2] is not None]
        top_dc = top_dcs_with_stock[0] if top_dcs_with_stock else (top["dcs"][0] if top["dcs"] else None)
        print(f"  BEST VALUE:  {top['name']}  —  ${top['secure']:.2f}/hr  ({top['vram']}GB VRAM)")
        if top_dc:
            print(f"  DEPLOY IN:   {top_dc[0]}  ({top_dc[1]})  stock: {top_dc[2] or 'unconfirmed'}")
        print()
    print(f"  $/GB = secure price per GB of VRAM/hr (lower = better value)")
    print(f"  stock None = API returned no count (may still be available; check console)")
    print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description="RunPod GPU availability checker")
    parser.add_argument("--vram", type=int, default=32, help="Minimum VRAM in GB (default: 32)")
    parser.add_argument("--all-regions", action="store_true", help="Include non-US datacenters")
    parser.add_argument("--show-no-stock", action="store_true", help="Show GPUs with no storage-DC match")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: RUNPOD_API_KEY environment variable not set.")
        sys.exit(1)

    print("Querying RunPod API...", end=" ", flush=True)
    try:
        data = fetch_data()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    print("done.")

    candidates = analyze(data, args.vram, us_only=not args.all_regions, show_no_stock=args.show_no_stock)
    print_report(candidates, args.vram, us_only=not args.all_regions)


if __name__ == "__main__":
    main()
