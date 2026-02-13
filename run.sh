#!/usr/bin/env bash
# Entry point — students run this from the repo root.
# Usage:  ./run.sh          (FPGA mode — default for workshop)
#         ./run.sh --cpu    (CPU-only mode for comparison)
set -e

# v3.0 required — latest (v3.5) dropped the DPUv3 compiler needed for U250
IMAGE="xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODE_FLAG=""
if [ "$1" = "--cpu" ]; then
    MODE_FLAG="--cpu"
fi

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "[ERROR] Docker not found. Ask your admin to run: sudo bash scripts/prepare-env.sh"
    exit 1
fi

# Pull image if missing
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "[PULL] Downloading Vitis AI image (~15 GB, one-time)..."
    docker pull "$IMAGE"
fi

# FPGA device passthrough (skipped in --cpu mode)
DEVICE_FLAGS=""
if [ -z "$MODE_FLAG" ]; then
    [ -d /dev/dri ] && DEVICE_FLAGS="$DEVICE_FLAGS --device /dev/dri"
    for dev in /dev/xclmgmt* /dev/xocl*; do
        [ -e "$dev" ] && DEVICE_FLAGS="$DEVICE_FLAGS --device $dev"
    done
fi

echo "[START] Launching Vitis AI container..."
docker run --rm -it \
    $DEVICE_FLAGS \
    -v /opt/xilinx:/opt/xilinx \
    -v "$SCRIPT_DIR":/workspace \
    -w /workspace \
    "$IMAGE" \
    bash -c "source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-pytorch && set -e && if [ -z '$MODE_FLAG' ]; then bash scripts/fetch-model.sh; fi && python3 scripts/demo.py $MODE_FLAG"
