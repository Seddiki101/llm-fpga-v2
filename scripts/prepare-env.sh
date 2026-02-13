#!/usr/bin/env bash
# Admin-only setup script — run once per machine with: sudo bash scripts/prepare-env.sh
set -e

echo "=== FPGA AI Demo: Environment Setup ==="

# 1. Docker
if command -v docker &>/dev/null; then
    echo "[OK] Docker found: $(docker --version)"
else
    echo "[INSTALL] Docker..."
    curl -fsSL https://get.docker.com | sh
    echo "[OK] Docker installed"
fi

# 2. Add current user to docker group (avoids sudo for students)
if ! groups "$SUDO_USER" 2>/dev/null | grep -q docker; then
    usermod -aG docker "$SUDO_USER"
    echo "[OK] Added $SUDO_USER to docker group (re-login required)"
fi

# 3. XRT (Xilinx Runtime)
if command -v xbutil &>/dev/null; then
    echo "[OK] XRT found: $(xbutil --version 2>/dev/null | head -1)"
else
    echo "[WARN] XRT not found."
    echo "       Install from: https://www.xilinx.com/products/boards-and-kits/alveo/u250.html#gettingStarted"
    echo "       Select the deployment package matching your OS and card."
fi

# 4. Check for Alveo U250
if xbutil examine &>/dev/null 2>&1; then
    echo "[OK] FPGA card detected"
    xbutil examine 2>/dev/null | grep -i "u250" && echo "[OK] Alveo U250 confirmed" || echo "[WARN] Card found but may not be U250"
else
    echo "[WARN] No FPGA card detected — demo will still work in --cpu mode"
fi

# 5. Pull Vitis AI Docker image
echo "[PULL] Vitis AI Docker image (this may take a while)..."
docker pull xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106
echo "[OK] Docker image ready"

echo ""
echo "=== Setup complete. Students can now run: ./run.sh ==="
