#!/usr/bin/env bash
# Download BERT-Base SQuAD from Vitis AI Model Zoo and compile for Alveo U250.
# Runs inside the Vitis AI Docker container.
# First run: ~3 min (download + compile). Cached after that.
set -e

MODEL_DIR="/workspace/models"
COMPILED="$MODEL_DIR/compiled/bert_base_squad.xmodel"
ZOO_DIR="$MODEL_DIR/zoo"
ARCH_JSON="/opt/vitis_ai/compiler/arch/DPUCADF8H/U250/arch.json"
ZOO_PKG="pt_bert-base_SQuADv1.1_384_70.66G_3.0"
ZOO_URL="https://www.xilinx.com/bin/public/openDownload?filename=${ZOO_PKG}.zip"

# Skip if already compiled (persists via mounted volume)
if [ -f "$COMPILED" ]; then
    echo "[OK] Compiled model found — skipping build"
    exit 0
fi

# --- Step 1: Download from Vitis AI Model Zoo ---
if [ ! -d "$ZOO_DIR/$ZOO_PKG" ]; then
    echo "[DOWNLOAD] BERT-Base SQuAD from Vitis AI Model Zoo (~500 MB)..."
    mkdir -p "$ZOO_DIR"
    wget -q --show-progress -O /tmp/bert_zoo.zip "$ZOO_URL"
    unzip -q /tmp/bert_zoo.zip -d "$ZOO_DIR"
    rm -f /tmp/bert_zoo.zip
fi

# --- Step 2: Find quantized xmodel in the package ---
QUANT=$(find "$ZOO_DIR/$ZOO_PKG" -name "*int.xmodel" -o -name "*.xmodel" 2>/dev/null | head -1)

if [ -z "$QUANT" ]; then
    echo "[INFO] No pre-quantized xmodel in package. Running quantization..."
    # Model zoo provides code/ and float/ dirs with XIR-compatible model + weights.
    # Use the package's own quantization if available.
    QUANT_SCRIPT=$(find "$ZOO_DIR/$ZOO_PKG" -path "*/code/test/*" -name "*.py" | head -1)
    if [ -n "$QUANT_SCRIPT" ]; then
        pip install -q --only-binary :all: 'transformers==4.30.0'
        cd "$ZOO_DIR/$ZOO_PKG"
        python3 "$QUANT_SCRIPT" --quant_mode test 2>&1 || true
        cd /workspace
        QUANT=$(find "$ZOO_DIR/$ZOO_PKG" -name "*.xmodel" 2>/dev/null | head -1)
    fi
fi

if [ -z "$QUANT" ]; then
    echo "[ERROR] No quantized xmodel found. Contents of model package:"
    find "$ZOO_DIR/$ZOO_PKG" -type f | head -30
    exit 1
fi
echo "[OK] Quantized model: $QUANT"

# --- Step 3: Compile for U250 ---
echo "[COMPILE] Compiling for Alveo U250 (DPUCADF8H)... ~1-3 min"
mkdir -p "$MODEL_DIR/compiled"
vai_c_xir \
    -x "$QUANT" \
    -a "$ARCH_JSON" \
    -o "$MODEL_DIR/compiled" \
    -n bert_base_squad

echo "[OK] Compiled model ready: $COMPILED"
