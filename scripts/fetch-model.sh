#!/usr/bin/env bash
# Download BERT-Base SQuAD from Vitis AI Model Zoo and compile for Alveo U250.
# Runs inside the Vitis AI Docker container.
# First run: ~3 min (download + compile). Cached after that.
set -e

MODEL_DIR="/workspace/models"
COMPILED="$MODEL_DIR/compiled/bert_base_squad.xmodel"
ZOO_DIR="$MODEL_DIR/zoo"
ARCH_JSON="/opt/vitis_ai/compiler/arch/DPUCADF8H/U250/arch.json"
SEQ_LEN=384
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
    # Use transformers 4.26.1 — compatible with Python 3.7, torch 1.12.x,
    # and does NOT require accelerate for Trainer (4.30+ does).
    pip install -q --only-binary :all: 'transformers==4.26.1' 'datasets' 'evaluate' 'accelerate==0.20.1'
    [ -f "$ZOO_DIR/$ZOO_PKG/requirement.txt" ] && pip install -q -r "$ZOO_DIR/$ZOO_PKG/requirement.txt" || true

    # Download float model from HuggingFace if the float/ dir is empty (only .keep)
    FLOAT_DIR="$ZOO_DIR/$ZOO_PKG/float"
    if [ -z "$(find "$FLOAT_DIR" -name "*.bin" -o -name "*.safetensors" -o -name "*.pt" 2>/dev/null | head -1)" ]; then
        echo "[INFO] Float model weights missing — downloading from HuggingFace..."
        python3 -c "
from transformers import BertForQuestionAnswering, BertTokenizer
m = 'csarron/bert-base-uncased-squad-v1'
BertForQuestionAnswering.from_pretrained(m).save_pretrained('$FLOAT_DIR')
BertTokenizer.from_pretrained(m).save_pretrained('$FLOAT_DIR')
print('[OK] Float model saved to $FLOAT_DIR')
"
    fi

    # Try 1: Use the package's own run_quant.sh (the intended entry point)
    if [ -f "$ZOO_DIR/$ZOO_PKG/run_quant.sh" ]; then
        echo "[INFO] Running zoo quantization via run_quant.sh..."
        cd "$ZOO_DIR/$ZOO_PKG"
        [ -f env_setup.sh ] && source env_setup.sh || true
        bash run_quant.sh 2>&1 || true
        cd /workspace
        QUANT=$(find "$ZOO_DIR/$ZOO_PKG" -name "*.xmodel" 2>/dev/null | head -1)
    fi

    # Try 2: Call run_qa.py directly with required arguments (skip _qat variant)
    if [ -z "$QUANT" ]; then
        QUANT_SCRIPT="$ZOO_DIR/$ZOO_PKG/code/run_qa.py"
        if [ -f "$QUANT_SCRIPT" ]; then
            echo "[INFO] Running zoo script directly: $QUANT_SCRIPT"
            cd "$ZOO_DIR/$ZOO_PKG"
            python3 "$QUANT_SCRIPT" \
                --model_name_or_path "$FLOAT_DIR" \
                --dataset_name squad \
                --output_dir "$ZOO_DIR/$ZOO_PKG/quantized" \
                --do_eval \
                --per_device_eval_batch_size 1 \
                --max_seq_length $SEQ_LEN \
                --quant_mode test 2>&1 || true
            cd /workspace
            QUANT=$(find "$ZOO_DIR/$ZOO_PKG" -name "*.xmodel" 2>/dev/null | head -1)
        fi
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
