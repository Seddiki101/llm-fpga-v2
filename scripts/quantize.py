#!/usr/bin/env python3
"""Quantize BERT-Base SQuAD to INT8 xmodel for Vitis AI deployment."""

import os, torch

# Bypass pytorch-nndct strict version check (container has torch 1.12.0, nndct wants 1.12.1)
os.environ["NNDCT_PYTORCH_VERSION_CHECK"] = "0"

from transformers import BertForQuestionAnswering
from pytorch_nndct.apis import torch_quantizer

MODEL_NAME = "csarron/bert-base-uncased-squad-v1"
SEQ_LEN = 384
OUTPUT_DIR = "/workspace/models/quantized"
CALIB_STEPS = 20


class BertQAWrapper(torch.nn.Module):
    """Wraps HuggingFace BertForQuestionAnswering to avoid aten::split.

    XIR doesn't support multi-output ops like split, so we return the raw
    logits tensor (batch, seq_len, 2) and split into start/end at runtime.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.bert = hf_model.bert
        self.qa_outputs = hf_model.qa_outputs

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        sequence_output = outputs[0]  # last_hidden_state
        logits = self.qa_outputs(sequence_output)
        return logits  # (batch, seq_len, 2)


os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Download pre-trained model from HuggingFace
print(f"[DOWNLOAD] {MODEL_NAME} from HuggingFace...")
hf_model = BertForQuestionAnswering.from_pretrained(MODEL_NAME)
hf_model.eval()

# 2. Wrap to avoid unsupported XIR ops
model = BertQAWrapper(hf_model)
model.eval()

# 3. Dummy inputs matching BERT QA signature
dummy_input = (
    torch.randint(0, 30000, (1, SEQ_LEN)),    # input_ids
    torch.ones(1, SEQ_LEN, dtype=torch.long),  # attention_mask
    torch.zeros(1, SEQ_LEN, dtype=torch.long), # token_type_ids
)

# 4. Step 1: Calibration — determines INT8 scale factors
print(f"[QUANTIZE] Step 1/2: Calibration ({CALIB_STEPS} steps)...")
quantizer = torch_quantizer("calib", model, dummy_input, output_dir=OUTPUT_DIR)
quant_model = quantizer.quant_model

for i in range(CALIB_STEPS):
    quant_model(*dummy_input)
    if (i + 1) % 5 == 0:
        print(f"  step {i + 1}/{CALIB_STEPS}")

quantizer.export_quant_config()

# 5. Step 2: Test/Export — produces the deployable xmodel
print("[QUANTIZE] Step 2/2: Exporting xmodel...")
quantizer = torch_quantizer("test", model, dummy_input, output_dir=OUTPUT_DIR)
quant_model = quantizer.quant_model
quant_model(*dummy_input)  # single forward pass to trace the graph
quantizer.export_xmodel(deploy_check=False, output_dir=OUTPUT_DIR)

# Verify output
xmodels = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".xmodel")]
if xmodels:
    print(f"[OK] Exported: {OUTPUT_DIR}/{xmodels[0]}")
else:
    print(f"[WARN] No .xmodel found. Files in {OUTPUT_DIR}:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")
