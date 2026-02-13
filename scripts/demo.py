#!/usr/bin/env python3
"""FPGA-Accelerated BERT QA Chatbot Demo — Alveo U250"""

import sys, time, argparse
import numpy as np

# ----- Configuration -----
XMODEL = "/workspace/models/compiled/bert_base_squad.xmodel"
HF_MODEL = "csarron/bert-base-uncased-squad-v1"
SEQ_LEN = 384

CONTEXT = (
    "The Alveo U250 is an FPGA accelerator card by AMD-Xilinx with 64GB DDR4 memory. "
    "It uses the UltraScale+ architecture and is optimized for machine learning inference. "
    "Vitis AI is the development platform that compiles neural networks into DPU instructions. "
    "The DPU (Deep Learning Processing Unit) is a programmable inference engine on the FPGA fabric. "
    "BERT is a transformer model by Google, fine-tuned on the SQuAD dataset for question answering. "
    "The model reads a context paragraph and a question, then predicts the answer span in the text."
)

# ----- FPGA Inference (Vitis AI / VART) -----

def load_fpga_runner(xmodel_path):
    """Load compiled xmodel, return a VART DPU runner."""
    import xir, vart
    graph = xir.Graph.deserialize(xmodel_path)
    subs = graph.get_root_subgraph().toposort_child_subgraph()
    dpu = [s for s in subs if s.has_attr("device") and s.get_attr("device") == "DPU"][0]
    return vart.Runner.create_runner(dpu, "run")


def fpga_infer(runner, input_ids, attention_mask, token_type_ids):
    """Run one forward pass on the FPGA and return (start_logits, end_logits)."""
    in_t = runner.get_input_tensors()
    out_t = runner.get_output_tensors()

    bufs_in = [np.zeros(t.dims, dtype=np.int8) for t in in_t]
    bufs_out = [np.zeros(t.dims, dtype=np.int8) for t in out_t]

    arrays = [input_ids, attention_mask, token_type_ids]
    for i, t in enumerate(in_t):
        fp = t.get_attr("fix_point")
        data = arrays[i].flatten()[: bufs_in[i].size]
        bufs_in[i].flat[: len(data)] = np.clip(data * (2 ** fp), -128, 127).astype(np.int8)

    job = runner.execute_async(bufs_in, bufs_out)
    runner.wait(job)

    # Model outputs single tensor (batch, seq_len, 2) — split here
    fp = out_t[0].get_attr("fix_point")
    logits = bufs_out[0].astype(np.float32) / (2 ** fp)
    logits = logits.reshape(-1, 2)  # (seq_len, 2)
    return logits[:, 0], logits[:, 1]


# ----- CPU Inference (PyTorch / HuggingFace) -----

def load_cpu_model():
    """Load BERT-QA model on CPU via HuggingFace (auto-downloads ~1.3GB)."""
    from transformers import BertForQuestionAnswering
    print(f"  Downloading {HF_MODEL} from HuggingFace...")
    model = BertForQuestionAnswering.from_pretrained(HF_MODEL)
    model.eval()
    return model


def cpu_infer(model, input_ids, attention_mask, token_type_ids):
    """Run one forward pass on CPU and return (start_logits, end_logits)."""
    import torch
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention_mask),
            token_type_ids=torch.tensor(token_type_ids),
        )
    return out.start_logits.numpy().flatten(), out.end_logits.numpy().flatten()


# ----- Shared QA logic -----

def answer_question(infer_fn, tokenizer, context, question):
    """Tokenize, run inference, decode answer span."""
    enc = tokenizer(
        question, context,
        max_length=SEQ_LEN, padding="max_length", truncation=True,
        return_tensors="np",
    )
    ids = enc["input_ids"]
    mask = enc["attention_mask"]
    tids = enc["token_type_ids"]

    t0 = time.time()
    start_logits, end_logits = infer_fn(ids, mask, tids)
    ms = (time.time() - t0) * 1000

    seq_len = int(mask.sum())
    s = int(np.argmax(start_logits[:seq_len]))
    e = int(np.argmax(end_logits[:seq_len])) + 1
    answer = tokenizer.decode(ids[0][s:e], skip_special_tokens=True)
    return answer or "(no answer found in context)", ms


# ----- Main -----

def main():
    from transformers import BertTokenizer

    parser = argparse.ArgumentParser(description="FPGA QA Chatbot")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of FPGA")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(HF_MODEL)

    if args.cpu:
        print("Loading BERT-Base on CPU (PyTorch)...")
        model = load_cpu_model()
        infer_fn = lambda ids, mask, tids: cpu_infer(model, ids, mask, tids)
        device_tag = "CPU"
    else:
        print("Loading BERT-Base on FPGA (Vitis AI)...")
        runner = load_fpga_runner(XMODEL)
        infer_fn = lambda ids, mask, tids: fpga_infer(runner, ids, mask, tids)
        device_tag = "FPGA"

    print(f"\n{'=' * 46}")
    print(f"  FPGA-Accelerated QA Chatbot (Alveo U250)")
    print(f"  Mode: {device_tag} | Context: {len(CONTEXT.split())} words")
    print(f"{'=' * 46}")
    print("Type 'quit' to exit.\n")

    while True:
        q = input("You: ").strip()
        if not q or q.lower() in ("quit", "exit"):
            break
        answer, ms = answer_question(infer_fn, tokenizer, CONTEXT, q)
        print(f"Bot: {answer}  [{ms:.0f}ms on {device_tag}]\n")

    print("Goodbye.")


if __name__ == "__main__":
    main()
