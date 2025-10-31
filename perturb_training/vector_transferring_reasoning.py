import os, math, gc
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(1234)
use_cuda = torch.cuda.is_available()
param_dtype = torch.bfloat16 if use_cuda else torch.float32

# ----------------------------
# Config
# ----------------------------
BASE = "/data/home/Jiarun/LLaMA-Factory/model_from_hf/Qwen2.5-Math-7B"
POST = "/data/home/Jiarun/LLaMA-Factory/model_from_hf/DeepSeek-R1-Distill-Qwen-7B"
DATA = "ricdomolm/MATH-500/train-00000-of-00001-Math-500.parquet"
MAX_LEN = 8192  # or 16384
BATCH_SIZE = 1
LR = 1e-2
TR_LAMBDA = 1e-3
STEPS = 1500
LOG_EVERY = 1
SAVE_DIR = f"./ckpt/2x_perturbed_training_remove_wasted_padding_steps{STEPS}_gsm8k_{MAX_LEN}"

# ----------------------------
# Tokenizer
# ----------------------------
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ----------------------------
# Load sharded models on multiple GPUs
# ----------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=param_dtype, device_map="auto", low_cpu_mem_usage=True
)
post_model = AutoModelForCausalLM.from_pretrained(
    POST, torch_dtype=param_dtype, device_map="auto", low_cpu_mem_usage=True
)
base_model.eval(); post_model.eval()

# ----------------------------
# Build name->param dicts (match by identical keys)
# ----------------------------
def named_params_dict(model: nn.Module):
    return {n: p for n, p in model.named_parameters()}

base_params = named_params_dict(base_model)
post_params = named_params_dict(post_model)
assert set(base_params.keys()) == set(post_params.keys()), "Param keys mismatch."

# ----------------------------
# Parametrization: W = W_base + (tanh(alpha) + 1) * Delta
# ----------------------------
class BlendDelta(nn.Module):
    def __init__(self, delta: torch.Tensor, delta_norm2: float):
        super().__init__()
        self.register_buffer("delta", delta, persistent=False)
        # Initialize to get scale = 1.0 at start
        # tanh(0) = 0 → (0 + 1) = 1.0
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32, device=delta.device))
        self.register_buffer("delta_norm2",
                             torch.tensor(delta_norm2, dtype=torch.float32, device=delta.device),
                             persistent=False)

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        # tanh(alpha) ∈ (-1, 1) → scale ∈ (0, 2)
        scale = (torch.tanh(self.alpha) + 1.0).to(self.delta.dtype)
        return base_weight + scale * self.delta


alpha_params = []

for name, b in base_params.items():
    if (not torch.is_floating_point(b)) or (not b.requires_grad):
        continue
    p = post_params[name]
    if b.shape != p.shape:
        continue

    delta = (p.data - b.data).detach()
    b.requires_grad_(False)

    with torch.no_grad():
        delta_norm2 = delta.float().norm().item() ** 2

    blend = BlendDelta(delta, delta_norm2)
    
    mod_name, param_name = name.rsplit(".", 1)
    mod = base_model.get_submodule(mod_name)
    parametrize.register_parametrization(mod, param_name, blend)

    alpha_params.append(blend.alpha)

print(f"Registered {len(alpha_params)} trainable scalars over {sum(p.numel() for p in base_model.parameters())} weights.")

def preprocess_function(examples):
    inputs = [f"Question: {q}\Solution: " for q in examples["problem"]]
    # Use the FULL answer including reasoning steps
    targets = [str(a) for a in examples["solution"]]

    # Tokenize without padding first
    tok_inp = tok(inputs, truncation=False, add_special_tokens=True)
    tok_tgt = tok(targets, truncation=False, add_special_tokens=False)

    out_ids, out_attn, out_lbl = [], [], []
    
    for inp_ids, inp_attn, tgt_ids in zip(tok_inp["input_ids"], tok_inp["attention_mask"], tok_tgt["input_ids"]):
        # Add EOS to target
        tgt_ids = tgt_ids + [tok.eos_token_id]
        
        # Combine input + target
        full_ids = inp_ids + tgt_ids
        full_attn = inp_attn + [1] * len(tgt_ids)
        # Label: -100 for input tokens, actual token IDs for target
        full_lbl = ([-100] * len(inp_ids)) + tgt_ids
        
        # Truncate if exceeds MAX_LEN
        if len(full_ids) > MAX_LEN:
            full_ids = full_ids[:MAX_LEN]
            full_attn = full_attn[:MAX_LEN]
            full_lbl = full_lbl[:MAX_LEN]
        
        # NO PADDING HERE - will be done dynamically in collate_fn
        out_ids.append(full_ids)
        out_attn.append(full_attn)
        out_lbl.append(full_lbl)

    return {"input_ids": out_ids, "attention_mask": out_attn, "labels": out_lbl}

dataset = load_dataset("parquet", data_files=DATA, split="train")
processed = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names,
                        desc="Tokenizing Math-500 dataset")

def collate_fn(batch):
    max_len_in_batch = max(len(b["input_ids"]) for b in batch)
    
    padded_ids, padded_attn, padded_lbl = [], [], []
    
    for b in batch:
        ids = b["input_ids"]
        attn = b["attention_mask"]
        lbl = b["labels"]
        
        pad_len = max_len_in_batch - len(ids)
        
        # Pad to max length in THIS batch only
        padded_ids.append(ids + [tok.pad_token_id] * pad_len)
        padded_attn.append(attn + [0] * pad_len)
        padded_lbl.append(lbl + [-100] * pad_len)
    
    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
        "labels": torch.tensor(padded_lbl, dtype=torch.long),
    }

loader = DataLoader(processed, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=True)
print("Dataset ready with dynamic padding.")


optimizer = torch.optim.AdamW(alpha_params, lr=LR, weight_decay=0.0)
base_model.train()

def train(max_steps=STEPS, tr_lambda=TR_LAMBDA, log_every=LOG_EVERY):
    step = 0
    running = 0.0

    first_device = next((p.device for _, p in base_model.named_parameters()), torch.device("cpu"))

    while step < max_steps:
        for batch in loader:
            step += 1
            batch = {k: v.to(first_device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            out = base_model(**batch)
            loss = out.loss

            # Trust-region penalty
            s_sq_terms = []
            for m in base_model.modules():
                if not hasattr(m, "parametrizations"):
                    continue
                for pname, plist in getattr(m, "parametrizations", {}).items():
                    for pr in plist:
                        if isinstance(pr, BlendDelta):
                            s = torch.tanh(pr.alpha) + 1.0
                            term = (s * s * pr.delta_norm2).to(first_device)
                            s_sq_terms.append(term)
            
            tr_penalty = torch.stack(s_sq_terms).mean() if s_sq_terms else torch.tensor(0.0, device=first_device, dtype=loss.dtype)

            (loss + tr_lambda * tr_penalty).backward()
            optimizer.step()

            running += loss.item()
            if step % log_every == 0:
                with torch.no_grad():
                    sigmas = []
                    for m in base_model.modules():
                        if not hasattr(m, "parametrizations"): continue
                        for plist in m.parametrizations.values():
                            for pr in plist:
                                if isinstance(pr, BlendDelta):
                                    sigmas.append((torch.tanh(pr.alpha) + 1.0).detach().float().cpu())
                    smean = torch.stack(sigmas).mean().item() if sigmas else 0.0
                    smax = torch.stack(sigmas).max().item() if sigmas else 0.0
                print(f"[step {step}] loss={running/log_every:.4f}  s_mean={smean:.4f}  s_max={smax:.4f}")
                running = 0.0

            if step >= max_steps:
                break

print("Starting multi-GPU training with device_map='auto'...")
train()
print("Training finished.")

prompt = "Solve: 12 + 35. Show your reasoning and put the final answer in \\boxed{}."
inputs = tok(prompt, return_tensors="pt").to(next(iter(base_model.parameters())).device)

with torch.no_grad():
    gen = base_model.generate(
        **inputs, max_new_tokens=200, do_sample=False,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
    )
print("\n[GEN - parametrized]\n", tok.decode(gen[0], skip_special_tokens=True))

def bake_and_save(model: nn.Module, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for module in model.modules():
        if not hasattr(module, "parametrizations"):
            continue
        for pname in list(module.parametrizations.keys()):
            parametrize.remove_parametrizations(module, pname, leave_parametrized=True)

    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

bake_and_save(base_model, SAVE_DIR)
print(f"Saved baked checkpoint to {SAVE_DIR}")
