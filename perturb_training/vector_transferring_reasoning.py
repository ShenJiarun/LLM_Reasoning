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
DATA = "/data/home/Jiarun/LLaMA-Factory/gsm8k_dataset/train-00000-of-00001.parquet"
MAX_LEN = 2048
BATCH_SIZE = 1
LR = 1e-5
TR_LAMBDA = 1e-3
STEPS = 1500
LOG_EVERY = 1
SAVE_DIR = "./perturbed_from_alpha_multi"

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
# Parametrization: W = W_base + sigmoid(alpha) * Delta
# We register one tiny scalar alpha per tensor, living on the same device as that tensor.
# ----------------------------
class BlendDelta(nn.Module):
    def __init__(self, delta: torch.Tensor, delta_norm2: float):
        super().__init__()
        self.register_buffer("delta", delta, persistent=False)         # stays on param's shard/device
        self.alpha = nn.Parameter(torch.zeros((), dtype=delta.dtype, device=delta.device))  # scalar
        # Store delta_norm2 as a buffer on the same device
        self.register_buffer("delta_norm2", torch.tensor(delta_norm2, dtype=delta.dtype, device=delta.device), persistent=False)
    
    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        # base_weight is the "unconstrained" param; we freeze it, and add scaled delta at forward.
        return base_weight + torch.sigmoid(self.alpha) * self.delta

# Freeze all original params; compute and register parametrizations
alpha_params = []                 # collect alpha scalars for the optimizer

for name, b in base_params.items():
    # Only float trainable tensors
    if (not torch.is_floating_point(b)) or (not b.requires_grad):
        continue
    p = post_params[name]
    if b.shape != p.shape:
        continue

    # Delta on the correct shard/device
    delta = (p.data - b.data).detach()
    # Freeze base weight (we train only the alpha scalars)
    b.requires_grad_(False)

    # Compute delta norm^2
    with torch.no_grad():
        delta_norm2 = delta.float().norm().item() ** 2

    # Create the parametrization object (now stores delta_norm2)
    blend = BlendDelta(delta, delta_norm2)
    
    # Register parametrization on the owning module
    mod_name, param_name = name.rsplit(".", 1)
    mod = base_model.get_submodule(mod_name)
    parametrize.register_parametrization(mod, param_name, blend)

    # Track alpha
    alpha_params.append(blend.alpha)

print(f"Registered {len(alpha_params)} trainable scalars over {sum(p.numel() for p in base_model.parameters())} weights.")

# ----------------------------
# Data: GSM8K SFT-style (train on answer only)
# ----------------------------
def preprocess_function(examples):
    inputs = [f"Question: {q}\nAnswer: " for q in examples["question"]]
    # final answer after '####'
    targets = [str(a).split("####")[-1].strip() for a in examples["answer"]]

    tok_inp = tok(inputs, truncation=True, max_length=1024, add_special_tokens=True)
    tok_tgt = tok(targets, truncation=True, max_length=1024, add_special_tokens=False)

    out_ids, out_attn, out_lbl = [], [], []
    for inp_ids, inp_attn, tgt_ids in zip(tok_inp["input_ids"], tok_inp["attention_mask"], tok_tgt["input_ids"]):
        tgt_ids = tgt_ids + [tok.eos_token_id]
        room = MAX_LEN - len(tgt_ids)
        if room <= 0:
            tgt_ids = tgt_ids[-MAX_LEN:]
            inp_ids, inp_attn = [], []
            room = 0
        if len(inp_ids) > room:
            inp_ids = inp_ids[-room:]
            inp_attn = inp_attn[-room:]
        full_ids = inp_ids + tgt_ids
        full_attn = inp_attn + [1]*len(tgt_ids)
        full_lbl = ([-100]*len(inp_ids)) + tgt_ids

        pad = MAX_LEN - len(full_ids)
        if pad > 0:
            full_ids += [tok.pad_token_id]*pad
            full_attn += [0]*pad
            full_lbl += [-100]*pad

        out_ids.append(full_ids)
        out_attn.append(full_attn)
        out_lbl.append(full_lbl)

    return {"input_ids": out_ids, "attention_mask": out_attn, "labels": out_lbl}

dataset = load_dataset("parquet", data_files=DATA, split="train")
processed = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names,
                        desc="Tokenizing GSM8K")

def collate_fn(batch):
    return {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }

loader = DataLoader(processed, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=True)
print("Dataset ready.")

# ----------------------------
# Optimizer over alpha scalars only
# (they sit on many devices; PyTorch can still step them in one process)
# ----------------------------
optimizer = torch.optim.AdamW(alpha_params, lr=LR, weight_decay=0.0)

# ----------------------------
# Training
# ----------------------------
base_model.train()  # dropout off by default for CausalLMs, but use train() to allow grads

def train(max_steps=STEPS, tr_lambda=TR_LAMBDA, log_every=LOG_EVERY):
    step = 0
    running = 0.0

    # Find the *first* device the model uses (for input placement).
    # For dispatch models, it's usually the first CUDA device.
    first_device = next((p.device for _, p in base_model.named_parameters()), torch.device("cpu"))

    while step < max_steps:
        for batch in loader:
            step += 1
            # Place inputs on the first device; HF dispatch will route as needed.
            batch = {k: v.to(first_device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            out = base_model(**batch)
            loss = out.loss

            # Trust-region penalty: mean(s^2 * ||Δ||^2)
            # Collect all alphas in registration order, moving to first_device
            s_sq_terms = []
            for m in base_model.modules():
                # walk every parametrized param; append their sigmoid(alpha)^2 * ||Δ||^2
                if not hasattr(m, "parametrizations"):
                    continue
                for pname, plist in getattr(m, "parametrizations", {}).items():
                    for pr in plist:
                        if isinstance(pr, BlendDelta):
                            s = torch.sigmoid(pr.alpha)
                            # Compute s^2 * delta_norm2 on the parameter's device, then move to first_device
                            term = (s * s * pr.delta_norm2).to(first_device)
                            s_sq_terms.append(term)
            
            tr_penalty = torch.stack(s_sq_terms).mean() if s_sq_terms else torch.tensor(0.0, device=first_device, dtype=loss.dtype)

            (loss + tr_lambda * tr_penalty).backward()
            optimizer.step()

            running += loss.item()
            if step % log_every == 0:
                # quick stats
                with torch.no_grad():
                    sigmas = []
                    for m in base_model.modules():
                        if not hasattr(m, "parametrizations"): continue
                        for plist in m.parametrizations.values():
                            for pr in plist:
                                if isinstance(pr, BlendDelta):
                                    sigmas.append(torch.sigmoid(pr.alpha).detach().float().cpu())
                    smean = torch.stack(sigmas).mean().item() if sigmas else 0.0
                    smax = torch.stack(sigmas).max().item() if sigmas else 0.0
                print(f"[step {step}] loss={running/log_every:.4f}  s_mean={smean:.4f}  s_max={smax:.4f}")
                running = 0.0

            if step >= max_steps:
                break

print("Starting multi-GPU training with device_map='auto'...")
train()
print("Training finished.")

# ----------------------------
# (A) Direct generation without baking (parametrizations remain active)
# ----------------------------
prompt = "Solve: 12 + 35. Show your reasoning and put the final answer in \\boxed{}."
inputs = tok(prompt, return_tensors="pt").to(next(iter(base_model.parameters())).device)

with torch.no_grad():
    gen = base_model.generate(
        **inputs, max_new_tokens=200, do_sample=False,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
    )
print("\n[GEN - parametrized]\n", tok.decode(gen[0], skip_special_tokens=True))

# ----------------------------
# (B) Optional: bake weights and save a normal HF checkpoint
# ----------------------------
def bake_and_save(model: nn.Module, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    # Replace parametrized weights with their current values
    for module in model.modules():
        if not hasattr(module, "parametrizations"):
            continue
        for pname in list(module.parametrizations.keys()):
            # leave_parametrized=True writes the composed weight into the parameter
            parametrize.remove_parametrizations(module, pname, leave_parametrized=True)

    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

bake_and_save(base_model, SAVE_DIR)
print(f"Saved baked checkpoint to {SAVE_DIR}")

# Test baked model with standard load (also works with device_map='auto')
baked = AutoModelForCausalLM.from_pretrained(SAVE_DIR, torch_dtype=param_dtype, device_map="auto")
baked.eval()
with torch.no_grad():
    gen2 = baked.generate(
        **tok(prompt, return_tensors="pt").to(next(iter(baked.parameters())).device),
        max_new_tokens=200, do_sample=False,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
    )
print("\n[GEN - baked]\n", tok.decode(gen2[0], skip_special_tokens=True))
