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
DATA = "/data/home/Jiarun/LLaMA-Factory/Math-500/train-00000-of-00001-Math-500.parquet"
MAX_LEN = 8192  # or 16384
BATCH_SIZE = 1
LR = 1e-2
TR_LAMBDA = 1e-3
STEPS = 1500
LOG_EVERY = 1
SAVE_DIR = f"./ckpt/1_5x_perturbed_training_remove_wasted_padding_steps{STEPS}_Math_500_{MAX_LEN}"

# New hyperparameters for improved TR control
S_TARGET = 1.0  # Target scale (sigmoid(0) * 1.5 = 0.75, so we aim for ~1.0)
S_MIN = 0.5     # Minimum allowed scale
S_MAX = 1.5     # Maximum allowed scale (1.5 is the upper bound from sigmoid * 1.5)
S_MARGIN = 0.2  # Acceptable deviation (±20%)
ALPHA_REG = 1e-4  # L2 regularization on alpha parameters
WARMUP_STEPS = 300  # Gradually increase penalty weight

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
# Parametrization: W = W_base + scale * Delta
# with improved scale control
# ----------------------------
class BlendDelta(nn.Module):
    def __init__(self, delta: torch.Tensor, delta_norm2: float):
        super().__init__()
        self.register_buffer("delta", delta, persistent=False)
        # Initialize alpha to get scale ≈ 1.0 at start
        # sigmoid(0) * 1.5 = 0.75, so we need alpha ≈ 0.8 to get scale ≈ 1.0
        # sigmoid(0.8) * 1.5 ≈ 1.0
        init_alpha = 0.8
        self.alpha = nn.Parameter(torch.full((), init_alpha, dtype=torch.float32, device=delta.device))
        self.register_buffer("delta_norm2",
                             torch.tensor(delta_norm2, dtype=torch.float32, device=delta.device),
                             persistent=False)

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        # Clamp alpha to prevent extreme values
        # This limits scale to reasonable range
        clamped_alpha = torch.clamp(self.alpha, min=-5.0, max=5.0)
        scale = torch.sigmoid(clamped_alpha).to(self.delta.dtype) * 1.5
        return base_weight + scale * self.delta
    
    def get_scale(self) -> torch.Tensor:
        """Get current scale value."""
        with torch.no_grad():
            clamped_alpha = torch.clamp(self.alpha, min=-5.0, max=5.0)
            return torch.sigmoid(clamped_alpha) * 1.5


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
    inputs = [f"Question: {q}\nSolution: " for q in examples["problem"]]
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

dataset = load_dataset(DATA.split('.')[-1], data_files=DATA, split="train")
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


optimizer = torch.optim.AdamW(alpha_params, lr=LR, weight_decay=0.0)  # No default weight decay
base_model.train()


# ----------------------------
# Improved TR Penalty Computation
# ----------------------------
def compute_improved_tr_penalty(model, s_target, s_min, s_max, margin, alpha_reg, device):
    """
    Compute improved TR penalty with:
    1. Bidirectional penalty (penalize both over and under target)
    2. Hard clipping to prevent extreme values
    3. L2 regularization on alpha parameters
    """
    scale_values = []
    alpha_values = []
    penalty_terms = []
    
    for m in model.modules():
        if not hasattr(m, "parametrizations"):
            continue
        for pname, plist in getattr(m, "parametrizations", {}).items():
            for pr in plist:
                if isinstance(pr, BlendDelta):
                    # Get scale with clamping
                    clamped_alpha = torch.clamp(pr.alpha, min=-5.0, max=5.0)
                    s = torch.sigmoid(clamped_alpha) * 1.5
                    
                    # Clip scale to valid range
                    s_clipped = torch.clamp(s, min=s_min, max=s_max)
                    
                    # Bidirectional penalty with margin
                    lower_bound = s_target * (1 - margin)
                    upper_bound = s_target * (1 + margin)
                    
                    if s_clipped < lower_bound:
                        # Penalty for being too small
                        penalty = (lower_bound - s_clipped) ** 2
                    elif s_clipped > upper_bound:
                        # Stronger penalty for being too large (2x weight)
                        penalty = 2.0 * (s_clipped - upper_bound) ** 2
                    else:
                        # Within acceptable range
                        penalty = torch.tensor(0.0, device=device, dtype=s.dtype)
                    
                    penalty_terms.append(penalty)
                    scale_values.append(s_clipped.detach())
                    alpha_values.append(pr.alpha.detach())
    
    # Average penalty across all parameters
    tr_penalty = torch.stack(penalty_terms).mean() if penalty_terms else torch.tensor(0.0, device=device)
    
    # L2 regularization on alpha parameters
    alpha_reg_loss = torch.tensor(0.0, device=device)
    if alpha_values:
        alpha_reg_loss = alpha_reg * torch.stack([a ** 2 for a in alpha_values]).mean()
    
    # Statistics for logging
    stats = {}
    if scale_values:
        scale_tensor = torch.stack(scale_values)
        stats = {
            's_mean': scale_tensor.mean().item(),
            's_std': scale_tensor.std().item(),
            's_min': scale_tensor.min().item(),
            's_max': scale_tensor.max().item(),
            's_median': scale_tensor.median().item(),
        }
    
    return tr_penalty, alpha_reg_loss, stats


def train(max_steps=STEPS, tr_lambda=TR_LAMBDA, log_every=LOG_EVERY):
    step = 0
    running_lm = 0.0
    running_tr = 0.0
    running_reg = 0.0

    first_device = next((p.device for _, p in base_model.named_parameters()), torch.device("cpu"))

    while step < max_steps:
        for batch in loader:
            step += 1
            batch = {k: v.to(first_device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            out = base_model(**batch)
            lm_loss = out.loss

            # Compute improved TR penalty
            tr_penalty, alpha_reg_loss, scale_stats = compute_improved_tr_penalty(
                base_model, 
                s_target=S_TARGET,
                s_min=S_MIN,
                s_max=S_MAX,
                margin=S_MARGIN,
                alpha_reg=ALPHA_REG,
                device=first_device
            )
            
            # Warmup: gradually increase penalty weight
            warmup_factor = min(1.0, step / WARMUP_STEPS)
            weighted_tr_penalty = warmup_factor * tr_lambda * (tr_penalty + alpha_reg_loss)
            
            # Total loss
            total_loss = lm_loss + weighted_tr_penalty
            total_loss.backward()
            
            # Optional: gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(alpha_params, max_norm=1.0)
            
            optimizer.step()

            running_lm += lm_loss.item()
            running_tr += tr_penalty.item()
            running_reg += alpha_reg_loss.item()
            
            if step % log_every == 0:
                avg_lm = running_lm / log_every
                avg_tr = running_tr / log_every
                avg_reg = running_reg / log_every
                
                print(f"step {step}/{max_steps}, "
                      f"lm_loss={avg_lm:.4f}, "
                      f"tr_penalty={avg_tr:.6f}, "
                      f"reg_loss={avg_reg:.6f}, "
                      f"warmup={warmup_factor:.3f}")
                
                if scale_stats:
                    print(f"  Scale stats: "
                          f"mean={scale_stats['s_mean']:.4f} "
                          f"std={scale_stats['s_std']:.4f} "
                          f"min={scale_stats['s_min']:.4f} "
                          f"max={scale_stats['s_max']:.4f} "
                          f"median={scale_stats['s_median']:.4f}")
                
                running_lm = 0.0
                running_tr = 0.0
                running_reg = 0.0

            if step >= max_steps:
                break

print("Starting multi-GPU training with improved TR penalty control...")
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
    
    # Log final scale statistics before baking
    print("\n=== Final Scale Statistics ===")
    final_scales = []
    for module in model.modules():
        if not hasattr(module, "parametrizations"):
            continue
        for pname, plist in module.parametrizations.items():
            for pr in plist:
                if isinstance(pr, BlendDelta):
                    final_scales.append(pr.get_scale().cpu())
    
    if final_scales:
        final_scales = torch.stack(final_scales)
        print(f"Mean: {final_scales.mean().item():.4f}")
        print(f"Std: {final_scales.std().item():.4f}")
        print(f"Min: {final_scales.min().item():.4f}")
        print(f"Max: {final_scales.max().item():.4f}")
        print(f"Median: {final_scales.median().item():.4f}")
    
    # Bake parameters
    for module in model.modules():
        if not hasattr(module, "parametrizations"):
            continue
        for pname in list(module.parametrizations.keys()):
            parametrize.remove_parametrizations(module, pname, leave_parametrized=True)

    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

bake_and_save(base_model, SAVE_DIR)
print(f"Saved baked checkpoint to {SAVE_DIR}")
