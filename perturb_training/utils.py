import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dtype(dtype_str: str, prefer_cuda: bool = True) -> torch.dtype:
    s = dtype_str.lower().strip()
    if s in ("bf16", "bfloat16"):
        # bf16 on CPU is not always supported depending on hardware/build; user likely uses CUDA anyway.
        return torch.bfloat16 if (torch.cuda.is_available() or not prefer_cuda) else torch.float32
    if s in ("fp16", "float16"):
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype '{dtype_str}'. Use bf16|fp16|fp32.")


def infer_dataset_builder(data_path: str) -> str:
    ext = os.path.splitext(data_path)[1].lower().lstrip(".")
    if ext in ("json", "jsonl"):
        return "json"
    if ext in ("parquet",):
        return "parquet"
    if ext in ("csv", "tsv"):
        return "csv"
    # fallback: let datasets try to guess; but most local files should hit the above
    return ext or "json"


def named_params_dict(model: nn.Module) -> Dict[str, nn.Parameter]:
    return {n: p for n, p in model.named_parameters()}


class BlendDelta(nn.Module):
    """
    Parametrization: W = W_base + (sigmoid(alpha) * 2.0) * delta
    """
    def __init__(self, delta: torch.Tensor, delta_norm2: float):
        super().__init__()
        self.register_buffer("delta", delta, persistent=False)
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32, device=delta.device))
        self.register_buffer(
            "delta_norm2",
            torch.tensor(delta_norm2, dtype=torch.float32, device=delta.device),
            persistent=False,
        )

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.alpha) * 2.0
        return base_weight + scale * self.delta


def register_blend_deltas(
    base_model: nn.Module,
    post_model: nn.Module,
) -> List[BlendDelta]:
    alpha_modules: List[BlendDelta] = []

    base_params = named_params_dict(base_model)
    post_params = named_params_dict(post_model)

    if set(base_params.keys()) != set(post_params.keys()):
        missing = set(base_params.keys()) ^ set(post_params.keys())
        raise ValueError(f"Param keys mismatch. Diff size={len(missing)}")

    for name, b in base_params.items():
        if (not torch.is_floating_point(b)) or (not b.requires_grad):
            continue

        p = post_params[name]
        if b.shape != p.shape:
            continue

        # Freeze base weights; train only alphas
        b.requires_grad_(False)

        # Ensure delta lives on the same device as base param to avoid device mismatch.
        # This is especially important if the two models end up with slightly different device_maps.
        with torch.no_grad():
            delta = (p.data.to(device=b.device, dtype=b.dtype) - b.data).detach()
            delta_norm2 = float((delta.float() * delta.float()).sum().item())

        blend = BlendDelta(delta, delta_norm2)

        mod_name, param_name = name.rsplit(".", 1)
        mod = base_model.get_submodule(mod_name)
        parametrize.register_parametrization(mod, param_name, blend)

        alpha_modules.append(blend)

    return alpha_modules


def build_tokenizer(model_path: str, use_fast: bool = True):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def make_preprocess_function(tok, max_len: int):
    def preprocess_function(examples):
        inputs = [str(q) for q in examples["problem"]]
        targets = [str(a) for a in examples["solution"]]

        tok_inp = tok(inputs, truncation=False, add_special_tokens=True)
        tok_tgt = tok(targets, truncation=False, add_special_tokens=False)

        out_ids, out_attn, out_lbl = [], [], []

        for inp_ids, inp_attn, tgt_ids in zip(
            tok_inp["input_ids"], tok_inp["attention_mask"], tok_tgt["input_ids"]
        ):
            tgt_ids = tgt_ids + [tok.eos_token_id]

            full_ids = inp_ids + tgt_ids
            full_attn = inp_attn + [1] * len(tgt_ids)
            full_lbl = ([-100] * len(inp_ids)) + tgt_ids

            if len(full_ids) > max_len:
                full_ids = full_ids[:max_len]
                full_attn = full_attn[:max_len]
                full_lbl = full_lbl[:max_len]

            out_ids.append(full_ids)
            out_attn.append(full_attn)
            out_lbl.append(full_lbl)

        return {"input_ids": out_ids, "attention_mask": out_attn, "labels": out_lbl}

    return preprocess_function


def make_collate_fn(tok):
    def collate_fn(batch):
        max_len_in_batch = max(len(b["input_ids"]) for b in batch)

        padded_ids, padded_attn, padded_lbl = [], [], []

        for b in batch:
            ids = b["input_ids"]
            attn = b["attention_mask"]
            lbl = b["labels"]

            pad_len = max_len_in_batch - len(ids)

            padded_ids.append(ids + [tok.pad_token_id] * pad_len)
            padded_attn.append(attn + [0] * pad_len)
            padded_lbl.append(lbl + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
            "labels": torch.tensor(padded_lbl, dtype=torch.long),
        }

    return collate_fn


def build_dataloader(
    data_path: str,
    tokenizer,
    max_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    builder = infer_dataset_builder(data_path)
    dataset = load_dataset(builder, data_files=data_path, split="train")
    processed = dataset.map(
        make_preprocess_function(tokenizer, max_len),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset...",
    )
    loader = DataLoader(
        processed,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


@contextmanager
def temporary_alpha_fill(alpha_modules: List[BlendDelta], value: float):
    saved = [m.alpha.data.clone() for m in alpha_modules]
    try:
        for m in alpha_modules:
            m.alpha.data.fill_(value)
        yield
    finally:
        for m, s in zip(alpha_modules, saved):
            m.alpha.data.copy_(s)


@torch.no_grad()
def compute_sigma_stats(alpha_modules: List[BlendDelta]) -> Tuple[float, float, float]:
    if not alpha_modules:
        return 0.0, 0.0, 0.0
    sigmas = torch.stack([(torch.sigmoid(m.alpha) * 2.0).float().cpu() for m in alpha_modules])
    return sigmas.mean().item(), sigmas.max().item(), sigmas.min().item()


def compute_tr_penalty(alpha_modules: List[BlendDelta], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not alpha_modules:
        return torch.tensor(0.0, device=device, dtype=dtype)
    terms = []
    for m in alpha_modules:
        s = torch.sigmoid(m.alpha) * 2.0
        # s^2 * ||delta||^2
        terms.append((s * s * m.delta_norm2).to(device=device, dtype=dtype))
    return torch.stack(terms).mean()


def default_save_dir(prefix: str, lr: float, steps: int, max_len: int) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"./ckpt/{prefix}_LR_{lr}_steps{steps}_Seq{max_len}_{stamp}"


def bake_and_save(model: nn.Module, tok, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    for module in model.modules():
        if not hasattr(module, "parametrizations"):
            continue
        for pname in list(module.parametrizations.keys()):
            # leave_parametrized=True bakes current parametrized value into the parameter
            parametrize.remove_parametrizations(module, pname, leave_parametrized=True)

    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)


def load_model(
    model_path: str,
    torch_dtype: torch.dtype,
    device_map,
    low_cpu_mem_usage: bool = True,
):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
