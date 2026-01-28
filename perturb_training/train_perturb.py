import argparse
import os
from typing import Optional

import torch
import torch.nn.functional as F

from utils import (
    set_seed,
    parse_dtype,
    build_tokenizer,
    load_model,
    register_blend_deltas,
    build_dataloader,
    temporary_alpha_fill,
    compute_tr_penalty,
    compute_asymmetric_tr_penalty,
    compute_sigma_stats,
    compute_entropy,
    compute_kl_divergence,
    temperature_scaled_softmax,
    temperature_scaled_log_softmax,
    bake_and_save,
    default_save_dir,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Perturbed training via BlendDelta parametrizations with multiple objectives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and data paths
    p.add_argument("--base", type=str, required=True, help="Base model path (HF format).")
    p.add_argument("--post", type=str, required=True, help="Post-trained model path (HF format).")
    p.add_argument("--data", type=str, required=True, help="Dataset file path (json/jsonl/parquet/csv).")

    # Data processing
    p.add_argument("--max-len", type=int, default=8192, help="Maximum sequence length.")
    p.add_argument("--batch-size", type=int, default=1, help="Training batch size.")

    # Optimization
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay for AdamW.")
    p.add_argument("--steps", type=int, default=2000, help="Total training steps.")
    p.add_argument("--log-every", type=int, default=1, help="Log every N steps.")

    # Objective selection
    p.add_argument(
        "--objective", 
        type=str, 
        default="ce_only",
        choices=["ce_only", "two_teacher", "reward_warmstart"],
        help="Training objective: ce_only (A), two_teacher (B), or reward_warmstart (C)."
    )

    # Objective A: CE-only parameters
    p.add_argument("--ce-kl-beta", type=float, default=0.25, 
                   help="[Obj A] KL divergence weight for CE-only objective.")
    p.add_argument("--ce-tr-lambda", type=float, default=1e-3,
                   help="[Obj A] Trust-region penalty weight for CE-only.")

    # Objective B: Two-teacher distillation parameters
    p.add_argument("--gamma", type=float, default=0.6,
                   help="[Obj B] Weight for post-teacher (1-gamma for base-teacher).")
    p.add_argument("--temp-post", type=float, default=2.0,
                   help="[Obj B] Temperature for post-teacher distillation.")
    p.add_argument("--temp-base", type=float, default=2.0,
                   help="[Obj B] Temperature for base-teacher distillation.")
    p.add_argument("--temp-student", type=float, default=1.0,
                   help="[Obj B] Temperature for student (current model).")
    p.add_argument("--entropy-beta", type=float, default=0.01,
                   help="[Obj B] Entropy bonus weight.")
    p.add_argument("--nll-lambda", type=float, default=0.05,
                   help="[Obj B] Hard NLL anchor weight (0 to disable).")

    # Objective C: Reward warm-start parameters
    p.add_argument("--kl-lambda", type=float, default=0.01,
                   help="[Obj C] KL-to-reference weight for reward optimization.")
    p.add_argument("--reward-type", type=str, default="ce_based",
                   choices=["ce_based", "verifier"],
                   help="[Obj C] Type of reward signal.")
    p.add_argument("--length-penalty", type=float, default=0.0,
                   help="[Obj C] Penalty per token to encourage efficiency.")

    # Trust-region options (all objectives)
    p.add_argument("--use-asymmetric-tr", action="store_true",
                   help="Use asymmetric trust-region penalty (penalize s>1 more).")
    p.add_argument("--extrapolation-penalty", type=float, default=2.0,
                   help="Multiplier for s>1 in asymmetric TR penalty.")
    p.add_argument("--clamp-sigma", action="store_true",
                   help="Clamp sigma values to [0, 2] after each step.")

    # System
    p.add_argument("--seed", type=int, default=1234, help="Random seed.")
    p.add_argument("--dtype", type=str, default="bf16", help="Data type: bf16|fp16|fp32.")
    p.add_argument("--device-map", type=str, default="auto", help='Device map (typically "auto").')
    p.add_argument("--no-low-cpu-mem-usage", action="store_true",
                   help="Disable low_cpu_mem_usage for model loading.")

    # Saving
    p.add_argument("--save-dir", type=str, default="", help="Save directory (auto-generated if empty).")
    p.add_argument("--save-prefix", type=str, default="perturbed_ckpt", help="Prefix for save directory.")

    # Generation test
    p.add_argument("--prompt", type=str, 
                   default="Solve: 12 + 35. Show your reasoning and put the final answer in \\boxed{}.",
                   help="Test prompt for generation.")
    p.add_argument("--max-new-tokens", type=int, default=200, help="Max tokens for test generation.")

    # Monitoring
    p.add_argument("--track-entropy", action="store_true",
                   help="Track and log token-level entropy (slight overhead).")

    return p.parse_args()


def objective_a_ce_only(
    args,
    base_model,
    alpha_modules,
    batch,
    first_device,
):
    """
    Objective A: CE + KL(perturbed||base) + TR penalty
    This is the baseline "SFT in low-dim parameterization"
    """
    # 1) Perturbed forward
    out = base_model(**batch)
    logits_perturbed = out.logits
    loss_ce = out.loss

    # 2) Base forward (no grad)
    with torch.no_grad():
        with temporary_alpha_fill(alpha_modules, -100.0):
            out_base = base_model(**batch)
            logits_base = out_base.logits

    # 3) KL(perturbed||base)
    logp_pert = F.log_softmax(logits_perturbed, dim=-1)
    p_base = F.softmax(logits_base, dim=-1)
    loss_kl = F.kl_div(logp_pert, p_base, reduction="batchmean")

    # 4) Trust-region penalty
    if args.use_asymmetric_tr:
        tr_penalty = compute_asymmetric_tr_penalty(
            alpha_modules, first_device, loss_ce.dtype, args.extrapolation_penalty
        )
    else:
        tr_penalty = compute_tr_penalty(alpha_modules, first_device, loss_ce.dtype)

    # 5) Total loss
    loss_total = loss_ce + args.ce_kl_beta * loss_kl + args.ce_tr_lambda * tr_penalty

    # 6) Entropy (optional)
    entropy_val = 0.0
    if args.track_entropy:
        mask = (batch["labels"] != -100).float()
        entropy_val = compute_entropy(logits_perturbed, mask)

    metrics = {
        "loss_total": loss_total.item(),
        "loss_ce": loss_ce.item(),
        "loss_kl": (args.ce_kl_beta * loss_kl).item(),
        "tr_penalty": tr_penalty.item(),
        "entropy": entropy_val,
    }

    return loss_total, metrics


def objective_b_two_teacher(
    args,
    base_model,
    post_model,
    alpha_modules,
    batch,
    first_device,
):
    """
    Objective B: Two-teacher distillation + entropy bonus + optional NLL anchor
    L = gamma * CE(q_post^T_p, p_alpha^T_s) + (1-gamma) * CE(q_base^T_b, p_alpha^T_s) 
        - beta * H(p_alpha) + [optional] lambda_nll * CE_hard
    """
    # Forward through current model
    out = base_model(**batch)
    logits_student = out.logits
    loss_ce_hard = out.loss  # for optional NLL anchor

    # Get base and post logits (no grad)
    with torch.no_grad():
        # Base logits
        with temporary_alpha_fill(alpha_modules, -100.0):
            out_base = base_model(**batch)
            logits_base = out_base.logits

        # Post logits (need to forward through post_model)
        # Note: post_model might have different device_map, so we need to be careful
        logits_post = post_model(**batch).logits

    # Temperature-scaled distributions
    # Student log-probs at T_s
    log_p_student = temperature_scaled_log_softmax(logits_student, args.temp_student)
    
    # Base teacher probs at T_b
    p_base = temperature_scaled_softmax(logits_base, args.temp_base)
    
    # Post teacher probs at T_p
    p_post = temperature_scaled_softmax(logits_post, args.temp_post)

    # Compute distillation losses (cross-entropy between distributions)
    # CE(q, p) = -sum(q * log(p))
    # We need to mask out padding tokens
    mask = (batch["labels"] != -100).float().unsqueeze(-1)  # [batch, seq, 1]

    # Loss with post teacher
    ce_post = -(p_post * log_p_student).sum(dim=-1)  # [batch, seq]
    ce_post = (ce_post * mask.squeeze(-1)).sum() / mask.sum()

    # Loss with base teacher
    ce_base = -(p_base * log_p_student).sum(dim=-1)  # [batch, seq]
    ce_base = (ce_base * mask.squeeze(-1)).sum() / mask.sum()

    # Weighted combination
    loss_distill = args.gamma * ce_post + (1 - args.gamma) * ce_base

    # Entropy bonus (negative because we want to maximize entropy)
    p_student = F.softmax(logits_student, dim=-1)
    entropy = -(p_student * log_p_student).sum(dim=-1)  # [batch, seq]
    entropy_mean = (entropy * mask.squeeze(-1)).sum() / mask.sum()
    loss_entropy = -args.entropy_beta * entropy_mean  # negative for bonus

    # Optional NLL anchor
    loss_nll = 0.0
    if args.nll_lambda > 0:
        loss_nll = args.nll_lambda * loss_ce_hard

    # Total loss
    loss_total = loss_distill + loss_entropy + loss_nll

    metrics = {
        "loss_total": loss_total.item(),
        "loss_distill": loss_distill.item(),
        "ce_post": ce_post.item(),
        "ce_base": ce_base.item(),
        "entropy": entropy_mean.item(),
        "loss_nll": loss_nll if isinstance(loss_nll, float) else loss_nll.item(),
    }

    return loss_total, metrics


def objective_c_reward_warmstart(
    args,
    base_model,
    alpha_modules,
    batch,
    first_device,
):
    """
    Objective C: Reward optimization + KL-to-reference
    max E[R(x,y)] - lambda * KL(pi_alpha || pi_base)
    
    For simplicity, we use negative CE as reward (higher is better).
    In production, you'd use a verifier or more sophisticated reward.
    """
    # 1) Forward through current policy
    out = base_model(**batch)
    logits_alpha = out.logits
    loss_ce = out.loss

    # 2) Compute reward (negative CE as proxy)
    if args.reward_type == "ce_based":
        # Negative CE: lower CE = higher reward
        reward = -loss_ce
        
        # Optional length penalty
        if args.length_penalty > 0:
            num_tokens = (batch["labels"] != -100).sum().float()
            reward = reward - args.length_penalty * num_tokens / batch["input_ids"].size(0)
    else:
        # Placeholder for verifier-based reward
        # In practice, you'd call a verifier model here
        reward = -loss_ce

    # 3) KL divergence to base policy
    with torch.no_grad():
        with temporary_alpha_fill(alpha_modules, -100.0):
            out_base = base_model(**batch)
            logits_base = out_base.logits

    mask = (batch["labels"] != -100).float()
    kl_div = compute_kl_divergence(logits_alpha, logits_base, mask, reduction="batchmean")

    # 4) Total loss (negative because we maximize reward)
    loss_total = -reward + args.kl_lambda * kl_div

    # 5) Entropy tracking
    entropy_val = 0.0
    if args.track_entropy:
        entropy_val = compute_entropy(logits_alpha, mask)

    metrics = {
        "loss_total": loss_total.item(),
        "reward": reward.item(),
        "kl_div": kl_div.item(),
        "entropy": entropy_val,
        "ce_proxy": loss_ce.item(),
    }

    return loss_total, metrics


def main():
    args = parse_args()

    # Setup
    set_seed(args.seed)
    torch_dtype = parse_dtype(args.dtype, prefer_cuda=True)
    low_cpu_mem_usage = not args.no_low_cpu_mem_usage

    print(f"Training Configuration: ")
    print(f"Objective: {args.objective}")
    print(f"Base model: {args.base}")
    print(f"Post model: {args.post}")
    print(f"Data: {args.data}")
    print(f"Steps: {args.steps}, LR: {args.lr}, Batch size: {args.batch_size}")
    print(f"Max length: {args.max_len}, Dtype: {args.dtype}")
    print(f"Device map: {args.device_map}")

    # Tokenizer
    tok = build_tokenizer(args.post, use_fast=True)

    # Load models
    print("Loading base model...")
    base_model = load_model(args.base, torch_dtype=torch_dtype, 
                           device_map=args.device_map, 
                           low_cpu_mem_usage=low_cpu_mem_usage)
    base_model.eval()

    print("Loading post model...")
    device_map_for_post = getattr(base_model, "hf_device_map", args.device_map)
    post_model = load_model(args.post, torch_dtype=torch_dtype, 
                           device_map=device_map_for_post, 
                           low_cpu_mem_usage=low_cpu_mem_usage)
    post_model.eval()

    # Register parametrizations
    print("Registering blend deltas...")
    alpha_modules = register_blend_deltas(base_model, post_model)
    total_weights = sum(p.numel() for p in base_model.parameters())
    print(f"Registered {len(alpha_modules)} trainable scalars over {total_weights:,} weights.")

    # Data
    print("Loading dataset...")
    loader = build_dataloader(
        data_path=args.data,
        tokenizer=tok,
        max_len=args.max_len,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    print(f"Dataset ready: {len(loader)} batches")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [m.alpha for m in alpha_modules], 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Training mode
    base_model.train()
    
    # Device for inputs
    first_device = next((p.device for p in base_model.parameters()), torch.device("cpu"))

    # Training loop
    print(f"Starting Training ({args.objective})...")
    
    running_metrics = {}
    step = 0

    print(f"Starting training ({args.device_map} sharded model)...")

    while step < args.steps:
        for batch in loader:
            step += 1
            batch = {k: v.to(first_device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            # Compute loss based on objective
            if args.objective == "ce_only":
                loss_total, metrics = objective_a_ce_only(
                    args, base_model, alpha_modules, batch, first_device
                )
            elif args.objective == "two_teacher":
                loss_total, metrics = objective_b_two_teacher(
                    args, base_model, post_model, alpha_modules, batch, first_device
                )
            elif args.objective == "reward_warmstart":
                loss_total, metrics = objective_c_reward_warmstart(
                    args, base_model, alpha_modules, batch, first_device
                )
            else:
                raise ValueError(f"Unknown objective: {args.objective}")

            # Backward and optimize
            loss_total.backward()
            optimizer.step()

            # Optional: clamp sigma values
            if args.clamp_sigma:
                with torch.no_grad():
                    for m in alpha_modules:
                        # sigmoid(alpha) * 2 should be in [0, 2]
                        # So alpha should be in [-inf, inf] but we can soft-clamp
                        # by clamping sigmoid(alpha) to [0, 1]
                        # Actually, sigmoid is already in [0,1], so *2 gives [0,2]
                        # If we want to clamp, we need to clamp alpha itself
                        # Let's clamp alpha to reasonable range
                        m.alpha.data.clamp_(-10, 10)  # sigmoid(-10) ≈ 0, sigmoid(10) ≈ 1

            # Accumulate metrics
            for k, v in metrics.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + v

            # Logging
            if step % args.log_every == 0:
                smean, smax, smin = compute_sigma_stats(alpha_modules)
                
                log_str = f"Step {step}/{args.steps} | "
                for k, v in running_metrics.items():
                    log_str += f"{k}: {v/args.log_every:.4f} | "
                log_str += f"σ_mean: {smean:.4f} | σ_max: {smax:.4f} | σ_min: {smin:.4f}"
                
                print(log_str)
                running_metrics = {}

            if step >= args.steps:
                break

    print("Training Finished...")

    # Generation test
    print("Generation Test...")
    base_model.eval()
    gen_device = next(iter(base_model.parameters())).device
    inputs = tok(args.prompt, return_tensors="pt").to(gen_device)

    with torch.no_grad():
        gen = base_model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    print("\n[Generated Output]")
    print(tok.decode(gen[0], skip_special_tokens=True))
    print()

    # Save
    save_dir = args.save_dir.strip()
    if not save_dir:
        save_dir = default_save_dir(args.save_prefix, args.objective, args.lr, args.steps, args.max_len)

    print(f"Saving checkpoint to {save_dir}...")
    bake_and_save(base_model, tok, save_dir)
    print(f"Saved successfully.")

    # Save training config
    import json
    config_path = os.path.join(save_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved training config to {config_path}")


if __name__ == "__main__":
    main()
