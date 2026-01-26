import argparse
import os

import torch
import torch.nn.functional as F

from perturb_training.utils import (
    set_seed,
    parse_dtype,
    build_tokenizer,
    load_model,
    register_blend_deltas,
    build_dataloader,
    temporary_alpha_fill,
    compute_tr_penalty,
    compute_sigma_stats,
    bake_and_save,
    default_save_dir,
)


def parse_args():
    p = argparse.ArgumentParser(description="Perturbed training via BlendDelta parametrizations (train alphas only).")

    p.add_argument("--base", type=str, required=True, help="Base model path (HF format).")
    p.add_argument("--post", type=str, required=True, help="Post-trained model path (HF format).")
    p.add_argument("--data", type=str, required=True, help="Dataset file path (json/jsonl/parquet/csv).")

    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=1)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--tr-lambda", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.25)

    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=1)

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", type=str, default="bf16", help="bf16|fp16|fp32")

    p.add_argument("--device-map", type=str, default="auto", help='Typically "auto".')
    p.add_argument("--no-low-cpu-mem-usage", action="store_true")

    p.add_argument("--save-dir", type=str, default="", help="If empty, auto-generate.")
    p.add_argument("--save-prefix", type=str, default="perturbed_ckpt")

    p.add_argument("--prompt", type=str, default="Solve: 12 + 35. Show your reasoning and put the final answer in \\boxed{}.")
    p.add_argument("--max-new-tokens", type=int, default=200)

    return p.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    torch_dtype = parse_dtype(args.dtype, prefer_cuda=True)

    low_cpu_mem_usage = not args.no_low_cpu_mem_usage

    # Tokenizer: keep as your original behavior (use POST tokenizer)
    tokenizer = build_tokenizer(args.post, use_fast=True)

    # Load base model first with device_map, then reuse the computed map for post model if available.
    base_model = load_model(args.base, torch_dtype=torch_dtype, device_map=args.device_map, low_cpu_mem_usage=low_cpu_mem_usage)
    base_model.eval()

    # If accelerate created a concrete map, reuse it to reduce device mismatch risk.
    device_map_for_post = getattr(base_model, "hf_device_map", args.device_map)
    post_model = load_model(args.post, torch_dtype=torch_dtype, device_map=device_map_for_post, low_cpu_mem_usage=low_cpu_mem_usage)
    post_model.eval()

    # Register parametrizations on the base_model
    alpha_modules = register_blend_deltas(base_model, post_model)
    total_weights = sum(p.numel() for p in base_model.parameters())
    print(f"Registered {len(alpha_modules)} trainable scalars over {total_weights} weights.")

    # Data
    loader = build_dataloader(
        data_path=args.data,
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    print("Dataset ready with dynamic padding.")

    # Optimizer over alphas only
    optimizer = torch.optim.AdamW([m.alpha for m in alpha_modules], lr=args.lr, weight_decay=args.weight_decay)
    base_model.train()

    # Choose a device for inputs (common approach when model is sharded)
    first_device = next((p.device for p in base_model.parameters()), torch.device("cpu"))

    running = 0.0
    step = 0

    print(f"Starting training ({args.device_map} sharded model)...")

    while step < args.steps:
        for batch in loader:
            step += 1
            batch = {k: v.to(first_device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            # 1) Perturbed forward (grad)
            out = base_model(**batch)
            logits_perturbed = out.logits
            loss_ce = out.loss

            # 2) Base forward (no grad) by forcing alpha -> ~0 scale
            with torch.no_grad():
                with temporary_alpha_fill(alpha_modules, -100.0):  # sigmoid(-100) ~ 0
                    out_base = base_model(**batch)
                    logits_base = out_base.logits

            # 3) KL loss (input must be log-probs; target probs)
            logp_pert = F.log_softmax(logits_perturbed, dim=-1)
            p_base = F.softmax(logits_base, dim=-1)
            loss_kl = F.kl_div(logp_pert, p_base, reduction="batchmean")

            # 4) Trust-region penalty
            tr_penalty = compute_tr_penalty(alpha_modules, device=first_device, dtype=loss_ce.dtype)

            # 5) Total
            loss_total = loss_ce + args.beta * loss_kl + args.tr_lambda * tr_penalty
            loss_total.backward()
            optimizer.step()

            running += float(loss_total.item())

            if step % args.log_every == 0:
                smean, smax, smin = compute_sigma_stats(alpha_modules)
                print(
                    "{"
                    f"step: {step}, "
                    f"loss: {running/args.log_every:.4f}, "
                    f"lambda_mean: {smean:.4f}, lambda_max: {smax:.4f}, lambda_min: {smin:.4f}, "
                    f"loss_kl: {(args.beta * float(loss_kl.item())):.4f}, "
                    f"loss_ce: {float(loss_ce.item()):.4f}, "
                    f"tr_penalty: {float(tr_penalty.item()):.4f}"
                    "}"
                )
                running = 0.0

            if step >= args.steps:
                break

    print("Training finished.")

    # Quick generation sanity check
    gen_device = next(iter(base_model.parameters())).device
    inputs = tokenizer(args.prompt, return_tensors="pt").to(gen_device)

    with torch.no_grad():
        gen = base_model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    print("Generation of resulting model:", tokenizer.decode(gen[0], skip_special_tokens=True))

    # Save
    save_dir = args.save_dir.strip()
    if not save_dir:
        save_dir = default_save_dir(args.save_prefix, args.lr, args.steps, args.max_len)

    bake_and_save(base_model, tokenizer, save_dir)
    print(f"Saved baked checkpoint to {save_dir}")


if __name__ == "__main__":
    main()
