## 📚 Overview

**LLM \_Reasoning** is a *minimal, end‑to‑end recipe* for giving large‑language‑models stronger mathematical‑reasoning skills.
It shows how to

1. **Re‑annotate raw math datasets** into Chain‑of‑Thought‑style (CoT) records (🗂 `data_process/`) ([github.com][1])
2. **Fine‑tune a base model** with supervised CoT (🗂 `train/`) ([github.com][2])
3. **Train a reward model** and **run rejection sampling** to automatically filter low‑quality traces (🗂 `reward_model/`, `reject_sampling/`) ([github.com][3])

The repo is intentionally lightweight—only Python + Shell scripts—so you can slot it into any larger alignment workflow.

---

## 🗺️ Directory structure


| Path               | Purpose                                                                                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_process`    | Convert OpenR1‑Math 220k parquet shards into Alpaca‑style JSON/Parquet with CoT answers (`data_handler.py`)[./data_process/data_handler.py]   |
| `train_sft`           | Supervised‑fine‑tuning (SFT) scripts. Example bash launcher in`train.sh` shows the core HF `Trainer` flags (fp16/bf16, long context, etc.)  |
| `reward_model`    | Reward‑model training (pairwise ranking / Bradley‑Terry‑style loss).                                                                           |
| `reject_sampling/` | Sample N answers per prompt, score with the reward model, keep the best.                                                                          |
| `*.sh`             | One‑liner helpers (`data_preprocess.sh`, `test_data_preprocess.sh`) for quick sanity tests                |
| `requirements.txt` | Fully pinned dependency list for reproducibility             |

---

## 🔧 Installation

```bash
# 1. Clone
git clone https://github.com/ShenJiarun/LLM_Reasoning.git
cd LLM_Reasoning

# 2. (Recommended) create a fresh environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

Key libraries and why they matter

* **🤗 Transformers 4.52** – long‑context support and speculation decoding hooks
* **PEFT 0.7.1** – LoRA / QLoRA for memory‑efficient SFT
* **Hydra‑Core 1.3** – clean experiment config management
* **TRL** – ready‑made reward‑model and RLHF trainers
* **vLLM** - effective inference backend

---

## 🚀 Quick start

### 1. Prepare data

Plesase first download [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/tree/472472d80032b525579a78b5fb8cf5c548ccccd4/extended) dataset from HuggingFace 🤗

```bash
bash data_preprocess.sh
# ➜  outputs reason_cot_data.parquet with columns: instruction | input | output
```

You can get your first reasoning dataset!

### 2. Supervised fine‑tune

```bash
bash train.sh
# default: 32‑bit AdamW → bf16, 1e‑5 lr, max_len 327 k tokens
```

After training is done, please check the result model in the output_dir path 🤗

### 3. Reward model + rejection sampling

After SFT finishes:

check `reject_sampling/sample.py` for further inferencing!

---

## 🧪 Evaluation

The repo does not hard‑code a benchmark to stay framework‑agnostic.
*Suggested*: run **MATH datasets exact‑answer accuracy** or **BIG‑Bench\_hard** on the filtered outputs, and compare to the base model.

---

## 🤝 Contributing

PRs are welcome! If you add new datasets or plug in DeepSpeed/FS‑DP, please:

1. Follow the existing logging pattern (Hydra + `accelerate`).
2. Update `requirements.txt` only with minimal, version‑pinned additions.
3. Document new CLI flags in code comments and this README.

---

## 📜 License

Currently no license file is present—so the default is “all rights reserved.”
Open an issue if you need explicit MIT/Apache licensing.

---

## ✨ Acknowledgements

* OpenR1‑Math 220k dataset for raw problems
* Hugging Face ecosystem (Transformers, PEFT, TRL)
* Everyone sharing ideas on structured reasoning data and reward modeling

---

<div style="text-align: center;">
  Happy reasoning! 🧠➕🤖
</div>
