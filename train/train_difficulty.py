import json
import logging

import torch
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

logger = logging.getLogger(__name__)


def read_jsonl_to_list(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                data_dict = json.loads(line)
                data_list.append(data_dict)
    logger.info(f"Successfully loaded JSON file from {file_path}...")
    return data_list

readin_data = read_jsonl_to_list("generate_output_math_difficult_Qwen3_1.7B.jsonl")

raw_ds = Dataset.from_dict(
    {"text": [d["prompt"]         for d in readin_data],
     "labels": [float(d["difficulty"]) for d in readin_data]}
)

train_ds, val_ds = raw_ds.train_test_split(test_size=0.2).values()

tok = AutoTokenizer.from_pretrained("/root/Qwen/Qwen3-0.6B-Base", add_eos_token=True)

def preprocess(batch):
    out = tok(batch["text"],
              padding=False,
              truncation=True)
    out["labels"] = batch["labels"]
    return out

train_ds = train_ds.map(preprocess, batched=True, num_proc=4, remove_columns=["text"])
val_ds   = val_ds.map(preprocess,   batched=True, num_proc=4, remove_columns=["text"])

model = AutoModelForSequenceClassification.from_pretrained("/root/Qwen/Qwen3-0.6B-Base", num_labels=1)
model.config.problem_type = "regression"

args = TrainingArguments(
    output_dir="results",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=1,
    save_steps=100000,
)

data_collator = DataCollatorWithPadding(tok, return_tensors="pt")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds  = preds.squeeze()
    mse = ((preds - labels) ** 2).mean().item()
    return {"mse": mse}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


model.eval()
problem = [""]
inputs  = tok(problem, return_tensors="pt")
with torch.no_grad():
    diff_score = model(**inputs).logits.squeeze().item()
print(f"Predicted difficulty score: {diff_score:.2f}")
