import logging

from dataclasses import dataclass, field
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser
)
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    use_lora: Optional[bool] = field(default=False)
    output_dir: Optional[str] = field(default="./save/output")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
        "help": "Path to the training data."})
    source_length: int = field(default=512)
    target_length: int = field(default=512)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_deepspeed: bool = field(default=False)


def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained("model_args.model_name_or_path")
    train_dataset = load_dataset(data_args.data_path, split="train")

    training_args = DPOConfig(output_dir=ModelArguments.output_dir, logging_steps=1)
    trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    train()