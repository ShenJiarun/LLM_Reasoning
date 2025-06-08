import copy
import logging
import logging
import os
import torch
import transformers
from pathlib import Path
from dataclasses import dataclass, field
from datasets import load_dataset
from functools import partial
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft.tuners.lora import LoraLayer
from typing import Dict, Optional, Sequence, List

logger = logging.getLogger(__name__)


IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "<|im_start|>user\nA conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>Put your final answer within \\\\boxed{}. {instruction}{input}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "prompt_no_input": (
        "<|im_start|>user\nA conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>Put your final answer within \\\\boxed{{}}. {instruction}<|im_end|>\n<|im_start|>assistant\n"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    use_lora: Optional[bool] = field(default=False)


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


def get_all_datapath(dir_name: str) -> List[str]:
    absolute_path = ""
    path = Path(dir_name)
    if path.is_dir():
        all_file_list = []
        
        for (root, dir, file_name) in os.walk(dir_name):
            # TODO: 使用全局路径
            for temp_file in file_name:
                standard_path = f"{root}/{temp_file}"

                all_file_list.append(standard_path)
        return all_file_list, absolute_path
    else:
        return [dir_name], absolute_path


def load_dataset_from_path(data_path: Optional[str] = None,
                           cache_dir: Optional[str] = "cache_data") -> Dataset:
    all_file_list, absolute_path = get_all_datapath(data_path)
    data_files = {'train': all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info(f"loading number of {len(all_file_list)} files from the path {absolute_path}...")

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
    )['train']
    return raw_datasets


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    ne_pad_token_id = IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(ne_pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(
        strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def make_train_dataset(tokenizer: transformers.PreTrainedTokenizer, data_path: str, data_args: DataArguments) -> Dataset:
    logging.warning("Loading data...")

    dataset = load_dataset_from_path(
        data_path=data_path,
    )
    logging.warning("Formatting inputs...")
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    def generate_sources_targets(examples: Dict, tokenizer: transformers.PreTrainedTokenizer):
        ins_data = examples['instruction']
        if 'input' not in examples.keys():
            input_data = [""] * len(ins_data)
        else:
            input_data = examples['input']
        output = examples['output']

        len_ = len(ins_data)

        sources = []
        for i in range(len_):
            if input_data[i] != "":
                current_data = prompt_input.format_map({'instruction': ins_data[i], 'input': input_data[i]})
            else:
                current_data = prompt_no_input.format_map({'instruction': ins_data[i]})
            sources.append(current_data)
        sources = [i[:data_args.source_length] for i in sources]
        targets = [
            f"{example[:data_args.target_length-1]}{tokenizer.eos_token}" for example in output]


        input_output = preprocess(
            sources=sources, targets=targets, tokenizer=tokenizer)
        examples['input_ids'] = input_output['input_ids']
        examples['labels'] = input_output['labels']
        return examples

    generate_sources_targets_p = partial(
        generate_sources_targets, tokenizer=tokenizer)

    dataset = dataset.map(
        function=generate_sources_targets_p,
        batched=True,
        desc="Running tokenizer on train dataset",
        num_proc=20
    ).shuffle()
    return dataset



def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments, data_args: DataArguments) -> tuple:

    model = AutoModelForCausalLM.from_pretrained(
        "/root/Qwen/Qwen2.5-1.5B",
        cache_dir=training_args.cache_dir,
        device_map='auto',
        torch_dtype='auto',
        # if model_args.model_name_or_path.find("falcon") != -1 else False
        trust_remote_code=True

        )

    if model_args.use_lora:

        logging.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model
        LORA_R = 32
        # LORA_ALPHA = 16
        LORA_DROPOUT = 0.05
        TARGET_MODULES = [
            "o_proj","gate_proj", "down_proj", "up_proj"
        ]

        config = LoraConfig(
            r=LORA_R,
            # lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # model = model.to(torch.bfloat16)
        model = get_peft_model(model, config)
        # peft_module_casting_to_bf16(model)
        model.print_trainable_parameters()


    # model.is_parallelizable = True
    # model.model_parallel = True
    # torch.cuda.empty_cache()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)

    return model, tokenizer


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = load_model_and_tokenizer(
        model_args, training_args, data_args)

    with training_args.main_process_first(desc="loading and tokenization"):

        train_dataset = make_train_dataset(
            tokenizer=tokenizer, data_path="reason_cot_data.parquet", data_args=data_args)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                           label_pad_token_id=IGNORE_INDEX
                                           )

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=None,
                      data_collator=data_collator)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    train()