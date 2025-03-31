from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    ""
)

from datasets import load_dataset

train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")

print(train_dataset)
print("--chosen--")
print(train_dataset[4]["chosen"])
print("--rejected--")
print(train_dataset[4]["rejected"])

def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_j = tokenizer(chosen, truncation=True)
        tokenized_k = tokenizer(rejected, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    return new_examples


train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= 512
    and len(x["input_ids_rejected"]) <= 512
)

from transformers import AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained(
    "",
    device_map={"": 0},
    trust_remote_code=True,
    num_labels=1,
)
model.config.use_cache = False


from transformers import TrainingArguments
from peft import LoraConfig
from trl import RewardTrainer

training_args = TrainingArguments(
    output_dir="./train_logs",
    max_steps=300,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=1.41e-5,
    optim="adamw_torch",
    save_steps=50,
    logging_steps=50,
    report_to="tensorboard",
    remove_unused_columns=False,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    bias="none",
    task_type="SEQ_CLS",
    modules_to_save=["scores"]
)

trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    max_length=512,
)

trainer.train()
trainer.model.save_pretrained("./reward_model")
