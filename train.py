from transformers import (AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, BitsAndBytesConfig)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# Paths
dataset_path = "api_training_data.jsonl"  # JSONL dataset
model_path = "tinyllama"   # Path to downloaded model

# Load dataset and split into training & evaluation
dataset = load_dataset("json", data_files=dataset_path)["train"]
dataset = dataset.train_test_split(test_size=0.1)  # 90% train, 10% eval


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Tokenize dataset properly
def tokenize_function(examples):
    return tokenizer(
        [p + "\n" + r for p, r in zip(examples["prompt"], examples["response"])],
        truncation=True,
        padding="max_length"
    )

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Tokenize train and eval datasets separately
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response"])

# Convert dataset to PyTorch format
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Enable 4-bit quantization for low VRAM usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",

)

# Load TinyLlama model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA for low-VRAM training
lora_config = LoraConfig(
    r=4, lora_alpha=8, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training arguments optimized for low VRAM
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Smallest batch size to fit in VRAM
    gradient_accumulation_steps=32,  # Higher accumulation to compensate
    eval_strategy="no",  # Disable evaluation to save VRAM
    save_strategy="epoch",
    learning_rate=2e-4,
    logging_dir="./logs",
    num_train_epochs=3,
    fp16=False,  # Enable mixed precision (saves VRAM)
    bf16=True,  # Enable mixed precision (saves VRAM)
    optim="paged_adamw_32bit",  # Reduce memory usage
    output_dir='./finetuned_navi',
    remove_unused_columns=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Start training
trainer.train()

# Save fine-tuned model
model.save_pretrained("./finetuned_navi")
tokenizer.save_pretrained("./finetuned_navi")

print("✅ Fine-tuning complete! Model saved to ./finetuned_navi")
