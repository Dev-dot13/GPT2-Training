from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load Pretrained Model and Tokenizer
model_name = "gpt2"  # Change to "gpt2-medium" or "gpt2-large" for better results
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add PAD token since GPT-2 does not have one
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

print("Model Loaded Successfully!")

# Load dataset
dataset = load_dataset("json", data_files="chatbot_dataset.json")

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Fixed Tokenization Function
def tokenize_function(example):
    inputs = tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=150, return_tensors="pt")
    targets = tokenizer(example["response"], padding="max_length", truncation=True, max_length=150, return_tensors="pt")
    
    return {
        "input_ids": inputs["input_ids"],  # Convert tensor to list
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"]
    }



# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split dataset into training and validation sets
train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Training configuration (✅ Increased batch size if GPU is used)
training_args = TrainingArguments(
    output_dir="./chatbot_model",
    per_device_train_batch_size=8 if torch.cuda.is_available() else 4,  # ✅ Increase batch size if GPU available
    per_device_eval_batch_size=8 if torch.cuda.is_available() else 4,
    num_train_epochs=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",
    save_steps=500,
    report_to="none",  # Set to "wandb" if using Weights & Biases
    fp16=True if torch.cuda.is_available() else False  # ✅ Use fp16 only if GPU is available
)

# Trainer setup (✅ Move data to GPU using torch.tensor())
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.with_format("torch"),  # ✅ Convert to PyTorch format for GPU support
    eval_dataset=eval_dataset.with_format("torch")
)

# Train model
trainer.train()

# Save model and tokenizer
model.save_pretrained("./chatbot_model")
tokenizer.save_pretrained("./chatbot_model")
print("Training Complete! Model Saved.")
