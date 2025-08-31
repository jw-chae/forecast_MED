import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

print("Importing torch...")
import torch
print("...torch imported.")

print("Importing datasets...")
from datasets import Dataset
print("...datasets imported.")

print("Importing peft...")
from peft import LoraConfig, get_peft_model
print("...peft imported.")

print("Importing transformers...")
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
print("...transformers imported.")

print("Importing trl...")
from trl import SFTTrainer
print("...trl imported.")

print("Importing llm_agent...")
from Tools.llm_agent import SYSTEM_PROMPT
print("...llm_agent imported.")


def format_prompt(example: Dict[str, Any]) -> str:
    """Formats a data point from the JSONL dataset into a prompt string."""
    observation = example.get("observation", {})
    # Recreate the user prompt structure used during data generation
    user_content = json.dumps(observation, ensure_ascii=False)
    
    # This format must match the chat template of the base model
    # For Qwen1.5-Chat, the format is "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def format_response(example: Dict[str, Any]) -> str:
    """Formats the ground truth response for the model to learn."""
    # The model should learn to output the JSON containing the proposed parameters.
    # We extract the 'action' part from the dataset, which corresponds to the LLM's previous proposal.
    action = example.get("action", {})
    # The raw LLM output during data generation also included rationale, etc.
    # Here, we focus on making the model learn the core action proposal.
    # For simplicity, we'll format it as a JSON string.
    # A more advanced approach might try to replicate the full JSON structure from `llm_raw`.
    return json.dumps({"proposed_params": action}, ensure_ascii=False)


def create_finetuning_dataset(dataset_path: str) -> Dataset:
    """Loads the JSONL file and prepares it for training."""
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    # We create a new dataset where each entry is a single string:
    # the formatted prompt followed by the formatted response.
    formatted_texts = []
    for rec in records:
        prompt = format_prompt(rec)
        response = format_response(rec)
        formatted_texts.append(prompt + response)

    return Dataset.from_dict({"text": formatted_texts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSONL dataset.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen1.5-7B-Chat", help="Base model ID from Hugging Face.")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_agent", help="Directory to save the trained LoRA adapter.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    args = parser.parse_args()

    # --- CUDA and GPU availability check ---
    if torch.cuda.is_available():
        print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("[WARN] CUDA is not available. Training will run on CPU, which will be very slow.")
    # --- End of check ---

    # 1. Load Dataset
    dataset = create_finetuning_dataset(args.dataset_path)

    # 2. Configure Quantization (for memory efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 3. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Set padding token

    # 5. Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Specific to model architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 6. Configure Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        logging_steps=10,
        fp16=True,
        save_strategy="epoch",
    )

    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=2048, # Adjust based on your context length
        tokenizer=tokenizer,
        args=training_args,
    )

    # 8. Start Training
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning completed.")

    # 9. Save the trained model
    trainer.save_model(args.output_dir)
    print(f"Trained LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
