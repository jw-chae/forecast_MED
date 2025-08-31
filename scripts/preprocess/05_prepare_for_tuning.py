

import json
import os
import random
import pandas as pd

# Define paths
BASE_DIR = '/home/joongwon00/Project_Tsinghua_Paper/med_deepseek'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
UNIFIED_FILE = os.path.join(PROCESSED_DIR, 'unified_dataset.json')
LORA_OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'lora_tuning_dataset.jsonl')
RLVR_OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'rlvr_preference_dataset.jsonl')

def load_unified_data(file_path):
    """Loads the unified JSON dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_prompt(record):
    """Creates a detailed prompt from a patient record."""
    prompt_parts = []
    if 'chief_complaint' in record and record['chief_complaint']:
        prompt_parts.append(f"The patient presents with: {record['chief_complaint']}.")
    if 'history_of_present_illness' in record and record['history_of_present_illness']:
        prompt_parts.append(f"History of present illness: {record['history_of_present_illness']}.")
    if 'past_medical_history' in record and record['past_medical_history']:
        prompt_parts.append(f"Past medical history: {record['past_medical_history']}.")
    if 'physical_examination' in record and record['physical_examination']:
        prompt_parts.append(f"Physical examination findings: {record['physical_examination']}.")
    if 'imaging_findings' in record and record['imaging_findings']:
         prompt_parts.append(f"Imaging findings: {record['imaging_findings']}.")
    if not prompt_parts:
        return None # Skip records with no useful information for a prompt
    return " ".join(prompt_parts) + " Based on this information, what is the diagnosis?"

def create_completion(record):
    """Creates a completion (the answer) from a patient record."""
    completion_parts = []
    if 'diagnosis' in record and record['diagnosis']:
        completion_parts.append(f"The diagnosis is {record['diagnosis']}.")
    if 'imaging_diagnosis' in record and record['diagnosis']:
        completion_parts.append(f"The imaging diagnosis is {record['imaging_diagnosis']}.")
    if not completion_parts:
        return None # Skip records with no useful information for a completion
    return " ".join(completion_parts)

def prepare_lora_dataset(data):
    """Prepares the dataset for LoRA supervised fine-tuning."""
    lora_records = []
    for record in data:
        prompt = create_prompt(record)
        completion = create_completion(record)
        
        if prompt and completion:
            lora_records.append({"prompt": prompt, "completion": completion})
    return lora_records

def prepare_rlvr_dataset(data):
    """Prepares a preference dataset for RLVR."""
    rlvr_records = []
    
    # Filter out records that can't be used to generate a completion
    valid_records = [r for r in data if create_completion(r) is not None]
    
    for i, record in enumerate(data):
        prompt = create_prompt(record)
        if not prompt:
            continue

        # The "chosen" response is the ground truth for the current record
        chosen_completion = create_completion(record)
        if not chosen_completion:
            continue

        # The "rejected" response is a completion from a *different* record
        # This creates a plausible but incorrect response for the given prompt
        while True:
            random_index = random.randint(0, len(valid_records) - 1)
            if random_index != i: # Ensure we don't pick the same record
                rejected_completion = create_completion(valid_records[random_index])
                if rejected_completion:
                    break
        
        rlvr_records.append({
            "prompt": prompt,
            "chosen": chosen_completion,
            "rejected": rejected_completion
        })
    return rlvr_records

def save_as_jsonl(data, file_path):
    """Saves a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def main():
    """Main function to prepare and save tuning datasets."""
    print("Loading unified data...")
    all_data = load_unified_data(UNIFIED_FILE)
    
    # --- Prepare LoRA Dataset ---
    print("Preparing dataset for LoRA (SFT)...")
    lora_data = prepare_lora_dataset(all_data)
    save_as_jsonl(lora_data, LORA_OUTPUT_FILE)
    print(f"LoRA tuning dataset saved to {LORA_OUTPUT_FILE}")
    
    # --- Prepare RLVR Dataset ---
    print("Preparing preference dataset for RLVR...")
    rlvr_data = prepare_rlvr_dataset(all_data)
    save_as_jsonl(rlvr_data, RLVR_OUTPUT_FILE)
    print(f"RLVR preference dataset saved to {RLVR_OUTPUT_FILE}")

    print("Dataset preparation for tuning is complete.")

if __name__ == "__main__":
    main()

