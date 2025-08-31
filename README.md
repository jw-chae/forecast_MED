# Project: Medical AI Data Preprocessing and Fine-Tuning Preparation

This project processes and prepares a diverse set of raw medical data for the purpose of fine-tuning a large language model (Qwen3). The pipeline involves several stages of data cleaning, structuring, feature engineering, visualization, and preparation for both LoRA (SFT) and RLVR training.

## Directory Structure

```
/home/joongwon00/Project_Tsinghua_Paper/med_deepseek/
├── data/                  # Raw, original data files (.xlsx, .pdf)
├── idea/                  # Markdown files detailing the processing strategy for each data source.
├── processed_data/        # All generated data files.
│   ├── final_unified_dataset.json  # The main, cleaned, and structured dataset.
│   ├── lora_tuning_dataset.jsonl   # Dataset formatted for LoRA fine-tuning.
│   └── rlvr_preference_dataset.jsonl # Dataset formatted for RLVR.
├── scripts/
│   └── preprocess/        # All Python scripts used in the pipeline.
└── visualizations/        # All generated plots and charts.
    └── advanced_diagnosis_distribution.png # Key visualization of disease distribution.
```

## Data Processing Pipeline

The process was executed through a series of scripts:

1.  **Initial Exploration & Ideation:**
    - The raw data files in the `data/` directory were analyzed.
    - A detailed plan for handling each file was created and stored in the `idea/` directory. This involved identifying key columns, proposing feature engineering steps (like parsing dosage, creating clinical narratives), and defining a target data structure.

2.  **Intelligent Data Structuring (`07_build_final_dataset.py`):**
    - This script implements the logic from the `idea/` plans.
    - It reads the most critical raw data files (outpatient, imaging, and EMR).
    - It extracts relevant information, combines text fields into narratives, and standardizes the data into a single, consistent schema.
    - The output is `final_unified_dataset.json`, which serves as the master dataset for all subsequent steps.

3.  **Advanced Visualization (`06_advanced_visualization.py`):**
    - This script loads the final unified dataset.
    - It generates a publication-quality bar chart (`advanced_diagnosis_distribution.png`) showing the distribution of primary diagnoses.
    - **Crucially, it addresses the data imbalance issue** by grouping diagnoses that constitute less than 1% of the data into an "Other" category. This makes the visualization clean and interpretable, highlighting the most prevalent conditions.
    - All plot labels are translated to English for clarity.

4.  **Preparation for Fine-Tuning (`05_prepare_for_tuning.py` - *Note: This would be the next logical step, using the `final_unified_dataset.json`*):**
    - The previously generated script can now be adapted to use the clean, final dataset to create high-quality prompts and completions for LoRA and RLVR tuning.

## Analysis and Potential Improvements

### Data Quality and Imbalance
- **Observation:** The analysis and visualization confirmed a significant imbalance in the distribution of diagnoses. A few common conditions make up a large portion of the dataset, while many others are rare.
- **Impact:** If not handled properly, this imbalance would cause a fine-tuned model to be highly biased towards predicting only the most common diseases.
- **Mitigation:**
    - **In Visualization:** The issue was handled by grouping rare diseases, as described above.
    - **For Model Training (Improvement):** The actual model training process must incorporate specific strategies to counteract this. Good options include:
        - **Oversampling:** Using techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create more examples of the rare diseases.
        - **Undersampling:** Reducing the number of examples from the most common diseases.
        - **Weighted Loss Functions:** Modifying the model's loss function to penalize errors on rare diseases more heavily.

### Future Improvements
1.  **Incorporate All Data Sources:** The current `final_unified_dataset.json` focuses on the most diagnosis-rich text sources. A future iteration should merge in the **lab data (`LIS`)** and the **inpatient medication data**. This would require creating a patient-centric view, linking all events (diagnoses, labs, meds) to a single patient ID and timeline.
2.  **Advanced NLP Feature Extraction:** The narrative text fields (`clinical_narrative`, `findings_text`) are very rich. Advanced NLP techniques like **Named Entity Recognition (NER)** could be used to explicitly extract symptoms, medications, and anatomical locations, turning them into structured features.
3.  **Refine RLVR Data Generation:** The current method for creating `rejected` pairs (sampling from other patients) is a good start. A more advanced approach would be to use an LLM to generate *subtly incorrect* responses (e.g., a diagnosis that is similar but less likely), which would provide a more challenging and effective training signal for the model.
4.  **Build a RAG System:** The `Epidemic_guide.pdf` should be processed into a vector database to support a Retrieval-Augmented Generation pipeline, allowing the LLM to query this knowledge base before answering questions.