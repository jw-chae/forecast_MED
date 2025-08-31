
# -*- coding: utf-8 -*-
"""
Evaluates the trained Temporal Fusion Transformer (TFT) model.

This script loads the trained model checkpoint and the unseen test data.
It then generates predictions and calculates key performance metrics, including
AUROC, AUPRC, and a classification report. Finally, it creates and saves a
calibration plot to assess model trust.
"""
import pandas as pd
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import torch

# Suppress warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.calibration import calibration_curve

# --- Configuration ---
PROCESSED_DATA_PATH = Path(__file__).parent.parent.parent / "processed_data"
MODEL_OUTPUT_PATH = Path(__file__).parent.parent.parent / "models"
REPORTS_PATH = Path(__file__).parent.parent.parent / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"

# Input files
TEST_FILE = PROCESSED_DATA_PATH / "test_timeseries.parquet"
MODEL_CHECKPOINT = MODEL_OUTPUT_PATH / "tft_model.pt"
DATASET_FILE = PROCESSED_DATA_PATH / "tft_training.tsd"


def evaluate_model():
    """
    Main function to load the model, run evaluation, and save results.
    """
    REPORTS_PATH.mkdir(exist_ok=True)
    FIGURES_PATH.mkdir(exist_ok=True)

    print("Loading test data, trained model, and training dataset...")
    test_df = pd.read_parquet(TEST_FILE)
    tft_model = TemporalFusionTransformer.load_from_checkpoint(MODEL_CHECKPOINT)
    training = TimeSeriesDataSet.load(DATASET_FILE)

    # Align columns between training and test sets
    train_cols = training.reals
    protected_cols = ['relative_time_idx', 'encoder_length']
    train_cols = [col for col in train_cols if col not in protected_cols]
    
    # Add essential columns for TimeSeriesDataSet
    essential_cols = ['time_idx', 'inpatient_id', 'outcome', 'sex', 'age']
    for col in essential_cols:
        if col not in train_cols:
            train_cols.append(col)

    test_cols = test_df.columns

    missing_in_test = list(set(train_cols) - set(test_cols))
    if missing_in_test:
        print(f"Adding missing columns to test set: {missing_in_test}")
        for col in missing_in_test:
            test_df[col] = 0
            
    test_df = test_df[train_cols]

    test_df['sex'] = test_df['sex'].astype('category').cat.codes

    print("Creating test dataloader...")
    test_dataloader = TimeSeriesDataSet.from_dataset(
        training, test_df, predict=True, stop_randomization=True
    ).to_dataloader(train=False, batch_size=256, num_workers=0)

    print("Predicting on the test set...")
    # Use the model's predict method directly for more robust prediction
    predictions = tft_model.predict(test_dataloader, mode="raw")["prediction"]
    y_pred_logits = torch.cat([p for p in predictions])
    y_pred_proba = torch.nn.functional.softmax(y_pred_logits, dim=1)[:, 1].cpu().numpy()
    
    y_true_list = [y[0] for _, y in iter(test_dataloader)]
    y_true = torch.cat(y_true_list).numpy().flatten()

    y_pred_binary = (y_pred_proba > 0.5).astype(int)

    print("\n--- Model Performance on Test Set ---")
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    
    print(f"  - Area Under the ROC Curve (AUROC): {auroc:.4f}")
    print(f"  - Area Under the PR Curve (AUPRC): {auprc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary, zero_division=0))

    print("Generating calibration plot...")
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10, strategy='uniform')
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label="TFT Model")
    plt.title("Calibration Plot")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.legend()
    plt.grid(True)
    
    output_plot_file = FIGURES_PATH / "tft_calibration_plot.png"
    plt.savefig(output_plot_file)
    print(f"  - Calibration plot saved to {output_plot_file}")
    print("--- Evaluation Complete ---")

if __name__ == "__main__":
    evaluate_model()
