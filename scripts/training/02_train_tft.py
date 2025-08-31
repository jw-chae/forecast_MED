# -*- coding: utf-8 -#
"""
Trains a Temporal Fusion Transformer (TFT) model using Optuna for hyperparameter tuning.

This script loads the pre-processed time-series data, defines an objective
function for Optuna to find the best hyperparameters, trains the model with
the best parameters, and saves the resulting artifact.
"""
import pandas as pd
from pathlib import Path
import warnings
import optuna
import joblib

# Suppress warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*and is already saved.*")

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy

# --- Configuration ---
PROCESSED_DATA_PATH = Path(__file__).parent.parent.parent / "processed_data"
MODEL_OUTPUT_PATH = Path(__file__).parent.parent.parent / "models"
TRAIN_FILE = PROCESSED_DATA_PATH / "train_timeseries.parquet"
VALIDATION_FILE = PROCESSED_DATA_PATH / "validation_timeseries.parquet"
MODEL_FILE = MODEL_OUTPUT_PATH / "tft_model.pt"
DATASET_FILE = PROCESSED_DATA_PATH / "tft_training.tsd"
STUDY_FILE = MODEL_OUTPUT_PATH / "tft_study.pkl"


# --- Optuna Objective Function ---
def objective(trial: optuna.Trial, train_dataloader, val_dataloader, training_dataset):
    """
    Optuna objective function to train and evaluate a model.
    """
    # Define hyperparameter search space
    params = {
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
        "attention_head_size": trial.suggest_categorical("attention_head_size", [1, 2, 4]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", [8, 16, 32, 64]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "gradient_clip_val": trial.suggest_float("gradient_clip_val", 0.01, 1.0, log=True),
    }

    # Configure trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        enable_model_summary=False,
        gradient_clip_val=params["gradient_clip_val"],
        callbacks=[early_stop_callback],
        enable_progress_bar=False,
        logger=False,
    )

    # Create model
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=params["learning_rate"],
        hidden_size=params["hidden_size"],
        attention_head_size=params["attention_head_size"],
        dropout=params["dropout"],
        hidden_continuous_size=params["hidden_continuous_size"],
        output_size=2,
        loss=CrossEntropy(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # Train the model
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Return validation loss
    return trainer.callback_metrics["val_loss"].item()


def train_with_best_params(best_params, train_dataloader, val_dataloader, training_dataset):
    """
    Trains the final model using the best hyperparameters found by Optuna.
    """
    print("\n--- Training final model with best hyperparameters ---")
    print(f"Best Parameters: {best_params}")

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min")
    trainer = pl.Trainer(
        max_epochs=100, # Train for longer with best params
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=best_params["gradient_clip_val"],
        callbacks=[early_stop_callback],
        enable_progress_bar=True,
        logger=False,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=best_params["learning_rate"],
        hidden_size=best_params["hidden_size"],
        attention_head_size=best_params["attention_head_size"],
        dropout=best_params["dropout"],
        hidden_continuous_size=best_params["hidden_continuous_size"],
        output_size=2,
        loss=CrossEntropy(),
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print(f"\nFinal training complete. Saving model to {MODEL_FILE}...")
    trainer.save_checkpoint(MODEL_FILE)


def main():
    """
    Main function to load data, run Optuna study, and train the final model.
    """
    MODEL_OUTPUT_PATH.mkdir(exist_ok=True)

    print("Loading pre-processed time-series data...")
    train_df = pd.read_parquet(TRAIN_FILE)
    val_df = pd.read_parquet(VALIDATION_FILE)

    # Align columns
    train_cols = set(train_df.columns)
    val_cols = set(val_df.columns)
    missing_in_val = list(train_cols - val_cols)
    if missing_in_val:
        for col in missing_in_val: val_df[col] = 0
    missing_in_train = list(val_cols - train_cols)
    if missing_in_train:
        for col in missing_in_train: train_df[col] = 0
    val_df = val_df[train_df.columns]

    train_df["outcome"] = train_df["outcome"].astype(int)
    val_df["outcome"] = val_df["outcome"].astype(int)
    
    all_cols = set(train_df.columns)
    exclude_cols = {'inpatient_id', 'time_idx'}
    feature_cols = sorted(list(all_cols - exclude_cols))
    static_features = ['age', 'sex']
    time_varying_features = sorted(list(set(feature_cols) - set(static_features)))
    known_reals = sorted([col for col in time_varying_features if 'med_' in col])
    observed_reals = sorted([col for col in time_varying_features if 'med_' not in col and col != 'outcome'])
    
    train_df['sex'] = train_df['sex'].astype('category').cat.codes
    val_df['sex'] = val_df['sex'].astype('category').cat.codes

    print("Creating TimeSeriesDataSet...")
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="outcome",
        group_ids=["inpatient_id"],
        max_encoder_length=71,
        max_prediction_length=1,
        static_categoricals=[], # 'sex' is now numeric
        static_reals=["age", "sex"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=observed_reals + ["outcome"],
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
        target_normalizer=None
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=128 * 2, num_workers=0)

    print("\n--- Starting Optuna Hyperparameter Study ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_dataloader, val_dataloader, training), n_trials=30)

    print(f"Study complete. Number of finished trials: {len(study.trials)}")
    print(f"Best trial value: {study.best_value}")
    print("Best trial params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Save the study object
    joblib.dump(study, STUDY_FILE)
    print(f"Optuna study saved to {STUDY_FILE}")

    # Train final model with best parameters
    train_with_best_params(study.best_params, train_dataloader, val_dataloader, training)
    
    print(f"Saving training TimeSeriesDataSet to {DATASET_FILE}...")
    training.save(DATASET_FILE)
    
    print("--- Process Complete ---")

if __name__ == "__main__":
    main()
