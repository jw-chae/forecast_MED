# -*- coding: utf-8 -*-
"""
Trains a baseline Logistic Regression model.

This script loads the training dataset, performs basic feature engineering,
trains a simple Logistic Regression model using scikit-learn, evaluates it,
and saves the trained model and feature list for later use.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# --- Configuration ---
PROCESSED_DATA_PATH = Path(__file__).parent.parent.parent / "processed_data"
MODEL_OUTPUT_PATH = Path(__file__).parent.parent.parent / "models"
INPUT_TRAIN_FILE = PROCESSED_DATA_PATH / "train_dataset.json"
INPUT_VALIDATION_FILE = PROCESSED_DATA_PATH / "validation_dataset.json"

MODEL_FILE = MODEL_OUTPUT_PATH / "baseline_slr_model.joblib"
PIPELINE_FILE = MODEL_OUTPUT_PATH / "baseline_slr_pipeline.joblib"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands and cleans the dataset to create a flat feature table suitable
    for a standard logistic regression model.
    """
    # Expand demographics
    df['age'] = df['demographics'].apply(lambda x: x.get('age', 0))
    df['sex'] = df['demographics'].apply(lambda x: x.get('sex', 'Unknown'))

    # Expand lab features
    # For this baseline, we'll just count the number of abnormal labs
    def count_critical_labs(lab_dict):
        if not isinstance(lab_dict, dict):
            return 0
        return sum(1 for value in lab_dict.values() if value.endswith('|1'))
    df['critical_lab_count'] = df['lab_features_packed'].apply(count_critical_labs)

    # Expand medication features
    # We'll count the number of unique ATC codes
    df['meds_atc_count'] = df['meds_atc_list'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Text features
    # For this baseline, we'll use the length of the text summaries
    df['pacs_summary_length'] = df['pacs_summary'].apply(len)
    df['emr_history_length'] = df['emr_history'].apply(len)

    # Select final features
    feature_df = df[[
        'age', 'sex', 'critical_lab_count', 'meds_atc_count',
        'pacs_summary_length', 'emr_history_length', 'outcome'
    ]].copy()

    return feature_df


def train_baseline_model():
    """
    Loads the training data, prepares features, trains, evaluates,
    and saves a logistic regression model.
    """
    # Create model output directory if it doesn't exist
    MODEL_OUTPUT_PATH.mkdir(exist_ok=True)

    print("Loading full dataset and split IDs...")
    full_df = pd.read_json(PROCESSED_DATA_PATH / "analysis_ready_data.json")
    train_ids = pd.read_json(INPUT_TRAIN_FILE).squeeze().astype(str).tolist()
    val_ids = pd.read_json(INPUT_VALIDATION_FILE).squeeze().astype(str).tolist()
    train_df = full_df[full_df['inpatient_id'].astype(str).isin(train_ids)].copy()
    validation_df = full_df[full_df['inpatient_id'].astype(str).isin(val_ids)].copy()

    # Merge outcome labels
    with open(PROCESSED_DATA_PATH / "outcomes.json", 'r') as f:
        outcomes_map = json.load(f)
    full_df['outcome'] = full_df['inpatient_id'].astype(str).map(lambda x: outcomes_map.get(x, 0))
    train_df['outcome'] = train_df['inpatient_id'].astype(str).map(lambda x: outcomes_map.get(x, 0))
    validation_df['outcome'] = validation_df['inpatient_id'].astype(str).map(lambda x: outcomes_map.get(x, 0))

    print("Preparing features for modeling...")
    train_features = prepare_features(train_df)
    validation_features = prepare_features(validation_df)

    X_train = train_features.drop('outcome', axis=1)
    y_train = train_features['outcome']
    X_val = validation_features.drop('outcome', axis=1)
    y_val = validation_features['outcome']

    # --- Define Preprocessing Pipeline ---
    # We need to scale numeric features and one-hot encode categorical features.
    numeric_features = ['age', 'critical_lab_count', 'meds_atc_count', 'pacs_summary_length', 'emr_history_length']
    categorical_features = ['sex']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # --- Define and Train Model ---
    # Create a pipeline that first preprocesses the data and then fits the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, class_weight='balanced'))
    ])

    print("\nTraining Logistic Regression model...")
    model_pipeline.fit(X_train, y_train)
    print("  - Training complete.")

    # --- Evaluate Model ---
    print("\nEvaluating model on the validation set...")
    y_pred_proba = model_pipeline.predict_proba(X_val)[:, 1]
    y_pred = model_pipeline.predict(X_val)

    auc_score = roc_auc_score(y_val, y_pred_proba)
    print(f"  - Validation ROC AUC Score: {auc_score:.4f}")
    print("  - Classification Report:")
    # Adding zero_division=0 to handle cases with no predicted samples in a class
    print(classification_report(y_val, y_pred, zero_division=0))

    # --- Save the Model and Pipeline ---
    print(f"Saving model pipeline to {PIPELINE_FILE}...")
    joblib.dump(model_pipeline, PIPELINE_FILE)
    print("  - Done.")


if __name__ == "__main__":
    train_baseline_model()
