#!/usr/bin/env python3
"""
EOG-based Movement Classification Pipeline

Steps:
1. Define problem
2. Load & clean data
3. Automated EDA
4. Preprocessing
5. Feature engineering
6. Train/test split
7. Model selection
8. Training
9. Evaluation (+ improvements)
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------------------
# Paths
BASE_DIR    = Path(__file__).resolve().parent.parent
CSV_PATH    = BASE_DIR / "data" / "eog_dataset.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Step 2: Load & clean data
def load_dataset(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    # Group rows (251 timesteps) into one array per trial
    grouped = (
        df.groupby(["filename", "label", "channel", "trial_id"])["value"]
        .apply(lambda x: np.array(x.values))
        .reset_index()
        .rename(columns={"value": "signal"})
    )
    return grouped

# -------------------------------------------------------------------
# Step 3: Automated EDA
def run_eda(df):
    # Class distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x="label", data=df)
    plt.title("Class Distribution")
    plt.savefig(RESULTS_DIR / "class_distribution.png")
    plt.close()

    # Example signal plot (first trial per class)
    plt.figure(figsize=(8,5))
    for lbl in df['label'].unique():
        subset = df[df['label'] == lbl]
        if subset.empty:
            continue
        sample = subset.iloc[0]['signal']
        plt.plot(sample, label=lbl)
    plt.legend()
    plt.title("Example Signals per Class")
    plt.savefig(RESULTS_DIR / "signal_examples.png")
    plt.close()

# -------------------------------------------------------------------
# Step 4: Preprocessing (placeholder)
def preprocess_signals(df):
    # TODO: apply band-pass, notch, normalization
    # For now, just z-score normalize each signal
    df['signal_proc'] = df['signal'].apply(
        lambda x: (x - np.mean(x)) / (np.std(x)+1e-8)
    )
    return df

# -------------------------------------------------------------------
# Step 5: Feature engineering (placeholder)
def extract_features(df):
    feats, labels = [], []
    for _, row in df.iterrows():
        sig = row['signal_proc']
        # Simple features: mean, std, max-min
        f = [np.mean(sig), np.std(sig), np.max(sig)-np.min(sig)]
        feats.append(f)
        labels.append(row['label'])
    X = np.array(feats)
    y = np.array(labels)
    return X, y

# -------------------------------------------------------------------
# Step 6–8: Train/test split, model selection, training
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "LogReg": LogisticRegression(max_iter=500, random_state=42)
    }

    metrics = {}
    best_model, best_acc = None, 0.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds).tolist()
        metrics[name] = {"accuracy": float(acc), "confusion_matrix": cm}
        if acc > best_acc:
            best_acc, best_model = acc, model

    # Save best model
    joblib.dump(best_model, MODELS_DIR / "eog_movement_model.joblib")

    # Save metrics
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

# -------------------------------------------------------------------
# Step 9: Evaluation (basic)
def evaluate(metrics):
    print(json.dumps(metrics, indent=4))

# -------------------------------------------------------------------
def main():
    df = load_dataset()
    run_eda(df)
    df = preprocess_signals(df)
    X, y = extract_features(df)
    metrics = train_models(X, y)
    evaluate(metrics)

if __name__ == "__main__":
    main()