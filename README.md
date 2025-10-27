# 👁️ EOG Signal Classifier

## 📖 Overview

This project classifies **eye movements** (Up, Down, Right, Left, Blink) from **Electrooculography (EOG) signals** using a reproducible machine learning pipeline.  

The goal is to build a robust workflow for **human‑computer interaction** applications, where eye movements can be used as intuitive control signals.

---

## 📂 Project Structure

EOG-Signal-Classifier/

├── data/        # eog_dataset.csv (unified dataset from raw .txt files)  
├── src/         # eog_pipeline.py (9-step ML workflow)  
├── results/     # metrics.json, class_distribution.png, signal_examples.png  
├── models/      # eog_movement_model.joblib (trained ML model)  
├── api/         # FastAPI app for deployment  
└── README.md    # this file  

---

## 🛠️ Workflow

This project follows a **9-step ML workflow**:

1. Define Problem  
2. Load & Clean Data (from eog_dataset.csv)  
3. Exploratory Data Analysis (EDA) → class distribution, sample signals, PSDs  
4. Preprocessing → band-pass filtering, notch filtering, normalization  
5. Feature Engineering → time-domain, frequency-domain, wavelet, cross-channel ratios  
6. Train/Test Split (stratified or subject-wise)  
7. Model Selection → SVM, Random Forest, Logistic Regression, CNN1D  
8. Training  
9. Evaluation → accuracy, F1-score, confusion matrix, latency profiling  
9b. Improvement → hyperparameter tuning, alternative models, deep learning baselines  

---

## 📊 Dataset

- **Source**: [Kaggle EOG Dataset for Movements Classification](https://www.kaggle.com/datasets/mohamedalisalama/eog-dataset-for-movements-classification)  
- **Size**: 278 trials (each trial = 251 samples)  
- **Classes**:  
  - Yukari → Up  
  - Asagi → Down  
  - Sag → Right  
  - Sol → Left  
  - Kirp → Blink  
- **Features**: Raw EOG signals (horizontal & vertical channels)  
- **Target Variable**: Eye movement class  

---

## 🤖 Models

Baseline models implemented:

- Logistic Regression  
- Support Vector Machine (RBF kernel)  
- Random Forest Classifier  

Future extensions include **1D CNNs** for raw signal classification.

---

## 📈 Results

Evaluation metrics are stored in `results/metrics.json`. Example structure:

{  
  "SVM": {  
    "accuracy": 0.85,  
    "confusion_matrix": [[...]]  
  },  
  "RandomForest": {  
    "accuracy": 0.82,  
    "confusion_matrix": [[...]]  
  },  
  "LogReg": {  
    "accuracy": 0.78,  
    "confusion_matrix": [[...]]  
  }  
}

Generated plots:

- `results/class_distribution.png` → class balance visualization  
- `results/signal_examples.png` → sample EOG signals per class  
- `results/confusion_matrix.png` → classification performance  

---

## 🚀 Deployment

The trained model is deployed via **FastAPI**.

Run locally:

cd api  
uvicorn main:app --reload  

Endpoints:

- GET /health → health check  
- POST /predict → returns predicted movement class for a given signal  

Example request:

{  
  "signal": [0.12, 0.15, 0.09, ... 251 values ...]  
}

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
