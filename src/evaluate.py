# evaluate.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    mean_squared_error
)
from sklearn.preprocessing import LabelEncoder

# ===== Robust Paths (Never Break) =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "resumes_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

def main():
    # -------- Checks --------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset:\n{DATA_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file:\n{MODEL_PATH}\nRun train.py first.")

    if not os.path.exists(VEC_PATH):
        raise FileNotFoundError(f"Missing vectorizer file:\n{VEC_PATH}\nRun train.py first.")

    # -------- Load Data --------
    df = pd.read_csv(DATA_PATH)
    if "Resume_clean" not in df.columns or "Category" not in df.columns:
        raise ValueError("resumes_clean.csv must contain columns: Category, Resume_clean")

    X = df["Resume_clean"].astype(str)
    y = df["Category"].astype(str)

    # Same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------- Load Model + Vectorizer --------
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)

    # -------- Predict --------
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    # -------- Metrics --------
    acc = accuracy_score(y_test, y_pred)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("=== Model Evaluation (Classification) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro):    {prec_macro:.4f}")
    print(f"Recall (macro):       {rec_macro:.4f}")
    print(f"F1-score (macro):     {f1_macro:.4f}")
    print(f"Precision (weighted): {prec_weighted:.4f}")
    print(f"Recall (weighted):    {rec_weighted:.4f}")
    print(f"F1-score (weighted):  {f1_weighted:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    # -------- Outputs folder --------
    os.makedirs(OUT_DIR, exist_ok=True)

    # -------- Confusion Matrix --------
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=6)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=6)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\nSaved confusion matrix image to: {fig_path}")

    # -------- RMSE (for reference only) --------
    le = LabelEncoder()
    y_test_enc = le.fit_transform(y_test)
    y_pred_enc = le.transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_enc, y_pred_enc))
    print(f"\nRMSE (label-encoded, for reference only): {rmse:.4f}")
    print("Note: RMSE is mainly for regression; for classification rely on Accuracy/F1/Precision/Recall.\n")

    # -------- Metrics Bar Chart --------
    metrics = {
        "Accuracy": acc,
        "Precision (Macro)": prec_macro,
        "Recall (Macro)": rec_macro,
        "F1-Score (Macro)": f1_macro
    }

    bar_chart_path = os.path.join(OUT_DIR, "performance_metrics.png")
    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values())
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(bar_chart_path, dpi=200)
    plt.close()

    print(f"Saved performance metrics bar chart to: {bar_chart_path}")

if __name__ == "__main__":
    main()