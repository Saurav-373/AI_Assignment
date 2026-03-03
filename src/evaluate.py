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
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

DATA_PATH = "../data/resumes_clean.csv"
MODEL_PATH = "../models/model.pkl"
VEC_PATH = "../models/vectorizer.pkl"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "Resume_clean" not in df.columns or "Category" not in df.columns:
        raise ValueError("resumes_clean.csv must contain columns: Category, Resume_clean")

    X = df["Resume_clean"].astype(str)
    y = df["Category"].astype(str)

    # Use the same split style as training (random_state fixed for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    # Metrics
    acc = accuracy_score(y_test, y_pred)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("=== Model Evaluation (Classification) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro):   {prec_macro:.4f}")
    print(f"Recall (macro):      {rec_macro:.4f}")
    print(f"F1-score (macro):    {f1_macro:.4f}")
    print(f"Precision (weighted):{prec_weighted:.4f}")
    print(f"Recall (weighted):   {rec_weighted:.4f}")
    print(f"F1-score (weighted): {f1_weighted:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    os.makedirs("../outputs", exist_ok=True)
    fig_path = "../outputs/confusion_matrix.png"

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

    # RMSE (not ideal for multi-class classification, but included as requested)
    # We encode class labels as integers then compute RMSE on predictions.
    # NOTE: This RMSE does NOT have a meaningful ordinal interpretation for classes.
    le = LabelEncoder()
    y_test_enc = le.fit_transform(y_test)
    y_pred_enc = le.transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_enc, y_pred_enc))
    print(f"\nRMSE (label-encoded, for reference only): {rmse:.4f}")
    print("Note: RMSE is mainly for regression; for classification rely on Accuracy/F1/Precision/Recall.\n")

    # -------------------- Metrics Bar Chart --------------------
    metrics = {
        "Accuracy": acc,
        "Precision (Macro)": prec_macro,
        "Recall (Macro)": rec_macro,
        "F1-Score (Macro)": f1_macro
    }

    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values())
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()

    bar_chart_path = "../outputs/performance_metrics.png"
    plt.savefig(bar_chart_path, dpi=200)
    plt.close()

    print(f"Saved performance metrics bar chart to: {bar_chart_path}")
if __name__ == "__main__":
    main()


