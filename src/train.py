import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# ===== Robust Paths (Never Breaks) =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "resumes_clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load cleaned dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at:\n{DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Total resumes:", len(df))

# X = resume text
X = df["Resume_clean"]

# y = job category
y = df["Category"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Data split into training and testing")

# Convert text into numbers
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Text converted into numerical features")

# Train model
model = LogisticRegression(max_iter=1000)

print("Training model...")
model.fit(X_train_vec, y_train)

print("Model training complete")

# Test model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

print("Model saved successfully in models folder")