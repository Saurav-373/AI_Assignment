import re
import pandas as pd

DATA_PATH = "../data/UpdatedResumeDataSet.csv"
OUT_PATH = "../data/resumes_clean.csv"

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()

    # Remove URLs and emails only
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep letters, numbers, and useful symbols (C++, C#, .NET)
    text = re.sub(r"[^a-z0-9\+\.\#\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    df = pd.read_csv(DATA_PATH)

    required_cols = {"Category", "Resume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {required_cols}. Found: {set(df.columns)}")

    print("Raw rows:", len(df))

    # Drop missing rows only
    df = df.dropna(subset=["Category", "Resume"]).copy()
    print("After dropna:", len(df))

    # Clean
    df["Resume_clean"] = df["Resume"].apply(clean_text)

    # Remove only very tiny/empty rows
    before_len = len(df)
    df = df[df["Resume_clean"].str.len() >= 20].copy()
    print("After len>=20:", len(df), "(removed", before_len - len(df), ")")

    # Save
    df[["Category", "Resume_clean"]].to_csv(OUT_PATH, index=False)

    print("\n✅ Preprocessing complete!")
    print("Saved:", OUT_PATH)
    print("Samples after cleaning:", len(df))
    print("\nTop categories:")
    print(df["Category"].value_counts().head(10))

if __name__ == "__main__":
    main()