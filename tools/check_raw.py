import pandas as pd

RAW_PATH = "../data/UpdatedResumeDataSet.csv"  # if your filename is different, update it here

df = pd.read_csv(RAW_PATH)

print("RAW rows:", len(df))
print("RAW columns:", df.columns.tolist())

# Try common column names safely
for col in ["Category", "Resume", "resume", "text", "Resume_str"]:
    if col in df.columns:
        print(f"\nUnique values in {col}:", df[col].nunique())
        if col == "Category":
            print("\nTop categories:")
            print(df["Category"].value_counts().head(10))
        break