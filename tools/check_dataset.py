import pandas as pd

df = pd.read_csv("../data/resumes_clean.csv")

print("Rows:", len(df))
print("Columns:", df.columns.tolist())

print("\nUnique categories:", df["Category"].nunique())
print("\nTop 20 category counts:")
print(df["Category"].value_counts().head(20))

print("\nExamples of Category values:")
print(df["Category"].dropna().unique()[:20])