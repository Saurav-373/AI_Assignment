import pandas as pd

# Load dataset
df = pd.read_csv("../data/UpdatedResumeDataSet.csv")

print("\nDataset loaded successfully!\n")

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Show dataset information
print("\nDataset info:")
print(df.info())

# Show total samples
print("\nTotal number of resumes:", len(df))

# Show category distribution
print("\nCategory distribution:")
print(df['Category'].value_counts())
