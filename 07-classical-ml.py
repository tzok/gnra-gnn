import pandas as pd

# Load the geometric features dataset
df = pd.read_csv("geometric_features.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"GNRA distribution:")
print(df["gnra"].value_counts())
