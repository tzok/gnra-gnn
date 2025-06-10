import pandas as pd
from itertools import combinations

# Load the geometric features dataset
df = pd.read_csv("geometric_features.csv")

print(f"Original dataset shape: {df.shape}")

# Generate column names to drop (raw angle values)
columns_to_drop = []

# Drop planar angle columns a{i}{j}{k}
for i, j, k in combinations(range(8), 3):
    columns_to_drop.append(f"a{i}{j}{k}")

# Drop torsion angle columns t{i}{j}{k}{l}
for i, j, k, l in combinations(range(8), 4):
    columns_to_drop.append(f"t{i}{j}{k}{l}")

# Drop the columns
df = df.drop(columns=columns_to_drop)

print(f"Dropped {len(columns_to_drop)} angle columns")
print(f"New dataset shape: {df.shape}")
print(f"Remaining columns: {list(df.columns)}")
print(f"GNRA distribution:")
print(df["gnra"].value_counts())
