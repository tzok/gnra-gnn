#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import combinations
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras

# Load the geometric features dataset
df = pd.read_csv("geometric_features.csv")

print(f"Original dataset shape: {df.shape}")

# Load clusters.json to get representative structures
with open("clusters.json", "r") as f:
    clusters_data = json.load(f)

# Extract representative filenames and remove .cif extension
representative_files = set()
for cluster in clusters_data["clusters"]:
    rep_file = cluster["representative"]
    if rep_file.endswith(".cif"):
        rep_file = rep_file[:-4]  # Remove .cif extension
    representative_files.add(rep_file)

print(f"Found {len(representative_files)} representative structures")

# Filter positive dataset to only contain representatives
# Only filter positive samples (gnra == 1), keep all negative samples
positive_mask = df["gnra"] == 1
negative_mask = df["gnra"] == 0

# For positive samples, keep only representatives
positive_df = df[positive_mask]
representative_mask = positive_df["source_file"].isin(representative_files)
filtered_positive_df = positive_df[representative_mask]

# Keep all negative samples
negative_df = df[negative_mask]

# Combine filtered positive and all negative samples
df = pd.concat([filtered_positive_df, negative_df], ignore_index=True)

print(f"After filtering to representatives:")
print(f"  Positive samples: {len(filtered_positive_df)} (was {len(positive_df)})")
print(f"  Negative samples: {len(negative_df)}")
print(f"  Total samples: {len(df)}")

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
print("GNRA distribution:")
print(df["gnra"].value_counts())

# Prepare features and target
X = df.drop(columns=["source_file", "gnra"])
y = df["gnra"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Standardize features for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize classifiers
classifiers = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "SVM": SVC(random_state=42),
}


# Create neural network model
def create_neural_network(input_dim):
    model = keras.Sequential(
        [
            keras.layers.Dense(50, activation="relu", input_shape=(input_dim,)),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Train and evaluate each classifier
results = {}

for name, classifier in classifiers.items():
    print(f"\n{'=' * 50}")
    print(f"Training {name}...")

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Find misclassified instances
    misclassified_mask = y_test != y_pred
    misclassified_indices = y_test[misclassified_mask].index

    if len(misclassified_indices) > 0:
        print(f"\nMisclassified instances ({len(misclassified_indices)} total):")
        for idx in misclassified_indices:
            true_label = y_test.loc[idx]
            pred_label = y_pred[y_test.index.get_loc(idx)]
            source_file = df.loc[idx, "source_file"]
            print(
                f"  Index {idx}: {source_file} - Ground truth: {true_label}, Predicted: {pred_label}"
            )
    else:
        print("\nNo misclassified instances!")

# Train and evaluate neural network
print(f"\n{'=' * 50}")
print("Training Neural Network...")

# Create and train the neural network
nn_model = create_neural_network(X_train_scaled.shape[1])

# Set up checkpoint callback
checkpoint = keras.callbacks.ModelCheckpoint(
    "best_model_gnra.keras", save_best_only=True, monitor="loss", mode="min", verbose=0
)

# Train the model
history = nn_model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint],
    verbose=0,
)

# Make predictions
y_pred_proba = nn_model.predict(X_test_scaled, verbose=0)
y_pred_nn = (y_pred_proba > 0.5).astype(int).flatten()

# Calculate metrics
nn_accuracy = accuracy_score(y_test, y_pred_nn)
nn_precision = precision_score(y_test, y_pred_nn)
nn_recall = recall_score(y_test, y_pred_nn)
nn_f1 = f1_score(y_test, y_pred_nn)
results["Neural Network"] = {
    "accuracy": nn_accuracy,
    "precision": nn_precision,
    "recall": nn_recall,
    "f1": nn_f1,
}

print(f"Accuracy: {nn_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nn))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nn))

# Find misclassified instances for neural network
misclassified_mask_nn = y_test != y_pred_nn
misclassified_indices_nn = y_test[misclassified_mask_nn].index

if len(misclassified_indices_nn) > 0:
    print(f"\nMisclassified instances ({len(misclassified_indices_nn)} total):")
    for idx in misclassified_indices_nn:
        true_label = y_test.loc[idx]
        pred_label = y_pred_nn[y_test.index.get_loc(idx)]
        source_file = df.loc[idx, "source_file"]
        print(
            f"  Index {idx}: {source_file} - Ground truth: {true_label}, Predicted: {pred_label}"
        )
else:
    print("\nNo misclassified instances!")

# Summary of results
print(f"\n{'=' * 50}")
print("SUMMARY OF RESULTS")
print(f"{'=' * 50}")
print(
    f"{'Classifier':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}"
)
print("-" * 70)
for name, metrics in sorted(
    results.items(), key=lambda x: x[1]["accuracy"], reverse=True
):
    print(
        f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}"
    )
