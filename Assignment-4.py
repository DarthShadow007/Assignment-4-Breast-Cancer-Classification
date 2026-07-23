"""
AI-ML Assignment 4
Breast Cancer Classification using K-Nearest Neighbors (KNN)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ============================================================
# TASK 1: DATA UNDERSTANDING (2 Marks)
# ============================================================
print("=" * 60)
print("TASK 1: DATA UNDERSTANDING")
print("=" * 60)

# 1. Load the dataset using Pandas
df = pd.read_csv("breast_cancer.csv")

# 2. Display the first five records
print("\nFirst 5 records:")
print(df.head())

# 3. Identify numerical features and target variable
numerical_features = [col for col in df.columns if col not in ["id", "diagnosis"]]
target_variable = "diagnosis"

print(f"\nNumber of numerical features: {len(numerical_features)}")
print(f"Numerical features: {numerical_features}")
print(f"\nTarget variable: {target_variable}")

# 4. Dataset information and summary statistics
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())


# ============================================================
# TASK 2: DATA PREPROCESSING (2 Marks)
# ============================================================
print("\n" + "=" * 60)
print("TASK 2: DATA PREPROCESSING")
print("=" * 60)

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum().sum(), "total missing values")

# Remove unnecessary columns - "id" is just a row identifier
df = df.drop("id", axis=1)

# Encode the target variable (M = Malignant -> 1, B = Benign -> 0)
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])  # B=0, M=1
print(f"\nTarget encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize/standardize feature values - essential for KNN since it is
# a distance-based algorithm
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size : {X_test.shape[0]} rows")
print("\nFeatures standardized using StandardScaler (mean=0, std=1).")


# ============================================================
# TASK 3: MODEL DEVELOPMENT (3 Marks)
# ============================================================
print("\n" + "=" * 60)
print("TASK 3: MODEL DEVELOPMENT")
print("=" * 60)

# 1 & 2. Train a KNN classifier with K = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 3. Predict class labels for the test dataset
y_pred = knn.predict(X_test_scaled)

print("\nModel trained successfully with K = 5.")
print("\nFirst 10 predictions vs actual (0=Benign, 1=Malignant):")
print(pd.DataFrame({"Actual": y_test.values[:10], "Predicted": y_pred[:10]}))


# ============================================================
# TASK 4: MODEL EVALUATION (2 Marks)
# ============================================================
print("\n" + "=" * 60)
print("TASK 4: MODEL EVALUATION")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy Score : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-Score       : {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix - Breast Cancer Classification (KNN, K=5)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nConfusion matrix plot saved as 'confusion_matrix.png'")

print(f"""
Observations:
1. The model achieves a high accuracy of {accuracy:.2%}, showing that KNN
   with K=5 separates malignant and benign tumors very well once features
   are standardized.
2. Precision ({precision:.2%}) is slightly higher than recall
   ({recall:.2%}) for the malignant class, meaning the few errors the
   model makes are more often missed malignant cases (false negatives)
   than false alarms - the more clinically concerning type of error.
3. Because KNN relies purely on distance between points, its strong
   performance here depends heavily on the standardization step; without
   it, features measured on larger scales (like area) would dominate the
   distance calculation and degrade accuracy.
""")


# ============================================================
# TASK 5: CONCLUSION (1 Mark)
# ============================================================
print("=" * 60)
print("TASK 5: CONCLUSION")
print("=" * 60)

conclusion = f"""
This project built a K-Nearest Neighbors (K=5) classifier to distinguish
malignant from benign breast tumors using diagnostic measurements,
achieving an accuracy of {accuracy:.2f}, precision of {precision:.2f},
recall of {recall:.2f}, and F1-score of {f1:.2f} on the test set. Feature
scaling was essential here, since KNN classifies points based on distance
to their nearest neighbors, and unscaled features measured on very
different ranges (such as area versus smoothness) would let
larger-magnitude features dominate the distance calculation and distort
predictions. After standardizing all features, the model separated
malignant and benign cases with high accuracy, confirming that these
diagnostic measurements are strong discriminators of tumor type. A key
limitation of KNN is that it is computationally expensive at prediction
time, since it must compute distances to all training points for every
new prediction, making it slow to scale to very large datasets compared
to models that learn a fixed set of parameters upfront.
"""
print(conclusion)

with open("conclusion.txt", "w") as f:
    f.write(conclusion.strip())

print("\nAll tasks completed. Outputs saved: confusion_matrix.png, conclusion.txt")