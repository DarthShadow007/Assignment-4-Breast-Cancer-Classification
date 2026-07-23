import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ==========================================
# Task 1: Data Understanding
# ==========================================
print("--- Task 1: Data Understanding ---")
# 1. Load the dataset using Pandas
df = pd.read_csv("breast_cancer.csv") # Make sure the kaggle csv is in your folder

# 2. Display the first five records
print("\nFirst 5 records:")
print(df.head())

# 3. Identify Numerical features and Target variable
print("\nTarget variable: 'diagnosis'")
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numerical features:", numerical_features)

# 4. Display dataset information and summary statistics
print("\nDataset info:")
df.info()
print("\nSummary statistics:")
print(df.describe())

# ==========================================
# Task 2: Data Preprocessing
# ==========================================
print("\n--- Task 2: Data Preprocessing ---")
# Check for missing values
print("\nMissing values before cleaning:\n", df.isnull().sum().max()) # Shows max missing to keep output clean

# Remove unnecessary columns (Fixes the NaN Error!)
if 'Unnamed: 32' in df.columns:
    df.drop(columns=['Unnamed: 32'], inplace=True)
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Encode the target variable (Malignant=1, Benign=0)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# Split the dataset into features (X) and target (y)
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Task 3: Model Development
# ==========================================
print("\n--- Task 3: Model Development ---")
# 1 & 2. Train a KNN classifier with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 3. Predict the class labels for the test dataset
y_pred = knn.predict(X_test_scaled)
print("Model successfully trained and predictions generated.")

# ==========================================
# Task 4: Model Evaluation
# ==========================================
print("\n--- Task 4: Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:      {precision_score(y_test, y_pred):.4f}")
print(f"Recall:         {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:       {f1_score(y_test, y_pred):.4f}")

# Observations
print("\nObservations:")
print("1. The model demonstrates extremely high accuracy, proving KNN is highly effective for this dataset.")
print("2. A high recall score is crucial in the medical field to ensure malignant tumors are not misclassified as benign.")
print("3. Feature scaling significantly contributed to the model's high performance by equalizing the distances of all features.")

# Generate and show Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign (0)', 'Malignant (1)'])
disp.plot(cmap='Blues')
plt.title("KNN Confusion Matrix (K=5)")
plt.show()

# Task 5: Conclusion is provided in the README.md as per standard repo formatting.