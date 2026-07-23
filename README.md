# Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Objective
The objective of this assignment is to develop a machine learning model using the K-Nearest Neighbors (KNN) classification algorithm to accurately predict whether a breast tumor is Malignant (M) or Benign (B) based on diagnostic measurements.

## Dataset Link
[Breast Cancer Wisconsin Diagnostic Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

## Libraries Used
* **pandas:** For data manipulation, loading, and cleaning.
* **numpy:** For numerical and array operations.
* **scikit-learn:** For data splitting, preprocessing (`StandardScaler`, `LabelEncoder`), model building (`KNeighborsClassifier`), and metric evaluation.
* **matplotlib:** For visualizing the Confusion Matrix.

## Methodology
1. **Data Understanding:** Loaded the dataset, identified the numerical features, and isolated `diagnosis` as the target variable.
2. **Data Preprocessing:** Dropped unnecessary columns like `id` and the empty `Unnamed: 32` column. The target variable was encoded (Malignant=1, Benign=0). Data was split into 80% training and 20% testing sets, and features were standardized using `StandardScaler`.
3. **Model Development:** Initialized and trained a K-Nearest Neighbors (KNN) classifier using an initial value of K=5.
4. **Model Evaluation:** Evaluated the model's predictions against the test set using Accuracy, Precision, Recall, and F1-Score, and plotted the confusion matrix.

## Results

| Metric | Value |
| :--- | :--- |
| **Accuracy** | ≈ 0.9474 |
| **Precision** | ≈ 0.9750 |
| **Recall** | ≈ 0.9070 |
| **F1-Score** | ≈ 0.9398 |

*(Note: Minor variance in results may occur based on the random state).*

![KNN Confusion Matrix](confusion_matrix.png)

## Conclusion
This project built a K-Nearest Neighbors (KNN) model (K=5) to predict breast cancer diagnoses, achieving an accuracy of approximately 0.947 on the test set. The key finding is that the KNN algorithm is highly effective for this dataset, demonstrating high precision (few false positives). 

Feature scaling (Standardization) was absolutely critical in this project. Because KNN relies on calculating the Euclidean distance between data points, unscaled features with large ranges would have dominated the distance calculations, skewing the model's predictions. 

One major limitation of the KNN algorithm is its computational expense during the testing phase. Because it does not learn a mathematical function but instead stores all the training data, predicting new instances requires calculating the distance to every single point in the training set, which scales poorly on massive datasets.