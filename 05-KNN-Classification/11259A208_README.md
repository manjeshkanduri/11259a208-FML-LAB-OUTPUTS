# 5. KNN Classification

[Back to experiments](../README.md)

[Python source](knn_classification.py) | [Dataset](data.csv) | [Text output](output.txt)

## Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load dataset
data = pd.read_csv("data.csv")
# Input features
X = data[['Hours_Studied', 'Attendance','Assignments']]
# Target variable
y = data['Result']
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)
# Train model
model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)
# Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Output

```text
Accuracy: 1.0

Confusion Matrix:
[[2 0]
 [0 2]]

Classification Report:
              precision    recall  f1-score   support

        Fail       1.00      1.00      1.00         2
        Pass       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```

## Run

From this experiment folder, run: `python knn_classification.py`.

