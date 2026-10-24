# 1. Data Preprocessing

[Back to experiments](../README.md)

[Python source](data_preprocessing.py) | [Dataset](data.csv) | [Text output](output.txt)

## Code

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("data (1).csv")

# Display first 5 rows
print("Original Dataset:")
print(data.head())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values (if any)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Select numerical columns
X = data[['Hours_Studied', 'Attendance', 'Assignments', 'Marks']]

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_Scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Display normalized dataset
print("\nNormalized Dataset:")
print(X_Scaled.head())










import pandas as pd
from sklearn.model_selection import train_test_split
# Load dataset
data = pd.read_csv("data.csv")
# Features (Input)
X = data[['Hours_Studied','Attendance','Assignments']]
# Target (Output)
y = data['Result']
# Split data into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
# Display sizes
print("Training Data:")
print(X_train)
print("\nTesting Data:")
print(X_test)
print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))
```

## Output

```text
Original Dataset:
   StudentID  Hours_Studied  Attendance  Assignments  Marks Result
0          1              2          75            3     45   Fail
1          2              3          80            4     52   Pass
2          3              4          85            5     60   Pass
3          4              1          65            2     35   Fail
4          5              5          90            5     75   Pass

Missing Values:
StudentID        0
Hours_Studied    0
Attendance       0
Assignments      0
Marks            0
Result           0

Normalized Dataset:
   Hours_Studied  Attendance  Assignments     Marks
0      -0.958215   -0.697680    -0.912871 -0.965884
1      -0.440261   -0.267013     0.000000 -0.603354
2       0.077693    0.163653     0.912871 -0.189034
3      -1.476169   -1.559013    -1.825742 -1.483784
4       0.595647    0.594320     0.912871  0.587817
```

## Run

From this experiment folder, run: `python data_preprocessing.py`.

