# 2. Data Splitting

[Back to experiments](../README.md)

[Python source](data_splitting.py) | [Dataset](data.csv) | [Text output](output.txt)

## Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
# Load dataset
data = pd.read_csv("data.csv")
# Features (Input)
X = data[['Hours_Studied', 'Attendance', 'Assignments']]
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
Training Data:
    Hours_Studied  Attendance  Assignments
8               3          78            4
5               6          95            5
11              5          92            5
3               1          65            2
18              5          91            5
16              7          99            5
13              2          72            3
2               4          85            5
9               4          88            4
19              6          97            5
4               5          90            5
12              6          96            5
7               7          98            5
10              1          60            2
14              4          84            4
6               2          70            3

Testing Data:
    Hours_Studied  Attendance  Assignments
0               2          75            3
17              1          68            2
15              3          79            4
1               3          80            4

Training Samples: 16
Testing Samples: 4
```

## Run

From this experiment folder, run: `python data_splitting.py`.

