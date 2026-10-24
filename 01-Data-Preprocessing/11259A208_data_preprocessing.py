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
