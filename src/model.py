import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the medical insurance dataset
print("Loading dataset from data/insurance.csv...")
df = pd.read_csv("data/insurance.csv")

# Print the first few rows to verify the data loaded correctly
print("\n--- Dataset Sample ---")
print(df.head())

# 2. Preprocess Data: Convert text column 'smoker' into numeric markers (0 or 1)
# Machine Learning models need numbers, not text!
df['is_smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

# 3. Separate the dataset into features (X) and the label we want to predict (y)
X = df[['age', 'bmi', 'children', 'is_smoker']]
y = df['charges']

# 4. Split data into Training set (80%) and Testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining with {X_train.shape[0]} samples.")
print(f"Testing with {X_test.shape[0]} samples.")

# 5. Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training successful!")

# 6. Test the model by making predictions on the test data
predictions = model.predict(X_test)

# 7. Calculate Evaluation Metrics to check accuracy
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Model Evaluation Results ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"R-squared Score (R2): {r2:.4f}")