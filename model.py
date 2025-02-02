import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
data = pd.read_csv('customer_churn_dataset-training-master.csv')

# Select only relevant features
relevant_features = [
    'Age', 
    'Tenure',
    'Usage Frequency',
    'Support Calls',
    'Payment Delay',
    'Total Spend',
    'Last Interaction',
    'Churn'
]

# Keep only relevant columns
data = data[relevant_features]

# Check for missing values before preprocessing
print("Missing values before preprocessing:", data.isnull().sum())

# Handle missing values
data = data.dropna()

# Prepare features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save both the model and scaler
with open('model.pkl', 'wb') as file:
    pickle.dump((model, scaler), file)

# Print model performance
print("Model training completed and saved as model.pkl")
print("Model accuracy:", model.score(X_test_scaled, y_test))

# Print feature names for reference
print("\nFeature names used in model:")
print(X.columns.tolist()) 