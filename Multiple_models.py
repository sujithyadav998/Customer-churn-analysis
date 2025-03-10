import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

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

# Create dictionary to store all models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'report': classification_report(y_test, y_pred)
    }
    print(f"{name} Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(results[name]['report'])

# Find the best model
best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
best_model = results[best_model_name]['model']

# Save the best model and scaler
with open('best_model.pkl', 'wb') as file:
    pickle.dump((best_model, scaler), file)

# Save all models for comparison
with open('all_models.pkl', 'wb') as file:
    pickle.dump((results, scaler), file)

print(f"\nBest performing model: {best_model_name}")
print(f"Best model accuracy: {results[best_model_name]['accuracy']:.4f}")

# Print feature importance for tree-based models
if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier, xgb.XGBClassifier)):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)