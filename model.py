import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Fix the style name to a valid matplotlib style
plt.style.use('seaborn-v0_8')  # or simply remove this line as it's optional

# Create directory for storing plots
Path("static/plots").mkdir(parents=True, exist_ok=True)

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

# Remove the train-test split and replace with cross-validation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform cross-validation with multiple metrics
cv_results = cross_validate(
    model, 
    X_scaled, 
    y, 
    cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    return_train_score=True
)

# Print cross-validation results
print("\nCross-validation results:")
print(f"Accuracy: {cv_results['test_accuracy'].mean():.3f} (+/- {cv_results['test_accuracy'].std() * 2:.3f})")
print(f"Precision: {cv_results['test_precision'].mean():.3f} (+/- {cv_results['test_precision'].std() * 2:.3f})")
print(f"Recall: {cv_results['test_recall'].mean():.3f} (+/- {cv_results['test_recall'].std() * 2:.3f})")
print(f"F1-score: {cv_results['test_f1'].mean():.3f} (+/- {cv_results['test_f1'].std() * 2:.3f})")

# Train final model on all data
model.fit(X_scaled, y)

# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('static/plots/correlation_heatmap.png')
plt.close()

# 2. Distribution Plots for Numerical Features
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()
for idx, col in enumerate(data.columns):
    sns.histplot(data=data, x=col, ax=axes[idx], kde=True)
    axes[idx].set_title(f'{col} Distribution')
plt.tight_layout()
plt.savefig('static/plots/feature_distributions.png')
plt.close()

# 3. Box Plots for Features by Churn
fig, axes = plt.subplots(3, 3, figsize=(20, 18))  # Changed to 3x3 grid
axes = axes.ravel()
numeric_cols = data.drop('Churn', axis=1).columns
for idx, col in enumerate(numeric_cols):
    sns.boxplot(data=data, x='Churn', y=col, ax=axes[idx])
    axes[idx].set_title(f'{col} by Churn Status')

# Remove empty subplots
for idx in range(len(numeric_cols), 9):  # Hide unused subplots
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('static/plots/boxplots_by_churn.png')
plt.close()

# 4. Feature Importance Plot (after model training)
def plot_feature_importance(model, feature_names):
    importance = pd.DataFrame({
        'features': feature_names,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x='importance', y='features')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('static/plots/feature_importance.png')
    plt.close()

# 5. Scatter Matrix for Top 4 Important Features
def plot_scatter_matrix(data, top_features):
    plt.figure(figsize=(12, 12))
    sns.pairplot(data[top_features + ['Churn']], hue='Churn', diag_kind='kde')
    plt.tight_layout()
    plt.savefig('static/plots/scatter_matrix.png')
    plt.close()

# After model training, create feature importance plot
plot_feature_importance(model, X.columns)

# Get top 4 important features
top_features = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).nlargest(4, 'importance')['feature'].tolist()

# Create scatter matrix for top features
plot_scatter_matrix(data, top_features)

# Save plot paths for frontend reference
plot_paths = {
    'correlation_heatmap': '/static/plots/correlation_heatmap.png',
    'feature_distributions': '/static/plots/feature_distributions.png',
    'boxplots_by_churn': '/static/plots/boxplots_by_churn.png',
    'feature_importance': '/static/plots/feature_importance.png',
    'scatter_matrix': '/static/plots/scatter_matrix.png'
}

# Save plot paths along with model and scaler
with open('model1.pkl', 'wb') as file:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'plot_paths': plot_paths
    }, file)

# Print feature names for reference
print("\nFeature names used in model:")
print(X.columns.tolist()) 