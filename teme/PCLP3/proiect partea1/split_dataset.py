import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('loan_approval_dataset.csv')

# Simulate deletion of 5% of values of column 'cibil_score' (Credit Score)
df.loc[df.sample(frac=0.05).index, 'cibil_score'] = np.nan

# Insert a new column for loan to annual income ratio
df.insert(6, 'loan_ratio', df['loan_amount'] / df['income_annum'])
df['loan_ratio'] = df['loan_ratio'].round(2)

# Add noise to asset columns
noise = np.random.normal(0, 0.05, size=len(df))
df['residential_assets_value'] *= (1 + noise)
df['commercial_assets_value'] *= (1 + noise)
df['luxury_assets_value'] *= (1 + noise)
df['bank_asset_value'] *= (1 + noise)

# Round out the values in the asset columns
df['residential_assets_value'] = df['residential_assets_value'].round(2)
df['commercial_assets_value'] = df['commercial_assets_value'].round(2)
df['luxury_assets_value'] = df['luxury_assets_value'].round(2)
df['bank_asset_value'] = df['bank_asset_value'].round(2)

# Insert a new column for total asset value
df.insert(13, 'total_asset_value', df['residential_assets_value'] +
		  						   df['commercial_assets_value'] +
		  						   df['luxury_assets_value'] +
		  						   df['bank_asset_value'])
df['total_asset_value'] = df['total_asset_value'].round(2)

# Split the data: 80% for training, 20% for testing
X = df.drop(['loan_status'], axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Combine X and y back for train and test data
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save to train and test csv files (and send files to colleague)
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)

# Minimal data processing
# Encode categorical values before split
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

# Fill empty values from cibil_score
train_data['cibil_score'] = train_data['cibil_score'].fillna(train_data['cibil_score'].mean())
test_data['cibil_score'] = test_data['cibil_score'].fillna(test_data['cibil_score'].mean())

# Split data
train_data_X = train_data.drop(['loan_status_Rejected'], axis=1)
train_data_y = train_data['loan_status_Rejected']
test_data_X = test_data.drop(['loan_status_Rejected'], axis=1)
test_data_y = test_data['loan_status_Rejected']

# Choose model for classification
model = LogisticRegression(max_iter=500)

# Train model
model.fit(train_data_X, train_data_y)

# Predict labels on the test dataset
y_pred = model.predict(test_data_X)

# Calculate accuracy
accuracy = accuracy_score(test_data_y, y_pred)
print(f"Accuracy of logistic regression model: {accuracy:.2f}")

# More comprehensive evaluation of model
print(classification_report(test_data_y, y_pred))
