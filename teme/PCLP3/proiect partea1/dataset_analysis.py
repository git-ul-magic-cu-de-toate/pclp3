import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# a) Number and percentage of missing values for each column
missing_train = train_df.isnull().sum()
missing_percent_train = (missing_train / len(train_df)) * 100
missing_test = test_df.isnull().sum()
missing_percent_test = (missing_test / len(test_df)) * 100

# Write these values in 'missing_values_data.txt'
f = open("missing_values_data.txt", "w")
f.write("Missing Values data:\n\n")
f.write("Train data:\n")
f.write(f"{str(missing_train)}\n\n")
f.write(f"{str(missing_percent_train)}\n\n")
f.write("Test data:\n")
f.write(f"{str(missing_test)}\n\n")
f.write(f"{str(missing_percent_test)}\n\n")
f.close()

# b) Descriptive statistics
# Write these values in 'descriptive_statistics.txt'
f = open("descriptive_statistics.txt", "w")
f.write("Descriptive statistics:\n\n")
f.write("Train statistics:\n")
f.write(f"{train_df.describe(include=int)}\n\n")
f.write(f"{train_df.describe(include=float)}\n\n")
f.write(f"{train_df.describe(include=object)}\n\n")
f.write("Test statistics:\n")
f.write(f"{test_df.describe(include=int)}\n\n")
f.write(f"{test_df.describe(include=float)}\n\n")
f.write(f"{test_df.describe(include=object)}\n\n")

# c) Variable distribution analysis
# train dataset
# for numerical (histogram)
train_df_hist = train_df.drop(columns = ['loan_id', 'education', 'self_employed', 'loan_status'])

fig, axes = plt.subplots(4, 3)
axes = axes.flatten()

for i, col in enumerate(train_df_hist.columns):
  axes[i].hist(train_df[col])
  axes[i].set_title(f'Histogram: {col}')

# plt.show()
plt.close(fig)

# for categorical (countplot)
train_df['education'].value_counts().plot.bar()
plt.title('education')
# plt.show()
train_df['self_employed'].value_counts().plot.bar()
plt.title('self_employed')
# plt.show()
train_df['loan_status'].value_counts().plot.bar()
plt.title('loan_status')
# plt.show()

# test dataset
# for numerical (histogram)
test_df_hist = test_df.drop(columns = ['loan_id', 'education', 'self_employed', 'loan_status'])

fig, axes = plt.subplots(4, 3)
axes = axes.flatten()

for i, col in enumerate(test_df_hist.columns):
	axes[i].hist(test_df[col])
	axes[i].set_title(f'Histogram: {col}')

# plt.show()
plt.close(fig)

# for categorical (countplot)
test_df['education'].value_counts().plot.bar()
plt.title('education')
# plt.show()
test_df['self_employed'].value_counts().plot.bar()
plt.title('self_employed')
# plt.show()
test_df['loan_status'].value_counts().plot.bar()
plt.title('loan_status')
# plt.show()
plt.close()

# d) Outlier detection
# train dataset
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

fig, axes = plt.subplots(4, 3)
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
	axes[i].boxplot(train_df[col])
	axes[i].set_title(f'Boxplot: {col}')

# plt.show()
plt.close(fig)

# plt.show()

# test dataset
fig, axes = plt.subplots(4, 3)
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
	axes[i].boxplot(test_df[col])
	axes[i].set_title(f'Boxplot: {col}')

# plt.show()
plt.close(fig)

# e) correlation matrix(heatmap)
numeric_df = train_df.select_dtypes(include=['int64', 'float64'])

# Train
corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Heatmap for numeric variables')
# plt.show()
plt.close()

# Test
numeric_df = test_df.select_dtypes(include=['int64', 'float64'])

# Train
corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Heatmap for numeric variables')
# plt.show()
plt.close()

# f) scatter plots for analysis of relation between numerical and target columns
# Train
plt_features = ['income_annum', 'loan_amount', 'loan_ratio', 'cibil_score', 'total_asset_value']

# Generate scatterplots
for feature in plt_features:
    sns.scatterplot(x=feature, y='loan_status', data=train_df)
    plt.title(f'{feature} vs loan_status')
    plt.xlabel(feature)
    plt.ylabel('loan_status')
    # plt.show()

# Test
for feature in plt_features:
    sns.scatterplot(x=feature, y='loan_status', data=test_df)
    plt.title(f'{feature} vs loan_status')
    plt.xlabel(feature)
    plt.ylabel('loan_status')
    # plt.show()
