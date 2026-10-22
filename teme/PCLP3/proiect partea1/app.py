import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load dataset and preprocess
df = pd.read_csv('loan_approval_dataset.csv')
df['loan_ratio'] = df['loan_amount'] / df['income_annum']
df['total_asset_value'] = (df['residential_assets_value'] +
                           df['commercial_assets_value'] +
                           df['luxury_assets_value'] +
                           df['bank_asset_value'])
df = pd.get_dummies(df, drop_first=True)
df['cibil_score'] = df['cibil_score'].fillna(df['cibil_score'].mean())

# Separate X and y
X = df.drop(['loan_status_Rejected'], axis=1)
y = df['loan_status_Rejected']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Prediction function
def predict_loan(dependents, education, self_employed,
                 income_annum, loan_amount, loan_term, cibil_score,
                 residential_assets_value, commercial_assets_value,
                 luxury_assets_value, bank_asset_value):
    loan_ratio = loan_amount / income_annum
    total_asset_value = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value

    input_data = {
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value,
        'loan_ratio': loan_ratio,
        'total_asset_value': total_asset_value,
        'no_of_dependents': int(dependents),
        'education_NotGraduate': 1 if education == 'NotGraduate' else 0,
        'self_employed_Yes': 1 if self_employed == 'Yes' else 0,
    }

    input_df = pd.DataFrame([input_data])
    for col in X.columns:
        if col not in input_df:
            input_df[col] = 0
    input_df = input_df[X.columns]

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    return f"Prediction: {'Approved' if pred == 1 else 'Rejected'} (Probability: {prob:.2f})"

# Interfață Gradio
demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Dropdown(['0', '1', '2', '3+'], label='Dependents'),
        gr.Dropdown(['Graduate', 'NotGraduate'], label='Education'),
        gr.Dropdown(['Yes', 'No'], label='Self Employed'),
        gr.Number(label='Annual Income'),
        gr.Number(label='Loan Amount'),
        gr.Number(label='Loan Term (months)'),
        gr.Number(label='CIBIL Score'),
        gr.Number(label='Residential Asset Value'),
        gr.Number(label='Commercial Asset Value'),
        gr.Number(label='Luxury Asset Value'),
        gr.Number(label='Bank Asset Value'),
    ],
    outputs="text",
    title="Loan Approval Predictor",
    description="Introduce details about the applicant to estimate if the loan will be approved."
)

demo.launch(share=True)
