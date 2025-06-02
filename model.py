# model.py

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_squared_log_error,
    explained_variance_score
)

# Cache model and explainer to avoid retraining every time
@st.cache_resource
def train_model():
    df = pd.read_csv("synthetic_cibil_scores.csv")

    # Define features and target
    X = df[['Payment_History', 'Credit_Utilization', 'Credit_Age',
            'Number_of_Accounts', 'Hard_Inquiries', 'Debt_to_Income_Ratio']]
    y = df['CIBIL_Score']

    # Handle missing values
    X.fillna(X.mean(), inplace=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the XGBoost Regressor
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Create SHAP explainer
    explainer = shap.Explainer(model)
    return model, explainer

# Predict and explain
def predict_cibil(model, explainer, values):
    cols = ["Payment_History", "Credit_Utilization", "Credit_Age",
            "Number_of_Accounts", "Hard_Inquiries", "Debt_to_Income_Ratio"]
    df_input = pd.DataFrame([dict(zip(cols, values))])
    score = model.predict(df_input)[0]
    shap_values = explainer(df_input)
    suggestions = get_suggestions(df_input, shap_values)
    return score, shap_values, suggestions

# Suggest improvements for negative SHAP values
def get_suggestions(df_input, shap_values):
    feature_names = df_input.columns.tolist()
    shap_val = shap_values.values[0]
    suggestions = []

    for i, val in enumerate(shap_val):
        if val < 0:
            feature = feature_names[i]
            tip = get_tip(feature, df_input.iloc[0][feature])
            suggestions.append(f"🔻 **{feature}** is negatively affecting your score. Tip: {tip}")

    return suggestions

def get_tip(feature, value):
    tips = {
        "Payment_History": "Aim for a consistent payment history close to 100%.",
        "Credit_Utilization": "Try to keep credit utilization below 30%.",
        "Credit_Age": "Avoid closing old accounts to improve average credit age.",
        "Number_of_Accounts": "Avoid opening too many new accounts quickly.",
        "Hard_Inquiries": "Limit the number of credit inquiries over a short period.",
        "Debt_to_Income_Ratio": "Reduce debt or increase income to lower this ratio."
    }
    return tips.get(feature, "General financial discipline can help.")
