# model.py

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_squared_log_error,
    explained_variance_score
)

# Load and train the model
def train_model():
    # Load data
    df = pd.read_csv("synthetic_cibil_scores.csv")
    print(df.info())

    # Define features and target
    X = df[['Payment_History', 'Credit_Utilization', 'Credit_Age',
            'Number_of_Accounts', 'Hard_Inquiries', 'Debt_to_Income_Ratio']]
    y = df['CIBIL_Score']

    # Check for missing values
    print("Missing values:\n", df.isnull().sum())

    # Handle missing values
    X.fillna(X.mean(), inplace=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the XGBoost Regressor
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    msle = mean_squared_log_error(y_test, y_pred)
    explained_variance = explained_variance_score(y_test, y_pred)

    print(f"R² Score: {r2}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")
    print(f"Mean Squared Logarithmic Error (MSLE): {msle:.4f}")
    print(f"Explained Variance Score: {explained_variance:.4f}")

    # Feature importance
    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=X.columns,
        columns=['Importance']
    ).sort_values(by='Importance', ascending=False)

    print("\nFeature Importances:")
    print(feature_importances)

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
