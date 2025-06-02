# main.py
import streamlit as st
import matplotlib.pyplot as plt
import shap
from model import train_model, predict_cibil

st.title("CIBIL Score Prediction")
st.write("Enter the following values to predict your CIBIL score:")

# Load model and SHAP explainer once using Streamlit caching
model, explainer = train_model()

# Input form
ph = st.slider("Payment History (0–100)%", 0, 100, 50)
cu = st.slider("Credit Utilization (0.0–1.0)", 0.0, 1.0, 0.5)
ca = st.slider("Credit Age (years)", 0, 50, 10)
na = st.slider("Number of Accounts", 1, 100, 10)
hi = st.slider("Hard Inquiries", 0, 10, 3)
dr = st.slider("Debt-to-Income Ratio (0.0–1.0)", 0.0, 1.0, 0.3)

# Predict button
if st.button("Predict"):
    score, shap_values, suggestions = predict_cibil(model, explainer, [ph, cu, ca, na, hi, dr])

    st.write(f"📈 **Predicted CIBIL Score**: {round(score)}")

    st.write("### 🔍 Explanation of the Prediction")
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    if suggestions:
        st.write("### 💡 Tips to Improve Your Score:")
        for tip in suggestions:
            st.markdown(tip)
    else:
        st.success("Great job! All factors are contributing positively or neutrally to your score.")
