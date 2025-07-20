import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

# Load model and data:
model = joblib.load('cancer_model.pkl')
data = load_breast_cancer()
feature_names = data.feature_names
target_names = data.target_names

st.title("🩺 Breast Cancer Prediction App")
st.write("Predict whether a tumor is **malignant or benign** using key features.")

# Sidebar input
st.sidebar.header("Input Features")
def user_input():
    inputs = {}
    for feature in feature_names[:30]:  
        min_val = float(np.min(data.data[:, list(feature_names).index(feature)]))
        max_val = float(np.max(data.data[:, list(feature_names).index(feature)]))
        default_val = float(np.mean(data.data[:, list(feature_names).index(feature)]))
        inputs[feature] = st.sidebar.slider(feature, min_val, max_val, default_val)
    return pd.DataFrame([inputs])

input_df = user_input()

# Prediction
prediction = model.predict(input_df)[0]
probs = model.predict_proba(input_df)[0]

st.subheader("User Input")
st.write(input_df)

st.subheader("Prediction Result")
st.write(f"**Prediction:** {target_names[prediction].capitalize()}")
st.write(f"**Probability:** Benign: {probs[1]:.2f}, Malignant: {probs[0]:.2f}")

# Feature importances
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

st.subheader("Top 10 Feature Importances")
st.bar_chart(importances.head(10))
