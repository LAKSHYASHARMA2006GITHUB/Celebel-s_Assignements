import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
import plotly.express as px

# Page Configuration

st.set_page_config(
    page_title="Wine Cultivar Predictor",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Model and Data Loading 
@st.cache_data
def load_data_and_train_model():
    # Load data directly as a pandas DataFrame for easier handling
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, wine.target_names, X

# Load the pre-trained model and associated data.
model, target_names, wine_features_df = load_data_and_train_model()


# Main title 
st.title("🍷 Wine Cultivar Predictor 🍷")
st.write(
    "This app uses a machine learning model to predict the cultivar of a wine "
    "based on its chemical analysis. Adjust the sliders in the sidebar to see the prediction change."
)

# Sidebar 
st.sidebar.header("Input Features")
st.sidebar.write("Use the sliders to provide the wine's chemical properties.")

input_dict = {}

for feature_name in wine_features_df.columns:
    min_val = float(wine_features_df[feature_name].min())
    max_val = float(wine_features_df[feature_name].max())
    mean_val = float(wine_features_df[feature_name].mean())

    slider_label = feature_name.replace('_', ' ').capitalize()

    input_dict[feature_name] = st.sidebar.slider(
        label=slider_label,
        min_value=min_val,
        max_value=max_val,
        value=mean_val,
        step=0.1
    )


# Create a DataFrame from the user input
input_data = pd.DataFrame([input_dict])

# Get the model's prediction and the probabilities 
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Get the string name of the predicted cultivar
predicted_cultivar = target_names[prediction[0]].replace("class", "Cultivar").replace("_", " ")

st.divider()

st.header("Model Prediction")
col1, col2 = st.columns([1, 3])

with col1:
    if predicted_cultivar == 'Cultivar 0':
        st.markdown('<p style="font-size: 80px; text-align: center;">🍇</p>', unsafe_allow_html=True)
    elif predicted_cultivar == 'Cultivar 1':
        st.markdown('<p style="font-size: 80px; text-align: center;">🍷</p>', unsafe_allow_html=True)
    else:  
        st.markdown('<p style="font-size: 80px; text-align: center;">🍾</p>', unsafe_allow_html=True)

with col2:
    st.metric(label="Predicted Cultivar", value=predicted_cultivar.capitalize())
    st.write("The model is most confident that the wine belongs to this cultivar.")


#Visualization of Confidence 
st.header("Prediction Confidence")
st.write("This chart shows the model's confidence score (probability) for each possible cultivar.")

prob_df = pd.DataFrame({
    'Cultivar': [name.replace("class", "Cultivar").replace("_", " ").capitalize() for name in target_names],
    'Probability': prediction_proba[0]
})

# Create and display an bar chart.
fig = px.bar(
    prob_df,
    x='Cultivar',
    y='Probability',
    color='Cultivar',
    color_discrete_map={
        'Cultivar 0': '#8A2BE2', 
        'Cultivar 1': '#DC143C', 
        'Cultivar 2': '#2E8B57'  
    },
    labels={'Probability': 'Confidence Score'},
    height=400
)


fig.update_layout(
    yaxis_range=[0, 1],
    showlegend=False,
    xaxis_title="Wine Cultivar",
    yaxis_title="Probability",
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Show Current Input Features"):
    st.write("The prediction above is based on these input values:")
    input_df_display = input_data.T.rename(columns={0: 'Input Value'})
    st.dataframe(input_df_display.style.format("{:.1f}"))
