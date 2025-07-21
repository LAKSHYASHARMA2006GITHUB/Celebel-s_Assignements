import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Config
st.set_page_config(page_title=" Smart Data Visualizer", layout="wide")
UPLOAD_FOLDER = "uploaded_files"
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

st.markdown("<h1 style='text-align: center; color: teal;'> Persistent Data Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Upload files, explore data, visualize insights</h4>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "txt", "data"])
if uploaded_file:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved as: {uploaded_file.name}")
    selected_file = uploaded_file.name
    selected_path = os.path.join(UPLOAD_FOLDER, selected_file)
    try:
        df = pd.read_csv(selected_path, header=None, delim_whitespace=True)
    except:
        df = pd.read_csv(selected_path)
    st.session_state.uploaded_df = df
    st.session_state.selected_file = uploaded_file.name

st.sidebar.header("Select Saved File")
saved_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith((".csv", ".txt", ".data"))]

if saved_files:
    selected_file = st.sidebar.selectbox("Choose  file to explore", saved_files, index=saved_files.index(st.session_state.get("selected_file", saved_files[0])) if saved_files else 0)
    selected_path = os.path.join(UPLOAD_FOLDER, selected_file)

    

    try:
        df = pd.read_csv(selected_path, header=None, delim_whitespace=True)
    except:
        df = pd.read_csv(selected_path)


    # Optional Delete Button
    with st.sidebar.expander("File Options"):
        if st.button("Delete Selected File"):
            os.remove(selected_path)
            st.sidebar.success(f"Deleted file: {selected_file}")
            st.rerun()


    try:
        df = pd.read_csv(selected_path, header=None, delim_whitespace=True)
    except:
        df = pd.read_csv(selected_path)

    st.sidebar.success(f"Loaded: {selected_file}")

    # Add column names if not present and known to be German dataset
    if df.shape[1] == 21 and df.iloc[:, -1].isin([1, 2]).all():
        df.columns = [
            "checking_account_status", "duration", "credit_history", "purpose", "credit_amount",
            "savings_account", "employment_since", "installment_rate", "personal_status_sex",
            "other_debtors", "residence_since", "property", "age", "other_installment_plans",
            "housing", "existing_credits", "job", "num_dependents", "telephone", "foreign_worker",
            "target"
        ]

        """ Decode categorical columns for German dataset as given in the user guide for the model
         in the word file. """
        decode_dict = {
            "checking_account_status": {
                'A11': '< 0 DM', 'A12': '0-200 DM', 'A13': '>= 200 DM / salary', 'A14': 'no account'
            },
            "credit_history": {
                'A30': 'no credits/all paid', 'A31': 'all paid at bank', 'A32': 'paid till now',
                'A33': 'delayed', 'A34': 'critical account'
            },
            "purpose": {
                'A40': 'car (new)', 'A41': 'car (used)', 'A42': 'furniture', 'A43': 'radio/TV',
                'A44': 'appliances', 'A45': 'repairs', 'A46': 'education', 'A48': 'retraining',
                'A49': 'business', 'A410': 'others'
            },
            "savings_account": {
                'A61': '< 100 DM', 'A62': '100-500 DM', 'A63': '500-1000 DM', 'A64': '>= 1000 DM', 'A65': 'unknown'
            },
            "employment_since": {
                'A71': 'unemployed', 'A72': '< 1 yr', 'A73': '1-4 yrs', 'A74': '4-7 yrs', 'A75': '>= 7 yrs'
            },
            "personal_status_sex": {
                'A91': 'male div/sep', 'A92': 'female div/married', 'A93': 'male single',
                'A94': 'male mar/wid', 'A95': 'female single'
            },
            "other_debtors": {
                'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'
            },
            "property": {
                'A121': 'real estate', 'A122': 'insurance/savings', 'A123': 'car/other', 'A124': 'unknown'
            },
            "other_installment_plans": {
                'A141': 'bank', 'A142': 'stores', 'A143': 'none'
            },
            "housing": {
                'A151': 'rent', 'A152': 'own', 'A153': 'free'
            },
            "job": {
                'A171': 'unskilled non-resident', 'A172': 'unskilled', 'A173': 'skilled', 'A174': 'high qual/self-employed'
            },
            "telephone": {
                'A191': 'none', 'A192': 'yes'
            },
            "foreign_worker": {
                'A201': 'yes', 'A202': 'no'
            }
        }

        for col, mapping in decode_dict.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

    st.sidebar.markdown("### Show Data Options:")
    show_data = st.sidebar.checkbox("Raw Data Table", True)
    show_summary = st.sidebar.checkbox("Summary Statistics", True)
    show_visuals = st.sidebar.checkbox("Auto Visualizations", True)
    show_corr_explorer = st.sidebar.checkbox("Correlation Explorer", False)
    show_model = st.sidebar.checkbox("Train Logistic Model", False)

    if show_data:
        st.subheader("Raw Data Table")
        columns = st.multiselect("Select columns to display", df.columns, default=list(df.columns))
        st.dataframe(df[columns].head(100))

    if show_summary:
        st.subheader("Summary Statistics")
        st.write(df.describe(include='all'))

        st.markdown("### Missing Values")
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        st.dataframe(pd.DataFrame({"Missing Values": missing, "% Missing": missing_percent}))

        st.markdown("### Data Types")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

    if show_visuals:
        st.subheader("Auto Visualizations")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

        if numeric_cols:
            st.markdown("### Numeric Distributions")
            for col in numeric_cols:
                fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
                st.plotly_chart(fig)

        if categorical_cols:
            st.markdown("### Categorical Distributions")
            for col in categorical_cols:
                if df[col].nunique() < 20:
                    fig = px.histogram(df, y=col, title=f"Counts of {col}")
                    st.plotly_chart(fig)

        if len(numeric_cols) >= 2:
            st.markdown("### Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

    if show_corr_explorer:
        st.subheader("Correlation Explorer")
        numeric_df = df.select_dtypes(include='number')
        corr = numeric_df.corr()
        corr_thresh = st.slider("Minimum absolute correlation", 0.0, 1.0, 0.5, 0.05)
        mask = (abs(corr) >= corr_thresh) & (abs(corr) < 1.0)
        filtered = corr[mask].dropna(how='all').dropna(axis=1, how='all')
        st.dataframe(filtered)

    if show_model and 'target' in df.columns:
        st.subheader("Logistic Regression Model")
        df_clean = df.dropna()
        if df_clean.empty:
            st.warning("Cannot train model: cleaned dataset is empty after dropping missing values.")
        else:
            df_encoded = pd.get_dummies(df_clean.drop('target', axis=1), drop_first=True)
            if df_encoded.empty:
                st.warning("Cannot train model: no usable features after encoding.")
            else:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(df_encoded, df_clean['target'], test_size=0.3, random_state=42)

                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.text("Classification Report")
                    st.text(classification_report(y_test, y_pred))

                    cm = confusion_matrix(y_test, y_pred)
                    st.text("Confusion Matrix")
                    st.write(cm)

                    if set(df['target'].unique()) == {1, 2}:
                        cost_matrix = np.array([[0, 1], [5, 0]])
                        cost_score = np.sum(cm * cost_matrix)
                        st.success(f"Cost-sensitive error score: {cost_score}")
                except ValueError as e:
                    st.error(f"Model training failed: {e}")

    st.download_button("Download Processed Data", df.to_csv(index=False), file_name="cleaned_data.csv")
else:
    st.info("Upload a dataset to begin. Supported: CSV, TXT, DATA")

st.markdown("---")
st.markdown("<center>Persistent Visual Dashboard | Built with Streamlit</center>", unsafe_allow_html=True)
