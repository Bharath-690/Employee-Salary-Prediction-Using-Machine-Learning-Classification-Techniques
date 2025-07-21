import streamlit as st
import pandas as pd
import joblib

# ✅ Must be first Streamlit command
st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="💼",
    layout="centered"
)
# Add this directly after st.set_page_config
st.markdown("""
    <style>
    .main {
        background-color: #d6f0ff;  /* Soft light blue */
        padding: 20px;
        border-radius: 10px;
    }
    .css-1d391kg {  /* Sidebar background (optional) */
        background-color: red !important;
    }
    [data-testid="stSidebar"] {
        background-color: #e0f7fa;  /* light cyan */
        padding: 20px;
        border-radius: 10px;
    }
    .streamlit-expanderHeader {
        background-color: #cceeff !important;  /* Light blue */
        color: black !important;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_list.pkl")
    return model, encoders, scaler, features

# Load model and preprocessing tools
model, encoders, scaler, feature_list = load_artifacts()

# Define which features are numeric
numeric_features = ['age', 'hours-per-week', 'experience']

# UI
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("📋 Input Employee Details")

age = st.sidebar.slider("Age", 18, 70, 30)
education = st.sidebar.selectbox("Education Level", list(encoders["education"].classes_))
occupation = st.sidebar.selectbox("Occupation", list(encoders["occupation"].classes_))
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Build raw input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.write("### 🔍 Input Data")

st.write(input_df) 

# Encode categoricals
input_encoded = input_df.copy()
for col in ['education', 'occupation']:
    input_encoded[col] = encoders[col].transform(input_encoded[col])

# Reorder and scale all features
input_encoded = input_encoded[feature_list]
input_encoded = pd.DataFrame(scaler.transform(input_encoded), columns=feature_list)

# Predict
if st.button("🔮 Predict Salary Class"):
    pred = model.predict(input_encoded)[0]
    income_label = encoders["income"].inverse_transform([pred])[0]
    st.success(f"✅ Prediction: {income_label}")

# --- REPLACED Batch Prediction section ---

import seaborn as sns
import matplotlib.pyplot as plt
from model_utils import auto_ml_pipeline

st.markdown("---")
st.title("🧠 AutoML Pipeline with Visualization")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

from model_utils import auto_ml_pipeline

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    if st.button("🚀 Run AutoML"):
        with st.spinner("Processing the dataset..."):
            report, visuals = auto_ml_pipeline(df)

        st.success("✅ AutoML Process Completed!")

        st.subheader("📝 Model Summary Report")

        # Print Best Model Name and Problem Type
        st.markdown(f"**🧪 Problem Type:** {report.get('Problem Type')}")
        st.markdown(f"**🏆 Best Model:** `{report.get('Best Model')}`")

        # Show appropriate metrics
        if report["Problem Type"] == "classification":
            st.markdown(f"**F1 Score:** `{report.get('F1 Score'):.4f}`")
            st.markdown("**Classification Report:**")
            st.json(report.get("Classification Report"))
        else:
            st.markdown(f"**RMSE:** `{report.get('RMSE'):.4f}`")
            st.markdown(f"**R² Score:** `{report.get('R2 Score'):.4f}`")

        st.subheader("📈 Visualizations")

        # Display all matplotlib figures
        for fig in visuals:
            st.pyplot(fig)
