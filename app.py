import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Credit Wise Loan System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💳 Credit Wise Loan System")
st.markdown("### AI-Powered Loan Approval Prediction System")
st.markdown("---")

@st.cache_resource
def load_models():
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('onehot_encoder.pkl', 'rb') as f:
        ohe = pickle.load(f)
    
    
    return model, scaler, ohe

model, scaler, ohe = load_models()
st.success("✅ Model loaded successfully!")

categorical_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
numeric_features = [
    "Age",
    "Applicant_Income",
    "Coapplicant_Income",
    "Loan_Amount",
    "Savings",
    "Education_Level",
    "DTI_Ratio_sq",
    "Credit_Score_sq",
]
feature_columns = numeric_features + list(ohe.get_feature_names_out(categorical_cols))

# Customer-facing interface (only prediction)
st.subheader("Predict Loan Approval")

# Fixed threshold (bank policy - not customer controlled)
threshold = 0.50  # 50% threshold

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Personal Information")
    age = st.number_input("Age", value=35, step=1)
    applicant_income = st.number_input("Applicant Income (Monthly)", value=30000, step=500)
    coapplicant_income = st.number_input("Co-applicant Income (Monthly)", value=20000, step=500)
    credit_score = st.number_input("Credit Score", value=700, step=10)
    savings = st.number_input("Savings", value=25000, step=1000)

with col2:
    st.markdown("### Loan Details")
    loan_amount = st.number_input("Loan Amount", value=100000, step=5000)
    dti_ratio_value = st.number_input(
        "DTI Ratio (0 to 1)",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.01,
        help="Debt-to-income ratio as a fraction (e.g., 0.35 = 35%)."
    )
    education_level = st.selectbox("Education Level", ["Not Graduate", "Graduate"])
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed", "Contract"])

col3, col4, col5 = st.columns(3)
with col3:
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
with col4:
    loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Home", "Business", "Education"])
with col5:
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

col6, col7 = st.columns(2)
with col6:
    gender = st.selectbox("Gender", ["Female", "Male"])
with col7:
    employer_category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Unemployed", "Business"])

st.markdown("---")

if st.button("🔮 Predict Loan Approval", use_container_width=True):
    # Map education level to numeric (0=Graduate, 1=Not Graduate - LabelEncoder alphabetical order)
    edu_encoded = 0 if education_level == "Graduate" else 1
    
    dti_sq = dti_ratio_value ** 2
    credit_score_sq = credit_score ** 2

    input_data = pd.DataFrame({
        'Age': [age],
        'Applicant_Income': [applicant_income],
        'Coapplicant_Income': [coapplicant_income],
        'Loan_Amount': [loan_amount],
        'Savings': [savings],
        'Education_Level': [edu_encoded],
        'DTI_Ratio_sq': [dti_sq],
        'Credit_Score_sq': [credit_score_sq]
    })
    
    cat_data = pd.DataFrame({
        'Employment_Status': [employment_status],
        'Marital_Status': [marital_status],
        'Loan_Purpose': [loan_purpose],
        'Property_Area': [property_area],
        'Gender': [gender],
        'Employer_Category': [employer_category]
    })
    
    encoded_cat = ohe.transform(cat_data)
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=ohe.get_feature_names_out(categorical_cols))
    
    input_df = pd.concat([input_data, encoded_cat_df], axis=1)
    
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[feature_columns]
    
    # DEBUG: Show feature count
    st.write(f"DEBUG: Input features: {input_df.shape[1]}")
    st.write(f"DEBUG: Feature columns: {feature_columns}")
    st.write(f"DEBUG: Input columns: {list(input_df.columns)}")
    
    # Convert to numpy array to avoid feature name validation issues
    input_scaled = scaler.transform(input_df.values)
    
    # Use Logistic Regression model prediction with calibration
    decision_score = model.decision_function(input_scaled)[0]
    # Normalize decision function to 0-1 probability scale
    # Apply sigmoid-like transformation
    approval_prob = 1 / (1 + np.exp(-decision_score))
    prediction = 1 if approval_prob >= threshold else 0
    
    st.markdown("---")
    st.markdown("### 📊 Prediction Results")
    
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        approval_status = "✅ APPROVED" if prediction == 1 else "❌ REJECTED"
        st.metric("Loan Status", approval_status)
    
    with result_col2:
        st.metric("Approval Probability", f"{approval_prob*100:.2f}%")
    
    with result_col3:
        rejection_prob = 1 - approval_prob
        st.metric("Rejection Probability", f"{rejection_prob*100:.2f}%")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = ['#FF6B6B', '#51CF66'] if prediction == 1 else ['#51CF66', '#FF6B6B']
    ax.barh(['Rejected', 'Approved'], [rejection_prob*100, approval_prob*100], color=colors)
    ax.axvline(x=threshold*100, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold*100:.1f}%)')
    ax.set_xlabel('Probability (%)')
    ax.set_title('Loan Approval Prediction Probability')
    ax.set_xlim([0, 100])
    ax.legend()
    ax.text(rejection_prob*100 + 1, 0, f'{rejection_prob*100:.2f}%', va='center')
    ax.text(approval_prob*100 + 1, 1, f'{approval_prob*100:.2f}%', va='center')
    st.pyplot(fig)

st.markdown("---")
st.markdown("<div style='text-align: center; padding: 1rem;'><p style='color: gray;'>Credit Wise Loan System | Built with Streamlit & ML 💻</p></div>", unsafe_allow_html=True)

