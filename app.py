import numpy as np
import pandas as pd
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


DATA_PATH = "loan_approval_data.csv"
CATEGORICAL_COLUMNS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
]
NUMERIC_INPUTS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
]


@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)

    # Match the notebook's missing-value treatment.
    categorical_columns = df.select_dtypes(include=["object"]).columns
    numeric_columns = df.select_dtypes(include=["number"]).columns
    df[numeric_columns] = SimpleImputer(strategy="mean").fit_transform(
        df[numeric_columns]
    )
    df[categorical_columns] = SimpleImputer(strategy="most_frequent").fit_transform(
        df[categorical_columns]
    )

    # The applicant ID is not a useful predictive feature.
    df = df.drop(columns=["Applicant_ID"])

    education_encoder = LabelEncoder()
    df["Education_Level"] = education_encoder.fit_transform(df["Education_Level"])

    target_encoder = LabelEncoder()
    df["Loan_Approved"] = target_encoder.fit_transform(df["Loan_Approved"])

    one_hot_encoder = OneHotEncoder(
        drop="first", sparse_output=False, handle_unknown="ignore"
    )
    encoded = one_hot_encoder.fit_transform(df[CATEGORICAL_COLUMNS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=one_hot_encoder.get_feature_names_out(CATEGORICAL_COLUMNS),
        index=df.index,
    )
    df = pd.concat(
        [df.drop(columns=CATEGORICAL_COLUMNS), encoded_df], axis=1
    )

    # Match the notebook's feature engineering.
    df["DTI_Ratio_Sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_Sq"] = df["Credit_Score"] ** 2

    X = df.drop(columns=["Loan_Approved", "DTI_Ratio", "Credit_Score"])
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
    }

    return {
        "model": model,
        "scaler": scaler,
        "one_hot_encoder": one_hot_encoder,
        "education_encoder": education_encoder,
        "target_encoder": target_encoder,
        "feature_columns": X.columns.tolist(),
        "metrics": metrics,
        "data": df,
    }


def make_input_row(values, artifacts):
    row = pd.DataFrame([values])
    row["Education_Level"] = artifacts["education_encoder"].transform(
        [row.loc[0, "Education_Level"]]
    )[0]

    encoded = artifacts["one_hot_encoder"].transform(row[CATEGORICAL_COLUMNS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=artifacts["one_hot_encoder"].get_feature_names_out(
            CATEGORICAL_COLUMNS
        ),
    )
    row = pd.concat([row.drop(columns=CATEGORICAL_COLUMNS), encoded_df], axis=1)
    row["DTI_Ratio_Sq"] = row["DTI_Ratio"] ** 2
    row["Credit_Score_Sq"] = row["Credit_Score"] ** 2
    row = row.drop(columns=["DTI_Ratio", "Credit_Score"])
    return row.reindex(columns=artifacts["feature_columns"], fill_value=0)


st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Loan Approval Prediction System")
st.write(
    "Enter an applicant's information to estimate whether the loan is likely to be approved."
)

try:
    artifacts = train_model()
except FileNotFoundError:
    st.error("The file loan_approval_data.csv was not found. Keep it in the same folder as app.py.")
    st.stop()

with st.sidebar:
    st.header("Model Information")
    st.write("Final model: Gaussian Naive Bayes")
    st.caption("The model follows the preprocessing and feature-engineering steps from the notebook.")
    st.divider()
    st.subheader("Test Metrics")
    for name, value in artifacts["metrics"].items():
        st.metric(name, f"{value:.2%}")

with st.form("loan_form"):
    st.subheader("Applicant Information")
    left, middle, right = st.columns(3)

    with left:
        applicant_income = st.number_input("Applicant Income", min_value=0.0, value=10548.0, step=100.0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=5205.5, step=100.0)
        age = st.number_input("Age", min_value=18.0, max_value=100.0, value=40.0, step=1.0)
        dependents = st.number_input("Dependents", min_value=0.0, max_value=10.0, value=1.0, step=1.0)

    with middle:
        credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=678.0, step=1.0)
        existing_loans = st.number_input("Existing Loans", min_value=0.0, max_value=20.0, value=2.0, step=1.0)
        dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.34, step=0.01, format="%.2f")
        savings = st.number_input("Savings", min_value=0.0, value=9880.5, step=100.0)

    with right:
        collateral_value = st.number_input("Collateral Value", min_value=0.0, value=24321.0, step=100.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=21210.5, step=100.0)
        loan_term = st.number_input("Loan Term (months)", min_value=1.0, max_value=480.0, value=48.0, step=1.0)
        employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])

    st.subheader("Background Information")
    left, middle, right = st.columns(3)
    with left:
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Business", "Home", "Education"])
    with middle:
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    with right:
        gender = st.selectbox("Gender", ["Female", "Male"])
        employer_category = st.selectbox("Employer Category", ["Private", "Government", "Unemployed", "MNC", "Business"])

    submitted = st.form_submit_button("Predict Loan Approval", type="primary", use_container_width=True)

if submitted:
    values = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
    }
    input_row = make_input_row(values, artifacts)
    input_scaled = artifacts["scaler"].transform(input_row)
    prediction = artifacts["model"].predict(input_scaled)[0]
    probabilities = artifacts["model"].predict_proba(input_scaled)[0]
    predicted_label = artifacts["target_encoder"].inverse_transform([prediction])[0]
    yes_index = list(artifacts["target_encoder"].classes_).index("Yes")
    approval_probability = probabilities[yes_index]

    st.divider()
    if predicted_label == "Yes":
        st.success(f"✅ Prediction: Loan likely approved ({approval_probability:.1%} estimated probability)")
    else:
        st.error(f"❌ Prediction: Loan likely not approved ({approval_probability:.1%} estimated approval probability)")
    st.caption("This is a machine-learning estimate and should not be treated as a guaranteed lending decision.")

with st.expander("View dataset summary"):
    raw_df = pd.read_csv(DATA_PATH)
    st.write(f"Dataset shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
    st.dataframe(raw_df.head(10), use_container_width=True)
