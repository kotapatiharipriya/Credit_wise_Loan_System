# Loan Approval Prediction System

This Streamlit application predicts whether a loan is likely to be approved using the machine-learning workflow developed in `Loan_Approval_System.ipynb`.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The application expects `loan_approval_data.csv` in the same folder as `app.py`.

## Deployment

Push this folder to GitHub, then deploy it through Streamlit Community Cloud by selecting:

- **Repository:** your GitHub repository
- **Branch:** `main`
- **Main file path:** `app.py`

## Model workflow

- Missing numerical values are replaced with the mean.
- Missing categorical values are replaced with the most frequent value.
- Categorical variables are encoded.
- Squared DTI ratio and squared credit score features are added.
- Numerical features are standardized.
- Gaussian Naive Bayes is used for prediction.
