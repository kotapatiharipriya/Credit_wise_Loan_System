# Loan Approval Prediction System

This Streamlit application predicts whether a loan is likely to be approved using the machine-learning workflow developed in `Loan_Approval_System.ipynb`.

## Live Demo

[Open the Streamlit App](https://credit-wise-loan-system.streamlit.app)


## Run Locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The application expects `loan_approval_data.csv` to be in the same folder as `app.py`.

## Features

- Interactive applicant information form
- Loan approval prediction
- Estimated approval probability
- Model performance metrics
- Dataset preview
- Public deployment using Streamlit Community Cloud

## Model Performance

- Accuracy: 86.5%
- Precision: 78.3%
- Recall: 77.1%
- F1 Score: 77.7%

## Model Workflow

- Missing numerical values are replaced with the mean.
- Missing categorical values are replaced with the most frequent value.
- Categorical variables are encoded.
- Squared DTI ratio and squared credit score features are added.
- Numerical features are standardized.
- Gaussian Naive Bayes is used for prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Deployment

This application is deployed using Streamlit Community Cloud.

To deploy the application:

- Repository: `kotapatiharipriya/Credit_wise_Loan_System`
- Branch: `main`
- Main file path: `app.py`

## Disclaimer

This application provides a machine-learning estimate for educational purposes. It should not be treated as a guaranteed lending decision.