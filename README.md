# Loan Performance Prediction | Credit Risk Classification

## Project Overview

This project develops a machine learning system to predict loan performance by classifying loans as **Good** or **Bad** based on applicant information, financial attributes and loan-related features.

The objective is to support credit-risk assessment by identifying potentially risky applicants and evaluating different loan approval policies using predicted probabilities.

## Business Problem

Lending institutions need to assess the risk associated with approving loan applications.

This project explores how machine learning can help:
- Identify applicants with a higher likelihood of loan default or poor repayment performance.
- Understand factors associated with loan performance.
- Compare classification models using credit-risk evaluation metrics.
- Simulate approval policies using different probability thresholds.

## Dataset Description

The original assignment contained two datasets:

**1. Application Data**
- Applicant demographics
- Income and residence information
- Banking information
- Credit-related scores
- Requested loan information
- Repayment-related attributes

**2. Loan Performance**
- Loan identifier
- Actual loan performance (`Good` or `Bad`)

The datasets were matched using customer identifiers to construct a supervised learning dataset.

*Note: Original applicant-level data is not included in this public repository due to privacy and data-sharing considerations.*

## Project Workflow

### 1. Data Cleaning and Preparation
- Inspected data types and missing values.
- Standardized customer identifiers.
- Matched application records with loan performance outcomes.
- Converted the target into binary format:
  - `0`: Good loan
  - `1`: Bad loan
- Investigated unusual credit-score values and missing-value indicators.

### 2. Exploratory Data Analysis
- Analyzed Good vs Bad loan distribution.
- Examined categorical variables and their relationship with loan performance.
- Studied numerical feature correlations.
- Investigated monthly loan performance patterns.
- Identified potential risk indicators.

### 3. Feature Engineering

Created additional features to represent applicant financial characteristics:

| Feature | Description |
|---|---|
| `age` | Applicant age at application time |
| `rent_to_income` | Monthly rent relative to monthly income |
| `loan_to_income` | Requested loan amount relative to monthly income |
| `fico_mean` | Average of available FICO-related scores |
| `purpose_group` | Grouped loan-purpose categories |
| `has_other_phone` | Indicator for availability of another phone type |

### 4. Data Preprocessing

Used Scikit-learn pipelines and ColumnTransformer.

**Numerical features**
- Median imputation
- StandardScaler

**Categorical features**
- Most-frequent imputation
- OneHotEncoder with `handle_unknown="ignore"`

Pipelines were used to keep preprocessing consistent and reduce data leakage during cross-validation.

### 5. Model Development

Compared three classification algorithms:

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline classification model |
| Random Forest | Ensemble learning using bagging |
| Gradient Boosting | Sequential ensemble learning |

### 6. Model Evaluation

Used five-fold Stratified Cross-Validation for model comparison.

Evaluation metrics:
- ROC-AUC
- Kolmogorov-Smirnov (KS) Statistic
- Accuracy
- Precision
- Recall

Logistic Regression was selected for further analysis based on the project's model evaluation and interpretability requirements.

### 7. Model Interpretability

Examined Logistic Regression coefficients to understand the direction and relative strength of feature relationships with the Bad-loan outcome.

This provided an interpretable view of the model's learned relationships.

### 8. Loan Approval Policy Analysis

Generated out-of-fold predicted probabilities using cross-validation and evaluated different approval thresholds.

Policy:

- Approve when predicted probability of a Bad loan is below the selected cutoff.
- Reject otherwise.

Compared:
- Approval percentage
- Bad-loan rate among approved applicants
- Good loans rejected
- Bad loans avoided

The analysis demonstrates the trade-off between lending volume and credit risk.

**Important:** The cutoff analysis is a policy simulation. The final production threshold requires business-cost analysis and validation on an untouched test dataset.

### 9. Model Serialization

Saved the trained Scikit-learn pipeline using Joblib.

```python
import joblib

joblib.dump(final_model, "loan_model.joblib")
```

Load the saved model:

```python
loaded_model = joblib.load("loan_model.joblib")
```

Generate predictions:

```python
probabilities = loaded_model.predict_proba(new_applicants)[:, 1]
```

The model returns the predicted probability of a Bad loan for each applicant.

## Technology Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Development:** Jupyter Notebook


## Limitations

- The dataset is relatively small and historical.
- Historical patterns may not represent future lending conditions.
- Model probabilities require calibration assessment before being interpreted as actual default probabilities.
- The selected approval threshold requires business validation.
- The model should support, not independently determine, lending decisions.
- Fairness, regulatory compliance and data privacy must be assessed before real-world deployment.

## Future Improvements

- Evaluate time-based validation to measure future performance.
- Perform probability calibration.
- Add cost-sensitive threshold optimization.
- Explore SHAP for individual prediction explanations.
- Monitor model drift and changing credit-risk patterns.
- Build an API for applicant-level risk prediction.

## Disclaimer

This project was developed for educational and machine learning assessment purposes. It is not a production-ready lending decision system. The model should not be used to make actual credit decisions without appropriate validation, regulatory review, privacy safeguards and fairness assessments.
