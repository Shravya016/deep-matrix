# ChurnZero – Customer Churn Prediction

## Team: Deep Matrix

## Overview

Customer churn is a major challenge for financial institutions because acquiring a new customer is significantly more expensive than retaining an existing one. This project develops a machine learning-based churn prediction system that identifies customers who are likely to close their accounts or become inactive in the upcoming period.

The solution leverages customer demographics, banking relationships, transaction patterns, product holdings, credit behavior, digital engagement, complaint history, and marketing interactions to estimate churn probability and support proactive retention strategies.

---

## Problem Statement

Given a customer's profile, banking relationship, transactional behavior, digital engagement, complaint history, and marketing response data, predict whether the customer will churn in the upcoming period.

### Evaluation Metrics

- **Primary Metric:** PR-AUC (Precision-Recall Area Under Curve)
- **Secondary Metric:** F1 Score
- **Business Cost Framework**
  - False Negative Cost: ₹40,000
  - False Positive Cost: ₹500

---

## Dataset

### Training Dataset

**File:** `ChurnZero_Dataset_v1.csv`

- Records: 8,101
- Features: 97
- Target Variable: `churn`

### Test Dataset

**File:** `ChurnZero_Test_v1.csv`

- Records: 2,026
- Features: 97
- Target Variable: Not Available

---

## Feature Categories

### Customer Profile
- Age
- Gender
- Marital Status
- Education Level
- Occupation
- Annual Income
- City Tier

### Relationship & Tenure
- Tenure Months
- Onboarding Channel
- Relationship Type
- Number of Products
- Customer Lifetime Value

### Account & Transaction Behaviour
- Average Monthly Balance
- Current Balance
- Transaction Count
- Transaction Value
- UPI Transactions
- Cash Withdrawals

### Product Holding
- Savings Account
- Current Account
- Credit Card
- Personal Loan
- Home Loan
- Fixed Deposit
- Insurance Products
- Investment Products

### Credit & Loan Behaviour
- Credit Card Limit
- Credit Utilization Ratio
- EMI Delays
- Loan Outstanding Amount
- Default Risk Score

### Digital Banking Engagement
- Mobile App Logins
- Website Logins
- Digital Transaction Ratio
- Last Login Days
- Email Open Rate

### Service & Complaint Behaviour
- Complaint Count
- Resolution Time
- Escalation Count
- Satisfaction Score
- NPS Score

### Marketing & Retention
- Campaign Response
- Cross-Sell Offers
- Upsell Offers
- Retention Offers
- Competitor Offer Awareness

---

# Project Workflow

## 1. Exploratory Data Analysis (EDA)

Performed comprehensive data exploration to understand:

- Missing values
- Feature distributions
- Class imbalance
- Correlation patterns
- Customer churn trends
- Numerical and categorical feature behavior

### Visualizations

- Churn distribution
- Correlation heatmaps
- Balance distributions
- Transaction activity analysis
- Digital engagement analysis
- Complaint behavior analysis

---

## 2. Data Cleaning

### Missing Value Treatment

- Median imputation for numerical variables
- Special handling for `app_rating_given`
- Consistent train-test preprocessing

### Categorical Processing

#### Ordinal Encoding

Applied custom ordinal mappings for naturally ordered categories such as:

- Income Band
- Education Level
- Customer Segment
- Customer Feedback Sentiment

#### Label Encoding

Applied label encoding for nominal categorical variables while maintaining consistency between train and test datasets.

---

## 3. Feature Engineering

Several domain-driven features were created to improve predictive power.

### Engineered Features

- Balance utilization indicators
- Product ownership aggregations
- Digital engagement measures
- Complaint intensity metrics
- Credit utilization trends
- Customer relationship metrics
- Banking activity indicators

These features help capture behavioral patterns that are strongly associated with churn.

---

## 4. Class Imbalance Handling

The churn class represented a minority of customers.

### Technique Used

**SMOTE (Synthetic Minority Oversampling Technique)**

Configuration:

```python
SMOTE(
    sampling_strategy=0.4,
    random_state=42,
    k_neighbors=5
)
```

SMOTE was applied **only on training folds** to prevent data leakage.

---

## 5. Model Development

### Baseline Model

#### Logistic Regression

Used as an interpretable benchmark model for comparison against boosting methods.

Metrics:

- PR-AUC
- F1 Score
- Confusion Matrix

---

### Gradient Boosting Models

#### LightGBM

Advantages:

- Fast training
- Handles high-dimensional data
- Robust to missing values
- Strong performance on tabular datasets

#### XGBoost

Advantages:

- Powerful regularization
- Excellent predictive capability
- Handles non-linear relationships effectively

---

## 6. Hyperparameter Optimization

### Optuna

Used Optuna for automated hyperparameter tuning.

Optimized Parameters:

- Learning Rate
- Number of Leaves
- Maximum Depth
- Minimum Child Samples
- Feature Fraction
- Bagging Fraction
- Bagging Frequency
- L1 Regularization
- L2 Regularization

### Validation Strategy

- Stratified 5-Fold Cross Validation
- Average Precision (PR-AUC) optimization objective

---

## 7. Model Explainability

### SHAP Analysis

SHAP values were used to explain model predictions and identify the most influential churn drivers.

Insights obtained:

- Key factors increasing churn risk
- Product usage impact
- Digital engagement influence
- Complaint-related churn indicators
- Credit utilization effects

---

## 8. Model Evaluation

### Evaluation Metrics

#### Precision-Recall AUC (Primary)

Measures performance under class imbalance.

#### F1 Score

Balances precision and recall.

#### Confusion Matrix

Provides business interpretation of:

- True Positives
- True Negatives
- False Positives
- False Negatives

---

## Final Prediction Generation

Predictions were generated for the 2,026 customers in the test dataset.

### Output File

`ChurnZero_DeepMatrix_Predictions.csv`

### Output Columns

| Column | Description |
|----------|-------------|
| customer_id | Unique customer identifier |
| churn_prediction | Binary prediction (0/1) |
| churn_probability | Predicted probability of churn |

---

## Technologies Used

### Programming Language

- Python 3.12

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-Learn
- LightGBM
- XGBoost
- Imbalanced-Learn

### Hyperparameter Tuning

- Optuna

### Explainability

- SHAP

---

## Repository Structure

```text
.
├── ChurnZero_Dataset_v1.csv
├── ChurnZero_Test_v1.csv
├── ChurnZero_DeepMatrix_Code.ipynb
├── ChurnZero_DeepMatrix_Predictions.csv
├── README.md
└── assets/
```

---

## Reproducibility

### Install Dependencies

```bash
pip install numpy pandas scipy scikit-learn imbalanced-learn \
lightgbm xgboost shap optuna matplotlib seaborn
```

### Run Notebook

```bash
jupyter notebook ChurnZero_DeepMatrix_Code.ipynb
```

Execute all cells sequentially to reproduce results and generate predictions.

