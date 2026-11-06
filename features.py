import pandas as pd

CATEGORICAL = ["home_ownership"]
NUMERIC = ["age", "annual_income", "employment_years", "credit_history_years",
           "num_credit_accounts", "credit_utilization_pct", "missed_payments_12m",
           "existing_monthly_debt", "loan_amount", "loan_term_months",
           "debt_to_income", "loan_to_income"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    payment = df["loan_amount"] / df["loan_term_months"] * 1.1
    df["debt_to_income"] = (df["existing_monthly_debt"] + payment) / (df["annual_income"] / 12)
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    return df

