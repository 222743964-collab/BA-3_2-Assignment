"""Generates a synthetic credit dataset (data/credit_data.csv).
Synthetic = made-up but statistically plausible. Not real people."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
N = 8000

age = rng.integers(19, 75, N)
employment_years = np.clip(rng.gamma(2.2, 3.5, N), 0, age - 18).round(1)
annual_income = np.clip(rng.lognormal(10.9, 0.55, N) * (1 + employment_years / 40), 12000, 400000).round(-2)
home_ownership = rng.choice(["rent", "mortgage", "own"], N, p=[0.42, 0.38, 0.20])
credit_history_years = np.clip(rng.gamma(2.5, 4, N), 0, age - 18).round(1)
num_credit_accounts = np.clip(rng.poisson(4 + credit_history_years / 6), 0, 20)
credit_utilization_pct = np.clip(rng.beta(2, 3.5, N) * 100, 0, 100).round(1)
missed_payments_12m = np.clip(rng.poisson(0.3 + credit_utilization_pct / 60), 0, 12)
existing_monthly_debt = (annual_income / 12 * np.clip(rng.beta(2, 5, N) * 0.8, 0, 0.7)).round(0)
loan_amount = np.clip(rng.lognormal(9.4, 0.8, N), 1000, 150000).round(-2)
loan_term_months = rng.choice([12, 24, 36, 48, 60, 84], N, p=[.1, .15, .3, .15, .22, .08])

monthly_income = annual_income / 12
est_payment = loan_amount / loan_term_months * 1.1
dti = (existing_monthly_debt + est_payment) / monthly_income
loan_to_income = loan_amount / annual_income
