import os, subprocess, sys, warnings
import joblib, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, jsonify, render_template, request
from features import add_features, NUMERIC, CATEGORICAL

app = Flask(__name__)

def load_bundle():
    """Load the saved model; if it was built with a different scikit-learn
    version (or is missing), retrain it on this machine automatically."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return joblib.load("model/model.joblib")
    except Exception as e:
        print(f"Saved model unusable ({type(e).__name__}); retraining...")
        subprocess.check_call([sys.executable, "train_model.py"])
        return joblib.load("model/model.joblib")

bundle = load_bundle()
model, medians = bundle["model"], bundle["medians"]

FIELDS = {  # name: (label, min, max)
    "age": ("Age", 18, 100),
    "annual_income": ("Annual income", 1, 10_000_000),
    "employment_years": ("Years employed", 0, 70),
    "credit_history_years": ("Years of credit history", 0, 70),
    "num_credit_accounts": ("Open credit accounts", 0, 50),
    "credit_utilization_pct": ("Credit utilization (%)", 0, 100),
    "missed_payments_12m": ("Missed payments, last 12 months", 0, 50),
    "existing_monthly_debt": ("Monthly debt payments", 0, 1_000_000),
    "loan_amount": ("Loan amount", 1, 10_000_000),
    "loan_term_months": ("Loan term (months)", 1, 480),
}
LABELS = {**{k: v[0] for k, v in FIELDS.items()},
          "home_ownership": "Housing situation", "debt_to_income": "Debt-to-income ratio",
          "loan_to_income": "Loan size vs. income"}

def predict(df):
    return model.predict_proba(df[CATEGORICAL + NUMERIC])[:, 1]

def band(p):
    return ("Low", "green") if p < 0.10 else ("Moderate", "amber") if p < 0.25 else \
           ("High", "orange") if p < 0.45 else ("Very high", "red")

@app.get("/")
def index():
    return render_template("index.html", fields=FIELDS, auc=round(bundle["auc"], 3))

@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    row = {}
    for k, (label, lo, hi) in FIELDS.items():
        try:
            v = float(data.get(k))
        except (TypeError, ValueError):
            return jsonify(error=f"{label} needs a number."), 400
        if not lo <= v <= hi:
            return jsonify(error=f"{label} must be between {lo:,} and {hi:,}."), 400
        row[k] = v
    if data.get("home_ownership") not in ("rent", "mortgage", "own"):
        return jsonify(error="Choose a housing situation."), 400
    row["home_ownership"] = data["home_ownership"]

    df = add_features(pd.DataFrame([row]))
    p = float(predict(df)[0])
