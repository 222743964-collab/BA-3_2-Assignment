"""Trains the credit-risk model on data/credit_data.csv and saves model/model.joblib"""
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from features import add_features, NUMERIC, CATEGORICAL

df = pd.read_csv("data/credit_data.csv")
X, y = add_features(df.drop(columns="default")), df["default"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
                        remainder="passthrough")
model = Pipeline([("pre", pre), ("gb", GradientBoostingClassifier(
    n_estimators=250, max_depth=3, learning_rate=0.06, subsample=0.8, random_state=7))])
model.fit(X_tr[CATEGORICAL + NUMERIC], y_tr)

proba = model.predict_proba(X_te[CATEGORICAL + NUMERIC])[:, 1]
print("ROC AUC:", round(roc_auc_score(y_te, proba), 3))
print(classification_report(y_te, proba > 0.5, zero_division=0))

joblib.dump({"model": model, "medians": X_tr[NUMERIC].median().to_dict(),
             "auc": float(roc_auc_score(y_te, proba))}, "model/model.joblib")
print("Saved model/model.joblib")

