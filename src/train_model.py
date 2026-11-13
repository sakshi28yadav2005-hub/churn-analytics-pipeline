"""Train and evaluate churn-prediction models using data from the SQL warehouse."""
import json
import sqlite3

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DB_PATH, FIG_DIR, MODEL_DIR, RANDOM_STATE, REPORTS_DIR

NUMERIC = ["age", "support_tickets", "avg_monthly_logins"]
CATEGORICAL = ["region", "acquisition_channel", "plan", "contract_type"]
TARGET = "churned"


def load_features() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM customers", conn)


def build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    df = load_features()
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE, max_depth=2),
    }

    metrics, best_name, best_auc, best_pipe = {}, None, 0.0, None
    for name, model in candidates.items():
        pipe = build_pipeline(model)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
        pipe.fit(X_train, y_train)
        test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        metrics[name] = {"cv_roc_auc": round(cv_auc, 4), "test_roc_auc": round(test_auc, 4)}
        print(f"{name:22s} CV AUC={cv_auc:.3f} | Test AUC={test_auc:.3f}")
        if test_auc > best_auc:
            best_name, best_auc, best_pipe = name, test_auc, pipe

    print(f"\nBest model: {best_name}\n")
    print(classification_report(y_test, best_pipe.predict(X_test)))

    MODEL_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_DIR / "churn_model.joblib")
    (REPORTS_DIR / "model_metrics.json").write_text(
        json.dumps({"best_model": best_name, "models": metrics}, indent=2)
    )

    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test, y_test, cmap="Blues")
    plt.title(f"Confusion Matrix ({best_name})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Interpretability: coefficients of the (always fitted) logistic regression
    lr = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X_train, y_train)
    names = lr.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(lr.named_steps["model"].coef_[0], index=names).sort_values()
    top = pd.concat([coefs.head(6), coefs.tail(6)])
    ax = top.plot(kind="barh", figsize=(8, 5), color=["#16a34a" if v < 0 else "#dc2626" for v in top])
    ax.set_title("Top Churn Drivers (logistic regression coefficients)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "churn_drivers.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
