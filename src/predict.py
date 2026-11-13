"""Score a single customer with the saved model (demo of model inference)."""
import joblib
import pandas as pd

from src.config import MODEL_DIR


def predict_churn(customer: dict) -> float:
    model = joblib.load(MODEL_DIR / "churn_model.joblib")
    return float(model.predict_proba(pd.DataFrame([customer]))[0, 1])


if __name__ == "__main__":
    example = {
        "age": 29, "support_tickets": 4, "avg_monthly_logins": 3,
        "region": "North", "acquisition_channel": "Paid Ads",
        "plan": "Premium", "contract_type": "Monthly",
    }
    print(f"Churn probability: {predict_churn(example):.1%}")
