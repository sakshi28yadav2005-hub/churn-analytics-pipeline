import numpy as np
import pandas as pd

from src.etl import clean_customers, clean_transactions, validate


def _raw_customers():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 2, 3],
            "signup_date": ["2024-01-01"] * 4,
            "age": [30, -1, -1, np.nan],
            "region": ["north", None, None, " East "],
            "plan": ["BASIC", "Premium", "Premium", "standard"],
            "churned": [0, 1, 1, 0],
        }
    )


def test_duplicates_removed():
    assert clean_customers(_raw_customers())["customer_id"].is_unique


def test_invalid_ages_imputed():
    assert clean_customers(_raw_customers())["age"].between(18, 100).all()


def test_text_standardised():
    out = clean_customers(_raw_customers())
    assert set(out["plan"]) == {"Basic", "Premium", "Standard"}
    assert set(out["region"]) == {"North", "Unknown", "East"}


def test_orphan_transactions_dropped():
    cust = clean_customers(_raw_customers())
    tx = pd.DataFrame(
        {"transaction_id": [1, 2], "customer_id": [1, 99],
         "billing_month": ["2024-02-01"] * 2, "amount": [10.0, 10.0]}
    )
    out = clean_transactions(tx, cust["customer_id"])
    assert list(out["customer_id"]) == [1]
    validate(cust, out)
