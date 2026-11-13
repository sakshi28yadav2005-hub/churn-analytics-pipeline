"""Generate a realistic, messy synthetic subscription-business dataset.

Why synthetic? It keeps the repo self-contained and reproducible (no downloads),
and lets us deliberately inject real-world problems (nulls, duplicates, bad
values, inconsistent casing) so the ETL step has something meaningful to fix.

Outputs (data/raw/):
    customers.csv      one row per customer
    transactions.csv   monthly billing records per customer
"""
import numpy as np
import pandas as pd

from src.config import N_CUSTOMERS, RANDOM_STATE, RAW_DIR

PLANS = {"Basic": 9.99, "Standard": 19.99, "Premium": 34.99}
REGIONS = ["North", "South", "East", "West", "Central"]
CHANNELS = ["Organic", "Referral", "Paid Ads", "Email"]


def generate_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    signup = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 730, n), unit="D"
    )
    plan = rng.choice(list(PLANS), n, p=[0.45, 0.35, 0.20])
    age = np.clip(rng.normal(36, 11, n).round(), 18, 80).astype(int)
    tickets = rng.poisson(1.5, n)
    logins = np.clip(rng.normal(14, 6, n).round(), 0, 40).astype(int)
    contract = rng.choice(["Monthly", "Annual"], n, p=[0.65, 0.35])

    # Churn probability driven by realistic factors -> the model has a real signal.
    logit = (
        -2.3
        + 0.55 * tickets
        - 0.14 * (logins - 14)
        + 1.1 * (contract == "Monthly")
        + 0.4 * (plan == "Premium")
        - 0.03 * (age - 36)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "signup_date": signup,
            "age": age.astype(float),
            "region": rng.choice(REGIONS, n),
            "acquisition_channel": rng.choice(CHANNELS, n, p=[0.35, 0.2, 0.3, 0.15]),
            "plan": plan,
            "contract_type": contract,
            "support_tickets": tickets,
            "avg_monthly_logins": logins,
            "churned": churn.astype(int),
        }
    )

    # --- inject data-quality problems ---
    df.loc[rng.choice(n, 120, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 40, replace=False), "age"] = -1  # invalid values
    lower = rng.choice(n, 200, replace=False)
    df.loc[lower, "region"] = df.loc[lower, "region"].str.lower()
    df.loc[rng.choice(n, 150, replace=False), "region"] = None
    upper = rng.random(n) < 0.05
    df.loc[upper, "plan"] = df.loc[upper, "plan"].str.upper()
    dupes = df.sample(60, random_state=RANDOM_STATE)
    return pd.concat([df, dupes], ignore_index=True)


def generate_transactions(customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    base = customers.drop_duplicates("customer_id")
    rows = []
    end = pd.Timestamp("2025-01-01")
    for r in base.itertuples():
        plan = str(r.plan).title()
        months = pd.date_range(r.signup_date, end, freq="MS")
        if r.churned:  # churned customers stop billing partway through
            keep = max(1, int(len(months) * rng.uniform(0.3, 0.9)))
            months = months[:keep]
        for m in months:
            amount = PLANS[plan] * (0.9 if r.contract_type == "Annual" else 1.0)
            rows.append((r.customer_id, m, round(amount, 2)))
    tx = pd.DataFrame(rows, columns=["customer_id", "billing_month", "amount"])
    tx.insert(0, "transaction_id", np.arange(1, len(tx) + 1))
    return tx


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    customers = generate_customers(N_CUSTOMERS, rng)
    transactions = generate_transactions(customers, rng)
    customers.to_csv(RAW_DIR / "customers.csv", index=False)
    transactions.to_csv(RAW_DIR / "transactions.csv", index=False)
    print(f"customers.csv: {len(customers):,} rows | transactions.csv: {len(transactions):,} rows")


if __name__ == "__main__":
    main()
