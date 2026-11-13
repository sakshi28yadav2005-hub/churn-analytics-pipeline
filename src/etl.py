"""ETL: extract raw CSVs -> clean & validate -> load into a SQLite warehouse."""
import logging
import sqlite3

import pandas as pd

from src.config import DB_PATH, PROCESSED_DIR, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("etl")


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the data-quality issues found during profiling."""
    n0 = len(df)
    df = df.drop_duplicates(subset="customer_id", keep="first").copy()
    log.info("Removed %d duplicate customers", n0 - len(df))

    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["plan"] = df["plan"].str.strip().str.title()
    df["region"] = df["region"].str.strip().str.title().fillna("Unknown")

    # invalid ages -> NaN, then impute with the median
    df.loc[~df["age"].between(18, 100), "age"] = float("nan")
    median_age = df["age"].median()
    log.info("Imputing %d missing/invalid ages with median=%.0f", df["age"].isna().sum(), median_age)
    df["age"] = df["age"].fillna(median_age).astype(int)
    return df


def clean_transactions(df: pd.DataFrame, valid_ids: pd.Series) -> pd.DataFrame:
    df = df.drop_duplicates("transaction_id").copy()
    df["billing_month"] = pd.to_datetime(df["billing_month"])
    df = df[df["amount"] > 0]
    orphans = ~df["customer_id"].isin(valid_ids)
    if orphans.any():
        log.warning("Dropping %d orphan transactions", orphans.sum())
    return df[~orphans]


def validate(customers: pd.DataFrame, transactions: pd.DataFrame) -> None:
    """Fail loudly if the cleaned data breaks basic expectations."""
    assert customers["customer_id"].is_unique, "customer_id must be unique"
    assert customers["age"].between(18, 100).all(), "age out of range"
    assert customers["churned"].isin([0, 1]).all(), "churned must be 0/1"
    assert not customers.isna().any().any(), "unexpected nulls after cleaning"
    assert transactions["customer_id"].isin(customers["customer_id"]).all(), "orphan FK"
    log.info("All data-quality checks passed")


def load_to_sqlite(customers: pd.DataFrame, transactions: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        transactions.to_sql("transactions", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_customer ON transactions(customer_id)")
    log.info("Loaded warehouse -> %s", DB_PATH)


def run() -> None:
    customers = clean_customers(pd.read_csv(RAW_DIR / "customers.csv"))
    transactions = clean_transactions(
        pd.read_csv(RAW_DIR / "transactions.csv"), customers["customer_id"]
    )
    validate(customers, transactions)
    load_to_sqlite(customers, transactions)


if __name__ == "__main__":
    run()
