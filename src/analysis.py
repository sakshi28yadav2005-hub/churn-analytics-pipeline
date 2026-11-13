"""Run the SQL analysis queries and save results + charts."""
import re
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import DB_PATH, FIG_DIR, REPORTS_DIR, SQL_DIR


def load_queries() -> dict[str, str]:
    text = (SQL_DIR / "analysis.sql").read_text()
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_all() -> dict[str, pd.DataFrame]:
    with sqlite3.connect(DB_PATH) as conn:
        return {name: pd.read_sql_query(q, conn) for name, q in load_queries().items()}


def make_charts(results: dict[str, pd.DataFrame]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rev = results["monthly_revenue"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rev["month"], rev["revenue"], marker="o", color="#2563eb")
    ax.set_title("Monthly Revenue")
    ax.set_ylabel("Revenue ($)")
    ax.set_xticks(rev["month"][::3])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_revenue.png", dpi=150)
    plt.close(fig)

    pc = results["churn_by_plan_and_contract"].pivot(
        index="plan", columns="contract_type", values="churn_rate_pct"
    )
    ax = pc.plot(kind="bar", figsize=(7, 4), color=["#16a34a", "#dc2626"])
    ax.set_title("Churn Rate by Plan and Contract Type")
    ax.set_ylabel("Churn rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "churn_by_plan_contract.png", dpi=150)
    plt.close()

    tk = results["churn_by_support_tickets"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(tk["tickets_capped_at_5"].astype(str), tk["churn_rate_pct"], color="#f59e0b")
    ax.set_title("Churn Rate vs. Support Tickets (5 = 5+)")
    ax.set_xlabel("Support tickets")
    ax.set_ylabel("Churn rate (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "churn_by_tickets.png", dpi=150)
    plt.close(fig)


def main() -> None:
    results = run_all()
    REPORTS_DIR.mkdir(exist_ok=True)
    for name, df in results.items():
        df.to_csv(REPORTS_DIR / f"{name}.csv", index=False)
        print(f"\n=== {name} ===\n{df.head(8).to_string(index=False)}")
    make_charts(results)
    print(f"\nCharts saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
