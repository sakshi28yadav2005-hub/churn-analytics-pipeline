-- Each query is preceded by a "-- name: <id>" line; src/analysis.py runs them all.

-- name: monthly_revenue
SELECT strftime('%Y-%m', billing_month) AS month,
       ROUND(SUM(amount), 2)            AS revenue,
       COUNT(DISTINCT customer_id)      AS active_customers
FROM transactions
GROUP BY month
ORDER BY month;

-- name: churn_by_plan_and_contract
SELECT plan,
       contract_type,
       COUNT(*)                          AS customers,
       ROUND(100.0 * AVG(churned), 1)    AS churn_rate_pct
FROM customers
GROUP BY plan, contract_type
ORDER BY churn_rate_pct DESC;

-- name: churn_by_channel
SELECT acquisition_channel,
       COUNT(*)                       AS customers,
       ROUND(100.0 * AVG(churned), 1) AS churn_rate_pct
FROM customers
GROUP BY acquisition_channel
ORDER BY churn_rate_pct DESC;

-- name: revenue_share_by_plan
-- Window functions: each plan's share of total revenue + running (cumulative) share.
WITH plan_rev AS (
    SELECT c.plan,
           COUNT(DISTINCT c.customer_id)        AS customers,
           ROUND(SUM(t.amount), 2)              AS revenue
    FROM customers c
    JOIN transactions t USING (customer_id)
    GROUP BY c.plan
)
SELECT plan,
       customers,
       revenue,
       ROUND(revenue / customers, 2)                                   AS revenue_per_customer,
       ROUND(100.0 * revenue / SUM(revenue) OVER (), 1)                AS revenue_share_pct,
       ROUND(100.0 * SUM(revenue) OVER (ORDER BY revenue DESC)
             / SUM(revenue) OVER (), 1)                                AS cumulative_share_pct
FROM plan_rev
ORDER BY revenue DESC;

-- name: month_over_month_growth
-- Window function LAG(): month-over-month revenue growth.
WITH m AS (
    SELECT strftime('%Y-%m', billing_month) AS month, SUM(amount) AS revenue
    FROM transactions
    GROUP BY month
)
SELECT month,
       ROUND(revenue, 2) AS revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM m
ORDER BY month;

-- name: cohort_retention
-- Customers from each signup-quarter cohort still billing N months after signup.
WITH cohorts AS (
    SELECT customer_id,
           signup_date,
           strftime('%Y', signup_date) || '-Q' ||
               ((CAST(strftime('%m', signup_date) AS INTEGER) + 2) / 3) AS cohort
    FROM customers
),
activity AS (
    SELECT c.cohort, c.customer_id,
           (CAST(strftime('%Y', t.billing_month) AS INTEGER) - CAST(strftime('%Y', c.signup_date) AS INTEGER)) * 12
         + (CAST(strftime('%m', t.billing_month) AS INTEGER) - CAST(strftime('%m', c.signup_date) AS INTEGER)) AS month_offset
    FROM cohorts c
    JOIN transactions t USING (customer_id)
)
SELECT cohort,
       month_offset,
       COUNT(DISTINCT customer_id) AS active_customers
FROM activity
WHERE month_offset IN (1, 3, 6, 12)
GROUP BY cohort, month_offset
ORDER BY cohort, month_offset;

-- name: churn_by_support_tickets
SELECT MIN(support_tickets, 5)        AS tickets_capped_at_5,
       COUNT(*)                       AS customers,
       ROUND(100.0 * AVG(churned), 1) AS churn_rate_pct
FROM customers
GROUP BY tickets_capped_at_5
ORDER BY tickets_capped_at_5;
