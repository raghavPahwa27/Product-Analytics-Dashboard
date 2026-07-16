-- payment_distribution.sql
-- Business question: How do customers prefer to pay (card, boleto, voucher)?
--
-- No JOIN needed — the payments table contains payment_type directly.
-- We aggregate at the order level first (subquery) to avoid counting
-- multi-instalment payments as separate payment-type events.

SELECT
    payment_type,
    COUNT(DISTINCT order_id)          AS total_orders,
    ROUND(SUM(payment_value), 2)      AS total_value,
    ROUND(AVG(payment_value), 2)      AS avg_value,
    ROUND(
        100.0 * COUNT(DISTINCT order_id)
        / SUM(COUNT(DISTINCT order_id)) OVER (),
        2
    )                                 AS pct_of_orders
FROM payments
GROUP BY payment_type
ORDER BY total_orders DESC;
