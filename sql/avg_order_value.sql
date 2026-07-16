-- avg_order_value.sql
-- Business question: What is the average value of a delivered order over time?
--
-- JOINs used:
--   orders → payments  (INNER JOIN via subquery)
--     Why: AOV requires total revenue per order. Because one order can have
--     multiple payment rows (different methods or instalments), we must SUM
--     payment_value per order_id first, then average the per-order totals.
--     Averaging directly from the payments table without this step would
--     average individual payment rows, not order totals — a common bug.

SELECT
    strftime('%Y-%m', o.order_purchase_timestamp)  AS month,
    COUNT(DISTINCT o.order_id)                      AS total_orders,
    ROUND(AVG(p.order_total), 2)                    AS avg_order_value
FROM orders o
INNER JOIN (
    SELECT order_id, SUM(payment_value) AS order_total
    FROM payments
    GROUP BY order_id
) p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
GROUP BY month
ORDER BY month;
