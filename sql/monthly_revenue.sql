-- monthly_revenue.sql
-- Business question: How is total revenue trending month over month?
--
-- JOINs used:
--   orders → payments  (INNER JOIN)
--     Why: We only want orders that have a recorded payment. An order without
--     a payment row has no revenue to attribute. INNER JOIN naturally drops
--     cancelled or pending orders that were never charged.
--
-- Note: payment_value is summed per order first (subquery), then grouped by
-- month. This handles orders paid in multiple instalments without double-counting.

SELECT
    strftime('%Y-%m', o.order_purchase_timestamp)  AS month,
    COUNT(DISTINCT o.order_id)                      AS total_orders,
    ROUND(SUM(p.payment_value), 2)                  AS total_revenue
FROM orders o
INNER JOIN (
    SELECT order_id, SUM(payment_value) AS payment_value
    FROM payments
    GROUP BY order_id
) p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
GROUP BY month
ORDER BY month;
