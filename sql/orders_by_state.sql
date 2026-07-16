-- orders_by_state.sql
-- Business question: Which Brazilian states generate the most orders and revenue?
--
-- JOINs used:
--   orders → customers  (INNER JOIN)
--     Why: The orders table has no location data. Customer state is only in
--     the customers table. This JOIN is the only way to get geography.
--
--   orders → payments  (INNER JOIN via subquery)
--     Why: Revenue lives in payments. Aggregating payment_value per order
--     first prevents double-counting for multi-instalment payments.

SELECT
    c.customer_state                  AS state,
    COUNT(DISTINCT o.order_id)        AS total_orders,
    ROUND(SUM(p.payment_value), 2)    AS total_revenue,
    ROUND(AVG(p.payment_value), 2)    AS avg_order_value
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
INNER JOIN (
    SELECT order_id, SUM(payment_value) AS payment_value
    FROM payments
    GROUP BY order_id
) p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
GROUP BY state
ORDER BY total_orders DESC;
