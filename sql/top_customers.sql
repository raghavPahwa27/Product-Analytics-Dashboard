-- top_customers.sql
-- Business question: Who are the highest-value customers (by lifetime spend)?
--
-- JOINs used:
--   orders → customers  (INNER JOIN)
--     Why: orders holds order-level data; customers holds the unique person ID
--     (customer_unique_id). We JOIN to identify the real individual behind
--     each order — critical because one person can appear in orders multiple
--     times with a different customer_id each time.
--
--   orders → payments  (INNER JOIN via subquery)
--     Why: Revenue is in the payments table. Pre-aggregating per order_id
--     prevents double-counting multi-payment orders.

SELECT
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    COUNT(DISTINCT o.order_id)      AS total_orders,
    ROUND(SUM(p.order_total), 2)    AS lifetime_value
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
INNER JOIN (
    SELECT order_id, SUM(payment_value) AS order_total
    FROM payments
    GROUP BY order_id
) p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
GROUP BY c.customer_unique_id
ORDER BY lifetime_value DESC
LIMIT 20;
