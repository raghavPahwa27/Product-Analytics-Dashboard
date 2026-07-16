-- repeat_customers.sql
-- Business question: How many customers have placed more than one order?
--
-- JOINs used:
--   orders → customers  (INNER JOIN)
--     Why: The `customer_id` in orders is order-scoped (one per order).
--     `customer_unique_id` in customers is person-scoped. To count how many
--     times the same real person ordered, we must JOIN to customers and group
--     by customer_unique_id — not customer_id.
--
-- The outer query then buckets customers by order count for a distribution view.

WITH customer_order_counts AS (
    SELECT
        c.customer_unique_id,
        COUNT(o.order_id)  AS order_count
    FROM orders o
    INNER JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
)
SELECT
    order_count,
    COUNT(customer_unique_id)  AS num_customers
FROM customer_order_counts
GROUP BY order_count
ORDER BY order_count;
