-- monthly_orders.sql
-- Business question: How many orders are placed each month?
--
-- No JOIN needed — the orders table alone answers this question.
-- Filtering to 'delivered' removes noise from cancellations and test orders.

SELECT
    strftime('%Y-%m', order_purchase_timestamp)  AS month,
    COUNT(order_id)                               AS total_orders
FROM orders
WHERE order_status = 'delivered'
GROUP BY month
ORDER BY month;
