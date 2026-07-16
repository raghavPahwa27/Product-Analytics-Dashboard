-- top_products.sql
-- Business question: Which products generate the most revenue?
--
-- JOINs used:
--   order_items → orders  (INNER JOIN)
--     Why: Filter to delivered orders only. order_items has no status column,
--     so we must reach into orders to apply the status filter.
--
--   order_items → products  (LEFT JOIN)
--     Why: Some items may reference a product_id that no longer exists in the
--     products table (deleted listings). LEFT JOIN retains those rows so we
--     don't silently drop revenue. We fall back to the raw product_id as name.

SELECT
    oi.product_id,
    COALESCE(p.product_category_name, 'unknown')  AS category,
    COUNT(DISTINCT oi.order_id)                    AS times_ordered,
    ROUND(SUM(oi.price), 2)                        AS total_revenue
FROM order_items oi
INNER JOIN orders o  ON oi.order_id   = o.order_id
LEFT  JOIN products p ON oi.product_id = p.product_id
WHERE o.order_status = 'delivered'
GROUP BY oi.product_id
ORDER BY total_revenue DESC
LIMIT 20;
