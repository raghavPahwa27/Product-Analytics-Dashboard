-- revenue_by_category.sql
-- Business question: Which product categories drive the most revenue?
--
-- JOINs used:
--   order_items → orders  (INNER JOIN)
--     Why: Restrict to delivered orders. Revenue from cancelled orders must
--     not appear in business reporting.
--
--   order_items → products  (INNER JOIN)
--     Why: We need the category name. Products with no category are excluded
--     intentionally — they cannot be grouped meaningfully.
--
--   products → product_category_translation  (LEFT JOIN)
--     Why: Not every Portuguese category has an English translation. LEFT JOIN
--     preserves all categories; COALESCE falls back to the Portuguese name.

SELECT
    COALESCE(t.product_category_name_english, p.product_category_name)  AS category,
    COUNT(DISTINCT oi.order_id)                                           AS total_orders,
    ROUND(SUM(oi.price), 2)                                               AS total_revenue,
    ROUND(AVG(oi.price), 2)                                               AS avg_item_price
FROM order_items oi
INNER JOIN orders  o  ON oi.order_id           = o.order_id
INNER JOIN products p  ON oi.product_id         = p.product_id
LEFT  JOIN product_category_translation t
           ON p.product_category_name           = t.product_category_name
WHERE o.order_status = 'delivered'
  AND p.product_category_name IS NOT NULL
GROUP BY category
ORDER BY total_revenue DESC;
