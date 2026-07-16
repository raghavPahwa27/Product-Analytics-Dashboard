-- avg_review_score.sql
-- Business question: What is customer satisfaction by category, and how does
-- it trend over time?
--
-- JOINs used:
--   reviews → orders  (INNER JOIN)
--     Why: reviews has no timestamp that's reliable for ordering. We use
--     order_purchase_timestamp from orders for time-based grouping. Also
--     needed to filter to delivered orders (delivered = a real purchase).
--
--   orders → order_items  (INNER JOIN)
--     Why: We need to connect the review (on an order) to the product category
--     that was purchased. order_items is the bridge between orders and products.
--
--   order_items → products  (INNER JOIN)
--     Why: To get the product category.
--
--   products → product_category_translation  (LEFT JOIN)
--     Why: English category names. LEFT JOIN keeps uncategorised products.

SELECT
    COALESCE(t.product_category_name_english, p.product_category_name, 'unknown')
                                                AS category,
    COUNT(r.review_id)                          AS total_reviews,
    ROUND(AVG(r.review_score), 2)               AS avg_score,
    SUM(CASE WHEN r.review_score >= 4 THEN 1 ELSE 0 END)  AS positive_reviews,
    SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END)  AS negative_reviews
FROM reviews r
INNER JOIN orders o   ON r.order_id           = o.order_id
INNER JOIN order_items oi ON o.order_id       = oi.order_id
INNER JOIN products p     ON oi.product_id    = p.product_id
LEFT  JOIN product_category_translation t
           ON p.product_category_name         = t.product_category_name
WHERE o.order_status = 'delivered'
GROUP BY category
ORDER BY total_reviews DESC
LIMIT 25;
