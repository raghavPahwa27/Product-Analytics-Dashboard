-- =============================================================================
-- schema.sql
-- Olist Brazilian E-Commerce — SQLite schema
--
-- Design: Star schema with `orders` as the central fact table.
-- SQLite does not enforce FK constraints by default.
-- Enable them per-connection with:  PRAGMA foreign_keys = ON;
-- =============================================================================

PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- 1. customers
--    Every row = one order's customer snapshot.
--    customer_unique_id links the same real person across multiple orders.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customers (
    customer_id               TEXT PRIMARY KEY,   -- FK target from orders
    customer_unique_id        TEXT NOT NULL,       -- real person identifier
    customer_zip_code_prefix  TEXT NOT NULL,
    customer_city             TEXT NOT NULL,
    customer_state            TEXT NOT NULL
);

-- -----------------------------------------------------------------------------
-- 2. geolocation
--    Latitude/longitude per zip-code prefix. No PK — one zip can have many
--    coordinates (street-level granularity). Linked via zip prefix to both
--    customers and sellers.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS geolocation (
    geolocation_zip_code_prefix  TEXT NOT NULL,
    geolocation_lat              REAL NOT NULL,
    geolocation_lng              REAL NOT NULL,
    geolocation_city             TEXT NOT NULL,
    geolocation_state            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_geo_zip ON geolocation (geolocation_zip_code_prefix);

-- -----------------------------------------------------------------------------
-- 3. sellers
--    One row per seller. Sellers fulfil order items.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sellers (
    seller_id                TEXT PRIMARY KEY,
    seller_zip_code_prefix   TEXT NOT NULL,
    seller_city              TEXT NOT NULL,
    seller_state             TEXT NOT NULL
);

-- -----------------------------------------------------------------------------
-- 4. product_category_translation
--    Maps Portuguese category names to English.
--    One row per category. PK = the Portuguese name.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS product_category_translation (
    product_category_name            TEXT PRIMARY KEY,
    product_category_name_english    TEXT NOT NULL
);

-- -----------------------------------------------------------------------------
-- 5. products
--    One row per product. FK into product_category_translation for English names.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS products (
    product_id                   TEXT PRIMARY KEY,
    product_category_name        TEXT,            -- may be NULL for some products
    product_name_length          INTEGER,
    product_description_length   INTEGER,
    product_photos_qty           INTEGER,
    product_weight_g             REAL,
    product_length_cm            REAL,
    product_height_cm            REAL,
    product_width_cm             REAL,

    FOREIGN KEY (product_category_name)
        REFERENCES product_category_translation (product_category_name)
);

-- -----------------------------------------------------------------------------
-- 6. orders
--    Central fact table. Every purchase event lives here.
--    customer_id → customers: who placed the order.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    order_id                          TEXT PRIMARY KEY,
    customer_id                       TEXT NOT NULL,
    order_status                      TEXT NOT NULL,
    order_purchase_timestamp          TEXT,
    order_approved_at                 TEXT,
    order_delivered_carrier_date      TEXT,
    order_delivered_customer_date     TEXT,
    order_estimated_delivery_date     TEXT,

    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders (customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_status   ON orders (order_status);
CREATE INDEX IF NOT EXISTS idx_orders_date     ON orders (order_purchase_timestamp);

-- -----------------------------------------------------------------------------
-- 7. order_items
--    One row per line item within an order (an order can have many items).
--    Composite PK: (order_id, order_item_id).
--    Links orders → products → sellers.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS order_items (
    order_id            TEXT NOT NULL,
    order_item_id       INTEGER NOT NULL,   -- item sequence within the order
    product_id          TEXT NOT NULL,
    seller_id           TEXT NOT NULL,
    shipping_limit_date TEXT,
    price               REAL NOT NULL,
    freight_value       REAL NOT NULL,

    PRIMARY KEY (order_id, order_item_id),
    FOREIGN KEY (order_id)   REFERENCES orders   (order_id),
    FOREIGN KEY (product_id) REFERENCES products  (product_id),
    FOREIGN KEY (seller_id)  REFERENCES sellers   (seller_id)
);

CREATE INDEX IF NOT EXISTS idx_items_product ON order_items (product_id);
CREATE INDEX IF NOT EXISTS idx_items_seller  ON order_items (seller_id);

-- -----------------------------------------------------------------------------
-- 8. payments
--    One row per payment instalment within an order.
--    An order can be split across payment types (card + voucher, etc.).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS payments (
    order_id              TEXT NOT NULL,
    payment_sequential    INTEGER NOT NULL,   -- instalment sequence
    payment_type          TEXT NOT NULL,
    payment_installments  INTEGER NOT NULL,
    payment_value         REAL NOT NULL,

    PRIMARY KEY (order_id, payment_sequential),
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
);

CREATE INDEX IF NOT EXISTS idx_payments_type ON payments (payment_type);

-- -----------------------------------------------------------------------------
-- 9. reviews
--    One row per review. An order may receive one review.
--    review_id is technically the PK but not always unique in the raw data,
--    so we use (review_id, order_id) as the composite PK.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reviews (
    review_id                 TEXT NOT NULL,
    order_id                  TEXT NOT NULL,
    review_score              INTEGER NOT NULL,   -- 1-5
    review_comment_title      TEXT,
    review_comment_message    TEXT,
    review_creation_date      TEXT,
    review_answer_timestamp   TEXT,

    PRIMARY KEY (review_id, order_id),
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
);

CREATE INDEX IF NOT EXISTS idx_reviews_score ON reviews (review_score);
