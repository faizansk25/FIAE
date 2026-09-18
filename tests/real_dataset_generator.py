"""Generate a realistic 50MB dataset with 20 different data types.

This creates a dataset mimicking a real-world e-commerce/financial scenario
with 20 distinct column types that test every FIAE operator family.
"""

import csv
import os
import random
from datetime import datetime, timedelta


def generate_dataset(output_path: str, n_rows: int = 100_000, target_size_mb: float = 10.0):
    """Generate a dataset with 20 different column types."""
    rng = random.Random(42)

    # Column definitions
    columns = [
        # 1. Integer (count)
        ("order_id", "integer"),
        # 2. Float (continuous numeric)
        ("price", "float"),
        # 3. Float (continuous numeric, skewed)
        ("revenue", "float_skewed"),
        # 4. Integer (percentage 0-100)
        ("discount_pct", "integer_range"),
        # 5. Float (ratio 0-1)
        ("conversion_rate", "float_ratio"),
        # 6. Categorical (low cardinality)
        ("category", "categorical_low"),
        # 7. Categorical (medium cardinality)
        ("product_type", "categorical_medium"),
        # 8. Categorical (high cardinality)
        ("customer_segment", "categorical_high"),
        # 9. Boolean
        ("is_returned", "boolean"),
        # 10. Datetime
        ("order_date", "datetime"),
        # 11. Datetime (hourly)
        ("timestamp", "datetime_hourly"),
        # 12. Text (short)
        ("product_name", "text_short"),
        # 13. Text (medium)
        ("description", "text_medium"),
        # 14. Float (monetary, many zeros)
        ("shipping_cost", "float_zeros"),
        # 15. Integer (ID-like, high cardinality)
        ("customer_id", "id_high"),
        # 16. Float (normal distribution)
        ("weight_kg", "float_normal"),
        # 17. Integer (ordinal 1-5)
        ("rating", "ordinal"),
        # 18. Float (temperature-like, negative possible)
        ("temperature", "float_signed"),
        # 19. String (country code)
        ("country", "string_code"),
        # 20. Float (time delta in hours)
        ("delivery_hours", "float_positive"),
    ]

    # Data pools
    categories = ["Electronics", "Clothing", "Home", "Sports", "Books", "Food", "Beauty", "Toys"]
    product_types = [
        "Laptop", "Phone", "Tablet", "Headphones", "Camera", "Watch",
        "Shirt", "Pants", "Shoes", "Jacket", "Dress", "Hat",
        "Chair", "Table", "Lamp", "Rug", "Curtain", "Shelf",
        "Ball", "Racket", "Weights", "Mat", "Bike", "Helmet",
        "Novel", "Textbook", "Comic", "Magazine", "Journal", "Guide",
        "Snack", "Drink", "Meal", "Ingredient", "Spice", "Sauce",
        "Shampoo", "Cream", "Lotion", "Makeup", "Perfume", "Soap",
        "Puzzle", "Doll", "Car", "Board Game", "LEGO", "Figure",
    ]
    segments = [f"Segment_{i}" for i in range(50)]
    countries = ["US", "UK", "DE", "FR", "JP", "CN", "IN", "BR", "CA", "AU",
                 "IT", "ES", "NL", "SE", "CH", "KR", "MX", "SG", "AE", "SA"]

    product_words = [
        "Premium", "Deluxe", "Standard", "Pro", "Ultra", "Mini", "Max", "Lite",
        "Wireless", "Smart", "Classic", "Modern", "Vintage", "Eco", "Organic",
        "HD", "4K", "USB", "Bluetooth", "Solar", "Bamboo", "Steel", "Wood",
    ]
    product_nouns = [
        "Widget", "Gadget", "Device", "Tool", "Kit", "Set", "Pack", "Bundle",
        "Adapter", "Charger", "Case", "Cover", "Stand", "Mount", "Holder",
    ]

    desc_words = [
        "High quality", "Best selling", "Top rated", "New arrival", "Limited edition",
        "Eco friendly", "Handmade", "Imported", "Lightweight", "Durable",
        "Waterproof", "Portable", "Rechargeable", "Adjustable", "Foldable",
    ]

    start_date = datetime(2024, 1, 1)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([col[0] for col in columns])

        for i in range(n_rows):
            # 1. order_id (integer)
            order_id = 1000000 + i

            # 2. price (float, right-skewed)
            price = round(rng.lognormvariate(3.0, 1.0), 2)
            price = max(0.01, min(price, 10000.0))

            # 3. revenue (float, highly skewed)
            quantity = rng.randint(1, 10)
            revenue = round(price * quantity * rng.uniform(0.8, 1.2), 2)

            # 4. discount_pct (integer 0-100)
            discount_pct = rng.choices(
                [0, 5, 10, 15, 20, 25, 30, 50, 75],
                weights=[40, 15, 15, 10, 8, 5, 4, 2, 1]
            )[0]

            # 5. conversion_rate (float 0-1)
            conversion_rate = round(rng.betavariate(2, 5), 4)

            # 6. category (low cardinality)
            category = rng.choice(categories)

            # 7. product_type (medium cardinality)
            product_type = rng.choice(product_types)

            # 8. customer_segment (high cardinality)
            customer_segment = rng.choice(segments)

            # 9. is_returned (boolean)
            is_returned = 1 if rng.random() < 0.08 else 0

            # 10. order_date (datetime)
            days_offset = rng.randint(0, 365)
            order_date = (start_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")

            # 11. timestamp (datetime with hours)
            hours_offset = rng.randint(0, 365 * 24)
            timestamp = (start_date + timedelta(hours=hours_offset)).strftime("%Y-%m-%d %H:%M:%S")

            # 12. product_name (short text)
            name_parts = [rng.choice(product_words), rng.choice(product_nouns)]
            product_name = " ".join(name_parts)

            # 13. description (medium text)
            n_desc_words = rng.randint(3, 8)
            desc = " ".join(rng.choices(desc_words, k=n_desc_words))

            # 14. shipping_cost (float, many zeros)
            shipping_cost = 0.0 if rng.random() < 0.3 else round(rng.uniform(2.99, 29.99), 2)

            # 15. customer_id (high cardinality ID)
            customer_id = f"CUST_{rng.randint(10000, 99999)}"

            # 16. weight_kg (float, normal distribution)
            weight_kg = round(max(0.01, rng.gauss(2.5, 1.5)), 2)

            # 17. rating (ordinal 1-5)
            rating = rng.choices([1, 2, 3, 4, 5], weights=[5, 10, 20, 35, 30])[0]

            # 18. temperature (float, signed)
            temperature = round(rng.gauss(20.0, 10.0), 1)

            # 19. country (string code)
            country = rng.choices(
                countries,
                weights=[30, 10, 8, 8, 7, 6, 5, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]
            )[0]

            # 20. delivery_hours (float, positive)
            delivery_hours = round(max(0.5, rng.gauss(48.0, 24.0)), 1)

            # Add some missing values (realistic)
            row = [
                order_id, price, revenue, discount_pct, conversion_rate,
                category, product_type, customer_segment, is_returned,
                order_date, timestamp, product_name, desc,
                shipping_cost, customer_id, weight_kg, rating,
                temperature, country, delivery_hours,
            ]

            # Randomly introduce missing values (2% chance per nullable column)
            nullable_cols = [1, 2, 4, 13, 15, 17, 19]  # price, revenue, conv_rate, shipping, weight, temp, delivery
            for col_idx in nullable_cols:
                if rng.random() < 0.02:
                    row[col_idx] = ""

            writer.writerow(row)

            if (i + 1) % 20000 == 0:
                print(f"  Generated {i + 1}/{n_rows} rows...")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Dataset: {output_path}")
    print(f"  Rows: {n_rows}")
    print(f"  Columns: {len(columns)}")
    print(f"  Size: {file_size_mb:.1f} MB")

    return output_path


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "real_test_data")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "ecommerce_20types.csv")
    generate_dataset(output_path, n_rows=100_000, target_size_mb=10.0)
