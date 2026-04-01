import pandas as pd

# Step 1: Load articles and compute per-product-type mean & std of estimated_price_usd
articles = pd.read_csv("articles.csv", usecols=["article_id", "product_type_name", "estimated_price_usd"])

type_stats = (
    articles.groupby("product_type_name")["estimated_price_usd"]
    .agg(type_mean="mean", type_std="std")
    .reset_index()
)

# Step 2: Join stats back to articles, then compute z-score normalized price per article
articles = articles.merge(type_stats, on="product_type_name", how="left")
articles["normalized_price"] = (
    (articles["estimated_price_usd"] - articles["type_mean"]) / articles["type_std"]
)

# If a product type has only one article, std=0 → z-score is undefined; set to 0.0
articles["normalized_price"] = articles["normalized_price"].fillna(0.0)

# Build article_id → normalized_price lookup
price_map = articles.set_index("article_id")["normalized_price"]

print(f"Articles processed: {len(articles)}")
print(f"Product types: {articles['product_type_name'].nunique()}")
print(f"Normalized price sample:\n{articles[['article_id','product_type_name','estimated_price_usd','type_mean','type_std','normalized_price']].head(10).to_string()}")

# Step 3: Process transactions_train.csv in chunks and replace price column
CHUNK_SIZE = 1_000_000
output_file = "transactions_train_new.csv"
first_chunk = True

for chunk in pd.read_csv("transactions_train.csv", chunksize=CHUNK_SIZE):
    chunk["article_id"] = chunk["article_id"].astype(int)
    chunk["price"] = chunk["article_id"].map(price_map)
    chunk.to_csv(output_file, mode="w" if first_chunk else "a", index=False, header=first_chunk)
    first_chunk = False

print(f"\nDone. Saved to {output_file}")
