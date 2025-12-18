import pickle
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
META_PATH = BASE_DIR / "vector_db" / "metadata.pkl"

def canonicalize(url):
    if not isinstance(url, str):
        return ""
    url = url.strip().lower().rstrip("/")
    url = url.replace(
        "https://www.shl.com/solutions/products/",
        "https://www.shl.com/"
    )
    return url

# Load metadata
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

metadata_urls = {canonicalize(m["url"]) for m in metadata}

print(f"Total metadata entries: {len(metadata_urls)}")

# Load train set
df = pd.read_csv("backend/data/train.csv")

missing = 0
present = 0

for _, row in df.iterrows():
    gt_url = canonicalize(row["Assessment_url"])

    if "/products/product-catalog/view/" not in gt_url:
        continue

    if gt_url in metadata_urls:
        present += 1
    else:
        missing += 1
        print("❌ NOT FOUND IN METADATA:", gt_url)

print("\nSUMMARY")
print("Present in metadata:", present)
print("Missing from metadata:", missing)
