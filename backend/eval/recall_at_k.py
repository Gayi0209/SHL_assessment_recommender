import pandas as pd
import requests
from collections import defaultdict

API_URL = "http://127.0.0.1:8000/recommend_eval"
TOP_K = 10

def canonicalize_shl_url(url):
    if not isinstance(url, str):
        return ""
    url = url.strip().lower().rstrip("/")
    url = url.replace(
        "https://www.shl.com/solutions/products/",
        "https://www.shl.com/"
    )
    return url

df = pd.read_csv("backend/data/train.csv")

ground_truth = defaultdict(set)

for _, row in df.iterrows():
    url = canonicalize_shl_url(row["Assessment_url"])
    if "/products/product-catalog/view/" in url:
        ground_truth[row["Query"]].add(url)

# Remove empty queries
ground_truth = {
    q: urls for q, urls in ground_truth.items() if len(urls) > 0
}

recalls = []

for i, (query, true_urls) in enumerate(ground_truth.items(), start=1):
    response = requests.post(API_URL, json={"query": query})
    preds = {
        canonicalize_shl_url(r["url"])
        for r in response.json()[:TOP_K]
    }

    hits = preds & true_urls
    recall = len(hits) / len(true_urls)
    recalls.append(recall)

    print(f"[{i}] Recall: {recall:.2f}")

print(f"\n✅ FINAL Recall@{TOP_K}: {sum(recalls)/len(recalls):.4f}")
