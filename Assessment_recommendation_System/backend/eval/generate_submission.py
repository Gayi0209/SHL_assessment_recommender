import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/recommend_eval"
TOP_K = 10

df = pd.read_csv("backend/data/test.csv")

rows = []

for query in df["Query"]:
    response = requests.post(API_URL, json={"query": query})
    results = response.json()

    for r in results[:TOP_K]:
        rows.append({
            "Query": query,
            "Assessment_url": r["url"]
        })

pd.DataFrame(rows).to_csv("submission.csv", index=False)
print("✅ submission.csv generated")
