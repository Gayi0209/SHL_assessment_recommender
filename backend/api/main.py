from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from backend.rag.recommender import SHLRecommender

app = FastAPI(title="SHL Assessment Recommendation API")

recommender = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


class AssessmentResponse(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/recommend")
def recommend(req: QueryRequest):
    global recommender

    if recommender is None:
        recommender = SHLRecommender()

    results = recommender.recommend(
        req.query,
        top_k=10,
        use_llm=True     # ✅ LLM USED
    )

    return [{"url": r["url"], "name": r["name"], "test_type": r["test_types_full"]} for r in results]

@app.post("/recommend_eval")
def recommend_eval(req: QueryRequest):
    global recommender

    if recommender is None:
        recommender = SHLRecommender()

    results = recommender.recommend(
        req.query,
        top_k=10,
        use_llm=False    # ❌ LLM DISABLED
    )

    return [{"url": r["url"]} for r in results]
