from dotenv import load_dotenv
load_dotenv()

import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="SHL Assessment Recommendation API")

# -------------------------------------------------
# ROOT ENDPOINT (RENDER HEALTH CHECK NEEDS THIS)
# -------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok"}

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}

# -------------------------------------------------
# LAZY-LOADED RECOMMENDER
# -------------------------------------------------
_recommender = None

def get_recommender():
    global _recommender
    if _recommender is None:
        print("Initializing SHLRecommender...")
        from backend.rag.recommender import SHLRecommender
        _recommender = SHLRecommender()
    return _recommender

# -------------------------------------------------
# REQUEST / RESPONSE MODELS
# -------------------------------------------------
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

# -------------------------------------------------
# RECOMMEND ENDPOINT (LLM ENABLED)
# -------------------------------------------------
@app.post("/recommend", response_model=List[AssessmentResponse])
def recommend(req: QueryRequest):
    try:
        recommender = get_recommender()
        results = recommender.recommend(
            req.query,
            top_k=req.top_k,
            use_llm=True
        )

        return [
            {
                "url": r.get("url"),
                "name": r.get("name"),
                "adaptive_support": r.get("adaptive_support", "No"),
                "description": r.get("description", ""),
                "duration": int(r.get("duration", 0)),
                "remote_support": r.get("remote_support", "Yes"),
                "test_type": r.get("test_types_full", [])
            }
            for r in results
        ]

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# EVALUATION ENDPOINT (LLM DISABLED)
# -------------------------------------------------
@app.post("/recommend_eval")
def recommend_eval(req: QueryRequest):
    try:
        recommender = get_recommender()
        results = recommender.recommend(
            req.query,
            top_k=req.top_k,
            use_llm=False
        )
        return [{"url": r.get("url")} for r in results]

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

