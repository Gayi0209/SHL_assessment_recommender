from backend.llm.query_parser import parse_query
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "vector_db" / "faiss.index"
META_PATH = BASE_DIR / "vector_db" / "metadata.pkl"


class SHLRecommender:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(str(INDEX_PATH))

        with open(META_PATH, "rb") as f:
            self.metadata = pickle.load(f)

    def recommend(self, query: str, top_k=10, use_llm=True):
        # ✅ LLM usage
        if use_llm:
            intent_data = parse_query(query)
            intent = intent_data.get("intent", "mixed")
        else:
            intent = "mixed"  # neutral intent

        query_vec = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k * 5)
        candidates = [self.metadata[i] for i in indices[0]]

        technical, behavioral = [], []

        for c in candidates:
            raw = c.get("test_types_list", [])
            types = set(eval(raw)) if isinstance(raw, str) else set(raw)

            if "K" in types:
                technical.append(c)
            elif any(t in types for t in ["P", "C", "B"]):
                behavioral.append(c)

        if intent == "technical":
            return technical[:top_k]

        if intent == "behavioral":
            return behavioral[:top_k]

        # mixed
        half = top_k // 2
        return (technical[:half] + behavioral[:top_k - half])[:top_k]
