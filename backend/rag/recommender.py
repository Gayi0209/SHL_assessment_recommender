import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

try:
    from backend.llm.query_parser import parse_query
except Exception:
    parse_query = None


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "vector_db" / "faiss.index"
META_PATH = BASE_DIR / "vector_db" / "metadata.pkl"


class SHLRecommender:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_PATH))

        print("Loading metadata...")
        with open(META_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"Loaded {len(self.metadata)} metadata entries")

    def recommend(self, query: str, top_k=10, use_llm=False):
        intent = "mixed"
        if use_llm and parse_query is not None:
            try:
                intent_data = parse_query(query)
                intent = intent_data.get("intent", "mixed")
            except Exception:
                intent = "mixed"
        query_vec = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k * 5)

        candidates = []
        for i in indices[0]:
            if i < len(self.metadata):
                candidates.append(self.metadata[i])

        technical, behavioral = [], []

        for c in candidates:
            raw = c.get("test_types_list", [])

            if isinstance(raw, str):
                raw = raw.strip("[]")
                types = set(
                    t.strip().strip("'").strip('"')
                    for t in raw.split(",")
                    if t.strip()
                )
            else:
                types = set(raw or [])

            if "K" in types:
                technical.append(c)
            elif any(t in types for t in ["P", "C", "B"]):
                behavioral.append(c)
        if intent == "technical":
            return technical[:top_k]

        if intent == "behavioral":
            return behavioral[:top_k]

        half = top_k // 2
        return (technical[:half] + behavioral[:top_k - half])[:top_k]
