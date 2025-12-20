import faiss
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "vector_db" / "faiss.index"
META_PATH = BASE_DIR / "vector_db" / "metadata.pkl"
 

class SHLRecommender:
    def __init__(self):
        # Nothing heavy is loaded here
        self.model = None
        self.index = None
        self.metadata = None
        self.parse_query = None

    # -------------------------------------------------
    # LAZY LOADERS (CRITICAL FOR RENDER)
    # -------------------------------------------------
    def _load_model(self):
        if self.model is None:
            print("Loading embedding model...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _load_index(self):
        if self.index is None:
            print("Loading FAISS index...")
            self.index = faiss.read_index(str(INDEX_PATH))

    def _load_metadata(self):
        if self.metadata is None:
            print("Loading metadata...")
            with open(META_PATH, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"Loaded {len(self.metadata)} metadata entries")

    def _load_llm(self):
        if self.parse_query is None:
            try:
                from backend.llm.query_parser import parse_query
                self.parse_query = parse_query
            except Exception:
                self.parse_query = None

    # -------------------------------------------------
    # RECOMMENDATION LOGIC
    # -------------------------------------------------
    def recommend(self, query: str, top_k=10, use_llm=False):
        # Lazy load everything
        self._load_model()
        self._load_index()
        self._load_metadata()

        intent = "mixed"
        if use_llm:
            self._load_llm()
            if self.parse_query:
                try:
                    intent_data = self.parse_query(query)
                    intent = intent_data.get("intent", "mixed")
                except Exception:
                    intent = "mixed"

        query_vec = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_vec)

        _, indices = self.index.search(query_vec, top_k * 5)

        candidates = []
        for i in indices[0]:
            if i < len(self.metadata):
                candidates.append(self.metadata[i])

        technical, behavioral = [], []

        for c in candidates:
            raw = c.get("test_types_list", [])

            if isinstance(raw, str):
                raw = raw.strip("[]")
                types = {
                    t.strip().strip("'").strip('"')
                    for t in raw.split(",")
                    if t.strip()
                }
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

