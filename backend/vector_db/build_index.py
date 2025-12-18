import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

DATA_PATH = "backend/data/shl_products_enriched.csv"
INDEX_PATH = "backend/vector_db/faiss.index"
META_PATH = "backend/vector_db/metadata.pkl"

def build_faiss_index():
    print("🔄 Loading enriched dataset...")
    df = pd.read_csv(DATA_PATH)

    texts = df["embedding_text"].tolist()

    print("🧠 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("⚡ Creating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    print("📦 Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    Path("backend/vector_db").mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    metadata = df.to_dict(orient="records")
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ FAISS index saved ({index.ntotal} vectors)")
    print("✅ Metadata saved")

if __name__ == "__main__":
    build_faiss_index()
