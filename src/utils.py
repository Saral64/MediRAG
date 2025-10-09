import os
import json
import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

# -----------------------------
# Helper functions
# -----------------------------
def ends_with_ending_punctuation(s: str) -> bool:
    return s.endswith(('.', '?', '!'))

def concat(title: str, content: str) -> str:
    """Combine title and content into a single string."""
    title = title.strip()
    content = content.strip()
    if not ends_with_ending_punctuation(title):
        title += "."
    return f"{title} {content}"

# -----------------------------
# Custom SentenceTransformer with CLS token pooling
# -----------------------------
class CustomizeSentenceTransformer(SentenceTransformer):
    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode_cls_token=True)
        return [transformer_model, pooling_model]

# -----------------------------
# Retriever class
# -----------------------------
class Retriever:
    """Retriever for a single corpus using FAISS and SentenceTransformer."""
    def __init__(self, chunk_dir="./corpus/FreeCorpus/chunk", HNSW=False):
        self.chunk_dir = os.path.normpath(chunk_dir)
        self.index_dir = os.path.join(os.path.dirname(self.chunk_dir), "index")
        os.makedirs(self.index_dir, exist_ok=True)

        # Initialize embedding model
        self.model = CustomizeSentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.model.eval()

        # Load or build FAISS index
        index_path = os.path.join(self.index_dir, "faiss.index")
        metadata_path = os.path.join(self.index_dir, "metadatas.jsonl")

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadatas = [json.loads(line) for line in f]
        else:
            self.build_index(HNSW)

    # -----------------------------
    # Build FAISS index
    # -----------------------------
    def build_index(self, HNSW=False):
        fnames = sorted(f for f in os.listdir(self.chunk_dir) if f.endswith(".jsonl"))
        embeddings_list = []
        self.metadatas = []

        for fname in tqdm.tqdm(fnames, desc="Building embeddings"):
            path = os.path.join(self.chunk_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
            texts = [json.loads(line) for line in lines]
            inputs = [concat(t["title"], t["content"]) for t in texts]
            embeddings = self.model.encode(inputs, convert_to_numpy=True)
            embeddings_list.append(embeddings)
            self.metadatas.extend([{"id": t["id"], "source": fname, "title": t["title"], "content": t["content"]} for t in texts])

        all_embeddings = np.vstack(embeddings_list).astype(np.float32)
        dim = all_embeddings.shape[1]

        # Create FAISS index
        if HNSW:
            self.index = faiss.IndexHNSWFlat(dim, 32)
        else:
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(all_embeddings)

        # Save index and metadata
        faiss.write_index(self.index, os.path.join(self.index_dir, "faiss.index"))
        with open(os.path.join(self.index_dir, "metadatas.jsonl"), 'w', encoding='utf-8') as f:
            for m in self.metadatas:
                f.write(json.dumps(m) + "\n")

    # -----------------------------
    # Retrieve top-k relevant documents
    # -----------------------------
    def get_relevant_documents(self, question, k=5):
        q_emb = self.model.encode([question], convert_to_numpy=True).astype(np.float32)
        scores, idxs = self.index.search(q_emb, k)
        results = []
        for i in idxs[0]:
            if i >= len(self.metadatas):
                continue
            doc = self.metadatas[i]
            results.append({"title": doc["title"], "content": doc["content"]})
        return results