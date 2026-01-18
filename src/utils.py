import os
import json
import numpy as np
import faiss
import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling


# -----------------------------
# Helper functions
# -----------------------------
def ends_with_ending_punctuation(s: str) -> bool:
    return s.endswith(('.', '?', '!'))


def concat(title: str, content: str) -> str:
    title = title.strip()
    content = content.strip()
    if not ends_with_ending_punctuation(title):
        title += "."
    return f"{title} {content}"


# -----------------------------
# Custom SentenceTransformer (CLS pooling)
# -----------------------------
class CustomizeSentenceTransformer(SentenceTransformer):
    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(
            transformer_model.get_word_embedding_dimension(),
            pooling_mode_cls_token=True
        )
        return [transformer_model, pooling_model]


# -----------------------------
# Retriever
# -----------------------------
class Retriever:
    def __init__(self, chunk_dir="./corpus/openfda/chunk", HNSW=False):
        self.chunk_dir = os.path.normpath(chunk_dir)
        self.index_dir = os.path.join(os.path.dirname(self.chunk_dir), "index")
        os.makedirs(self.index_dir, exist_ok=True)

        # Embedding model
        self.model = CustomizeSentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        self.model.eval()

        self.index_path = os.path.join(self.index_dir, "faiss.index")
        self.meta_path = os.path.join(self.index_dir, "metadatas.jsonl")

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load_index()
        else:
            self._build_index(HNSW)

    # -----------------------------
    # Load existing index
    # -----------------------------
    def _load_index(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]

    # -----------------------------
    # Build FAISS index (one-time)
    # -----------------------------
    def _build_index(self, HNSW=False):
        fnames = sorted(f for f in os.listdir(self.chunk_dir) if f.endswith(".jsonl"))
        embeddings_list = []
        self.metadatas = []

        for fname in tqdm.tqdm(fnames, desc="Building embeddings"):
            path = os.path.join(self.chunk_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")

            texts = [json.loads(line) for line in lines]
            inputs = [concat(t["title"], t["content"]) for t in texts]

            print(f"Encoding {len(inputs)} chunks from {fname}")

            batch_size = 64
            all_embeds = []

            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                emb = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                all_embeds.append(emb)

            embeddings = np.vstack(all_embeds)
            embeddings_list.append(embeddings)

            self.metadatas.extend([
                {
                    "id": t["id"],
                    "source": fname,
                    "title": t["title"],
                    "content": t["content"]
                }
                for t in texts
            ])

        # Stack + normalize
        all_embeddings = np.vstack(embeddings_list).astype(np.float32)
        faiss.normalize_L2(all_embeddings)

        dim = all_embeddings.shape[1]

        # FAISS index
        if HNSW:
            self.index = faiss.IndexHNSWFlat(dim, 32)
        else:
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(all_embeddings)

        # Save
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self.metadatas:
                f.write(json.dumps(m) + "\n")

    # -----------------------------
    # Retrieval
    # -----------------------------
    def get_relevant_documents(self, question, k=5):
        q_emb = self.model.encode(
        [question],
        convert_to_numpy=True
        ).astype(np.float32)

        faiss.normalize_L2(q_emb)

        scores, idxs = self.index.search(q_emb, k)

        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i >= len(self.metadatas):
               continue
            doc = self.metadatas[i]
            results.append({
                "title": doc["title"],
                "content": doc["content"],
                "score": float(score)
        })

        return results
