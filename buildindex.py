from src.utils import Retriever

print("🔄 Building FAISS index (one-time)...")

retriever = Retriever(chunk_dir="./corpus/openfda/chunk")

print("✅ Index ready. You can now run Streamlit.")
