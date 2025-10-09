import os
from src.medrag import MedRAG

# Initialize MedRAG
model = MedRAG(
    llm_name="gemini-2.5-flash",
    rag=True,
    corpus_name="wikipedia"
)

question = "What are the early symptoms of Depression?"

# Generate answer WITHOUT retrieval
answer = model.medrag_answer(question, k=0)
print("\n=== ANSWER ===")
answer_lines = answer.strip().splitlines()
print("\n".join(answer_lines[:3]))