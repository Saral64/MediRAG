---

# MedRAG

**MedRAG** (Medical Retrieval-Augmented Generation) is a lightweight medical question-answering system built using Google Gemini and a custom retrieval mechanism. It can answer medical questions using a local mini-corpus or generate answers directly via Gemini when relevant documents are not found.

---

## Features

* ✅ Plain-English answers: Outputs concise, human-readable answers without JSON formatting.
* ✅ Fallback to Gemini: Automatically generates answers if no relevant documents are found in the corpus.
* ✅ Mini-corpus support: Ships with a small Wikipedia-based corpus for testing.
* ✅ Optional retrieval: Can retrieve relevant documents from a local corpus using embeddings (FAISS + SentenceTransformer).
* ✅ Lightweight & easy to extend: Add more documents or fine-tune the system easily.

---

## Installation

1. Clone the repository:

git clone https://github.com/Saral64/MediRAG.git
cd MedRAG

2. Install dependencies:

pip install -r requirements.txt

3. Set your Google Gemini API key:

For Linux/macOS:
export GOOGLE_API_KEY="your_api_key"

For Windows (PowerShell):
setx GOOGLE_API_KEY "your_api_key"

---

## Folder Structure

MedRAG/
├─ src/
│  ├─ medrag.py           # Main MedRAG class
│  ├─ utils.py            # Retriever and embedding utilities
│  └─ template.py         # Templates for prompts
├─ corpus/
│  └─ wikipedia/          # Mini Wikipedia corpus
├─ main.py                # Example usage
└─ README.md

---

## Usage

from src.medrag import MedRAG

# Initialize MedRAG

model = MedRAG(
llm_name="gemini-2.5-flash",
rag=True,
corpus_name="wikipedia"
)

# Ask a medical question

question = "What are the early symptoms of Diabetes?"
answer = model.medrag_answer(question)

print("Answer:")
print(answer)

---

## Example Output

Answer:
Early symptoms of diabetes include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, blurred vision, and slow-healing sores.

---

## Notes

* The retrieval mechanism uses **FAISS** and **SentenceTransformer** for document embeddings.
* The mini-corpus is small (~3 articles) and intended for testing only. You can add more documents in `corpus/wikipedia/chunk`.
* Gemini API is optional; without it, MedRAG will fallback to using only the local corpus.

---

## License

MIT License

---
