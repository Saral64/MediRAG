import os
import json
import regex as re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------
# Helper functions
# -----------------------------
def ends_with_ending_punctuation(s: str) -> bool:
    return s.strip().endswith(('.', '?', '!'))

def concat(title: str, content: str) -> str:
    """Combine title and content, adding a period if needed."""
    title = title.strip()
    content = content.strip()
    if ends_with_ending_punctuation(title):
        return f"{title} {content}"
    else:
        return f"{title}. {content}"

# -----------------------------
# Create mini Wikipedia corpus
# -----------------------------
if __name__ == "__main__":
    mini_corpus = [
        {
            "id": "1",
            "title": "Diabetes",
            "text": (
                "Diabetes mellitus is a group of metabolic disorders characterized by high blood sugar levels over a prolonged period. "
                "Common symptoms include frequent urination, increased thirst, and increased hunger. "
                "If untreated, diabetes can cause many complications."
            )
        },
        {
            "id": "2",
            "title": "Hypertension",
            "text": (
                "Hypertension, also known as high blood pressure, is a long-term medical condition in which the blood pressure in the arteries is persistently elevated. "
                "Symptoms may include headaches, shortness of breath, or nosebleeds. "
                "Long-term hypertension can lead to heart disease and stroke."
            )
        },
        {
            "id": "3",
            "title": "Insulin",
            "text": (
                "Insulin is a peptide hormone produced by beta cells of the pancreatic islets. "
                "It regulates the metabolism of carbohydrates, fats, and protein by promoting the absorption of glucose from the blood into liver, fat, and skeletal muscle cells."
            )
        }
    ]

    # Chunk directory
    chunk_dir = os.path.join("corpus", "wikipedia", "chunk")
    os.makedirs(chunk_dir, exist_ok=True)

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    saved_text = []

    for doc in mini_corpus:
        chunks = text_splitter.split_text(doc['text'])
        for j, chunk in enumerate(chunks):
            saved_text.append(json.dumps({
                "id": f"{doc['id']}_{j}",
                "title": doc['title'],
                "content": re.sub(r"\s+", " ", chunk),
                "contents": concat(doc['title'], re.sub(r"\s+", " ", chunk))
            }))

    # Save as JSONL
    output_file = os.path.join(chunk_dir, "mini_wikipedia.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(saved_text))

    print(f"✅ Mini corpus generated: {output_file}")