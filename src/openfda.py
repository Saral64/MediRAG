import os
import json
import regex as re
from pathlib import Path
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

def load_drug_data(file_path):
    """Loads OpenFDA JSON and extracts relevant medical fields."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    processed_docs = []
    for i, rec in enumerate(data.get("results", [])):
        openfda = rec.get("openfda", {})
        generic = openfda.get("generic_name", [None])[0]
        brand = openfda.get("brand_name", [None])[0]
        title = generic if generic else (brand if brand else "Unknown Medication")

        sections = {
            "Indications": rec.get("indications_and_usage", []),
            "Dosage": rec.get("dosage_and_administration", []),
            "Warnings": rec.get("warnings", []),
            "Adverse Reactions": rec.get("adverse_reactions", []),
            "Drug Interactions": rec.get("drug_interactions", [])
        }

        full_text_list = []
        for section_name, content in sections.items():
            if content:
                clean_section = " ".join(content)
                full_text_list.append(f"[{section_name}]: {clean_section}")

        combined_text = " ".join(full_text_list)
        combined_text = re.sub(r"\s+", " ", combined_text).strip()

        if combined_text:
            processed_docs.append({
                "id": str(i),
                "title": title,
                "text": combined_text
            })
    
    return processed_docs

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # --- PATH SETUP ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent 
    
    INPUT_FILE = PROJECT_ROOT / "data" / "drug-label-0001-of-0013.json"
    chunk_dir = PROJECT_ROOT / "corpus" / "openfda" / "chunk"
    
    if not INPUT_FILE.exists():
        print(f"❌ Could not find data file at: {INPUT_FILE}")
        exit(1)

    print(f"📂 Loading data from: {INPUT_FILE.name}...")
    mini_corpus = load_drug_data(INPUT_FILE)

    # FIXED: Using the variable you already defined
    chunk_dir.mkdir(parents=True, exist_ok=True)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    saved_text = []

    print(f"✂️  Chunking {len(mini_corpus)} drug records...")

    for doc in mini_corpus:
        chunks = text_splitter.split_text(doc['text'])
        for j, chunk in enumerate(chunks):
            clean_chunk = re.sub(r"\s+", " ", chunk)
            saved_text.append(json.dumps({
                "id": f"{doc['id']}_{j}",
                "title": doc['title'],
                "content": clean_chunk,
                "contents": concat(doc['title'], clean_chunk)
            }, ensure_ascii=False)) # ensure_ascii=False handles special medical symbols better

    output_file = chunk_dir / "clean_openfda.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(saved_text))

    print(f"✅ Curated drug corpus generated: {output_file}")
