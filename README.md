# DrugIntelRAG 🩺

DrugIntelRAG is a **clinical-grade Retrieval-Augmented Generation (RAG) system** designed for medical question answering with explicit **evidence strength estimation** and **drug–drug interaction (DDI) analysis**. It combines a curated OpenFDA corpus, FAISS-based semantic retrieval, and Google's Gemini models to produce structured, safety-aware medical responses.

> ⚠️ **Disclaimer**: DrugIntelRAG is for research and informational purposes only. It is not a substitute for professional medical advice.

---

## Key Capabilities

* **Medical RAG Pipeline**: Grounds answers strictly in an OpenFDA-derived corpus before falling back to general medical knowledge.
* **Evidence Strength Scoring**: Automatically labels responses as **LOW / MEDIUM / HIGH** based on corpus coverage and semantic support.
* **Drug–Drug Interaction Mode**: Detects interaction intent and generates a structured clinical interaction report with severity scoring.
* **Severity Assessment**: Produces a numeric interaction severity score (1–10) with risk stratification.
* **Clinical Formatting**: Bullet-driven, sectioned outputs suitable for decision-support use cases.
* **Streamlit UI**: Interactive, ChatGPT-style medical interface.

---

## System Architecture

```
User Query
   │
   ▼
Medical Concept Extraction
   │
   ▼
FAISS Retriever (Sentence-Transformers)
   │
   ▼
Evidence Strength Estimator
   │
   ├── DDI Detected → DDI Prompt + Severity Scoring
   │
   └── General Clinical Prompt
   │
   ▼
Gemini LLM (Grounded Generation)
   │
   ▼
Structured Clinical Response
```

---

## Repository Structure

```
DrugIntelRAG/
├── corpus/
│   └── openfda/
│       ├── chunk/              # Chunked OpenFDA corpus (JSONL)
│       └── index/              # FAISS index + metadata
├── data/                       # Raw OpenFDA drug label JSON (LFS)
├── src/
│   ├── medrag.py               # Core RAG + reasoning engine
│   ├── openfda.py              # OpenFDA ingestion & chunking
│   ├── template.py             # Prompt templates (general + DDI)
│   └── utils.py                # FAISS retriever & embeddings
├── buildindex.py               # One-time FAISS index builder
├── main.py                     # Streamlit application
├── requirements.txt
└── README.md
```

---

## Evidence Strength Logic

Evidence strength reflects **corpus coverage**, not clinical importance.

* **LOW**

  * Sparse or missing corpus support
  * Low semantic similarity
  * Model may fall back to general medical knowledge (explicitly disclosed)

* **MEDIUM**

  * Partial corpus overlap
  * At least one supporting document

* **HIGH**

  * Strong semantic scores
  * Multiple documents with significant keyword overlap

This logic is implemented in `MedRAG._evidence_strength()` using:

* Top-k FAISS similarity scores
* Keyword overlap ratio between query and retrieved chunks

---

## Drug–Drug Interaction (DDI) Detection

A query is routed to **DDI mode** if:

* Interaction intent keywords are detected *(interact, combine, vs, co-administered, etc.)*, **or**
* Two or more distinct drug entities are identified in retrieved titles

DDI responses follow a **strict clinical structure**:

* Interaction Summary
* Mechanism
* Clinical Risk
* Monitoring Recommendation
* Overall Assessment
* **Severity Score (1–10)**

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/your-username/DrugIntelRAG.git
cd DrugIntelRAG
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Gemini API Key

**Windows (PowerShell):**

```bash
setx GOOGLE_API_KEY "YOUR_API_KEY"
```

**Linux / macOS:**

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

---

## Corpus Preparation

### Step 1: Generate OpenFDA Chunks

```bash
python src/openfda.py
```

This:

* Extracts key medical sections (indications, dosage, warnings, interactions)
* Cleans and chunks text
* Saves a JSONL corpus under `corpus/openfda/chunk/`

### Step 2: Build FAISS Index (One-Time)

```bash
python buildindex.py
```

---

## Running the Application

```bash
streamlit run main.py
```

---

## Example Queries

* "Explain the drug interactions between atenolol and chlorthalidone"
* "What are the adverse effects of metformin?"
* "Is combining drug A and drug B safe?"

---

## Design Principles

* **Grounded First**: Corpus evidence is always prioritized over LLM knowledge
* **Transparency**: Evidence strength is always shown
* **Clinical Safety**: Structured outputs, disclaimers, and severity scoring
* **Extensibility**: Corpus, retriever, and scoring logic are modular

---

## MIT License

Copyright <YEAR> <COPYRIGHT HOLDER>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
