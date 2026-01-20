import os
import re
import tiktoken
from .utils import Retriever
from .template import (
    simple_medrag_system,
    simple_medrag_prompt,
    ddi_medrag_prompt
)

class MedRAG:
    def __init__(self, llm_name="gemini-2.5-flash", rag=True, corpus_name="openfda", db_dir="./corpus"):
        self.llm_name = llm_name
        self.rag = rag
        self.retrieval_system = None
        if rag:
            chunk_dir = os.path.join(db_dir, corpus_name, "chunk")
            if os.path.exists(chunk_dir):
                self.retrieval_system = Retriever(chunk_dir=chunk_dir)

        self.templates = {
            "system": simple_medrag_system,
            "prompt": simple_medrag_prompt
        }

        self.model = None
        try:
            import google.generativeai as genai
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key: raise RuntimeError("GOOGLE_API_KEY not set")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.llm_name)
        except Exception as e:
            print("❌ Gemini init failed:", e)

    def generate(self, messages):
        if not self.model: raise RuntimeError("Gemini model not initialized")
        # Combine system and user prompt for Gemini
        combined_prompt = f"SYSTEM: {messages[0]['content']}\n\nUSER: {messages[1]['content']}"
        response = self.model.generate_content(combined_prompt)
        return response.text if response.candidates else "No response generated."

    def _key_terms(self, text):
        fillers = {
        "understand", "know", "mean", "think", "explain", "tell", 
        "describe", "give", "information", "detail", "details"
        }
        stopwords = {"what", "is", "are", "does", "do", "you", "by","can", "the", "a", "an", "with", "if", "when", "how", "happens", "interacts", "patient", "patients", "treatment", "use", "used", "indicated", "mg", "dose", "clinical"}
        all_stops = fillers | stopwords
        return {
                w.lower() for w in re.findall(r"[a-zA-Z]+", text)
                if w.lower() not in all_stops and len(w) > 2
        }
    
    def _extract_medical_concepts(self, question):
         prompt = f"""
                   Extract only medical concepts from the query.
                   Remove intent, tone, and explanation-related words.
                   Return a comma-separated list. No explanation.

                   Query:
                   {question}
                """
         try:
           response = self.model.generate_content(prompt)
           text = response.text.strip()
        # safety: fallback to original question if extraction fails
           return text if text else question
         except Exception:
           return question


    def _evidence_strength(self, docs, question):
        if not docs: return "LOW"

        q_terms = self._key_terms(question)
    
    # If there are 2 or fewer key terms, the evidence must be 
    # extremely strong to be anything other than LOW.
        if len(q_terms) <= 2:
           top_score = docs[0].get("score", 0.0) if docs else 0
           if top_score < 0.650: 
              return "LOW"
           
        scores = [d.get("score", 0.0) for d in docs]
        avg_top_score = sum(scores[:3]) / min(len(scores), 3)
        q_terms = self._key_terms(question)
        if not q_terms: return "LOW"

        supporting_docs = 0
        for d in docs:
            overlap = q_terms & self._key_terms(d["content"])
            if len(overlap) / len(q_terms) >= 0.4:
                supporting_docs += 1

        if avg_top_score >= 0.70 and supporting_docs >= 3: return "HIGH"
        if avg_top_score >= 0.55 and supporting_docs >= 1: return "MEDIUM"
        return "LOW"

    def _is_ddi_query(self, question, docs, concept_query=None):
        intent_words = {"interact", "interaction", "interacts", "combined", "combine", "combines", "react", "reaction", "reacts", "reacted", "combination", "together", "co-administered", "concurrent", "versus", "vs"}
        has_intent = any(w in question.lower() for w in intent_words)
        
        # Count unique drugs mentioned in the question that appear in our retrieved titles
        detected_drugs = set()
        q_lower = f"{question} {concept_query}".lower() if concept_query else question.lower()
        for d in docs:
            title = d["title"].lower()
            # If the drug title (e.g. "Potassium Phosphate") is in the question
            if title in q_lower:
                detected_drugs.add(title)
        
        return has_intent or len(detected_drugs) >= 2
    
    def _extract_severity(self, text):
        """Extracts the numeric score from the LLM response."""
        match = re.search(r"SEVERITY_SCORE:\s*(\d+)", text)
        if match:
            score = int(match.group(1))
            # Clean the tag out of the final text so the user doesn't see raw code
            clean_text = re.sub(r"SEVERITY_SCORE:\s*\d+", "", text).strip()
            return clean_text, score
        return text, None

    def medrag_answer(self, question, k=5):
        context_text = ""
        evidence = "LOW"
        docs = []

        concept_query = question
        if self.retrieval_system and k > 0:
            # Increase K slightly for DDI to get both drug labels
            ddi_keywords = ["interact", "combine", "interacts", "combines", "combined","reacts", "react"]
            search_k = k + 3 if any(kw in question.lower() for kw in ddi_keywords) else k
            concept_query = self._extract_medical_concepts(question)
            docs = self.retrieval_system.get_relevant_documents(concept_query, search_k)
            context_text = "\n".join(f"SOURCE [{d['title']}]: {d['content']}" for d in docs)
            evidence = self._evidence_strength(docs, concept_query)

        ddi_mode = self._is_ddi_query(question, docs, concept_query)

        if ddi_mode:
            # Override system prompt for DDI to ensure structure
            sys_msg = "You are a Clinical Decision Support System. Output ONLY the requested sections."
            prompt = ddi_medrag_prompt.render(
                context=context_text,
                question=question,
                evidence_level=evidence # Pass evidence level to template
            )
        else:
            sys_msg = self.templates["system"]
            prompt = self.templates["prompt"].render(
                context=context_text,
                question=question,
                evidence_level=evidence # Pass evidence level to template
            )

        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]
        ans = self.generate(messages).strip()

        ans, severity_score = self._extract_severity(ans)

        if evidence == "LOW" and not ans.startswith("Note:"):
           ans = f"Note: The following information is based on general medical knowledge as it is not fully detailed in the provided corpus.\n\n{ans}"
        
        severity_display = ""
        if severity_score is not None:
            # Assign a visual label based on the score
            if severity_score >= 8:
                severity_display = f"🚨 **Severity Score: {severity_score}/10 (High Risk)**"
            elif severity_score >= 4:
                severity_display = f"⚠️ **Severity Score: {severity_score}/10 (Moderate Risk)**"
            else:
                severity_display = f"✅ **Severity Score: {severity_score}/10 (Low Risk)**"

        # Final Formatting for Clinical Look
        if ddi_mode:
            header = "## CLINICAL INTERACTION REPORT\n"
            if severity_display:
                header += f"{severity_display}\n\n"
            ans = f"{header}{ans}"
        
        ans += f"\n\n---\n**Evidence Strength:** {evidence} | **Analysis Mode:** {'Drug-Drug Interaction' if ddi_mode else 'General Clinical'}"
        return re.sub(r"\n{3,}", "\n\n", ans).strip()
