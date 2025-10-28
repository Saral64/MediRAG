import os
import streamlit as st
from src.medrag import MedRAG

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="MediRAG - AI Medical Assistant",
    page_icon="🩺",
    layout="centered"
)

# ---------- TITLE ----------
st.title("🩺 MediRAG — Your AI Medical Assistant")
st.markdown("Ask any medical question and get an instant answer powered by RAG and Gemini.")

# ---------- INITIALIZE MODEL ----------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load and cache the MedRAG model once."""
    return MedRAG(
        llm_name="gemini-2.5-flash",
        rag=True,
        corpus_name="wikipedia"
    )

model = load_model()

# ---------- USER INPUT ----------
question = st.text_area(
    "💬 Enter your medical question:",
    placeholder="e.g. What are the early symptoms of depression?"
)

# ---------- SUBMIT ----------
if st.button("Get Answer"):
    if not question.strip():
        st.warning("⚠️ Please enter a question first.")
    else:
        with st.spinner("Thinking... please wait ⏳"):
            try:
                answer = model.medrag_answer(question, k=0)
                answer_lines = answer.strip().splitlines()

                # Display formatted answer
                st.success("✅ Answer Found:")
                st.markdown(f"### 🧾 Response:\n{answer_lines[0] if answer_lines else answer}")
                
                if len(answer_lines) > 1:
                    st.markdown("<br>".join(answer_lines[1:]), unsafe_allow_html=True)
            except Exception as e:
                st.error("⚠️ Error generating the answer.")
                st.exception(e)
