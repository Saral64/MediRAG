import os
import streamlit as st
from src.medrag import MedRAG

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="MediRAG | AI Medical Assistant",
    page_icon="🩺",
    layout="wide", # Wider layout for a more professional feel
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM STYLING (Dark Theme Focus) ----------
st.markdown("""
    <style>
    /* Main background and text */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Custom Card Style */
    .med-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    /* Gradient Title */
    .main-title {
        background: -webkit-linear-gradient(#60A5FA, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------- SIDEBAR ----------
# ---------- SIDEBAR ----------
with st.sidebar:
    # Use a large emoji instead of a URL to avoid loading issues
    st.markdown("<h1 style='text-align: center; font-size: 60px; margin-bottom: 0;'>🩺</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-top: 0;'>MediRAG</h3>", unsafe_allow_html=True)
    st.divider()
    
    # Using a container for the configuration box
    with st.container():
        st.markdown("### ⚙️ Configuration")
        st.info(f"""
        - **System:** RAG Enabled
        - **Corpus:** Wikipedia
        - **Model:** Gemini-2.5-Flash
        """)
    
    st.divider()
    st.warning("⚠️ **Disclaimer:** This tool is for informational purposes only. Consult a professional for medical advice.")

# ---------- INITIALIZE MODEL ----------
@st.cache_resource(show_spinner=False)
def load_model():
    return MedRAG(
        llm_name="gemini-2.5-flash",
        rag=True,
        corpus_name="wikipedia"
    )

model = load_model()

# ---------- HEADER ----------
st.markdown('<h1 class="main-title">MediRAG</h1>', unsafe_allow_html=True)
st.markdown("##### *Advanced Clinical Reasoning & Knowledge Retrieval*")
st.divider()

# ---------- CHAT INTERFACE ----------
# Using chat_input for a modern "ChatGPT-style" feel
question = st.chat_input("Ask a clinical or medical question...")

if question:
    # Display User Question
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # Display Assistant Response
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Analyzing medical literature..."):
            try:
                # k=0 is passed as per your original code
                answer = model.medrag_answer(question, k=0)
                
                # Split answer into summary and details if applicable
                st.markdown("### Clinical Response")
                st.markdown(answer)
                
                # Optional: Add a 'Disclaimer' footer to every response
                st.caption("Sources used: Wikipedia Medical Corpus. Response generated via RAG.")
                
            except Exception as e:
                st.error("An error occurred while processing the request.")
                with st.expander("Technical Error Details"):
                    st.exception(e)

# ---------- EMPTY STATE ----------
if not question:
    st.markdown("""
    <div style="text-align: center; padding: 50px; opacity: 0.5;">
        <p>Enter a query below to start the medical analysis.</p>
    </div>
    """, unsafe_allow_html=True)