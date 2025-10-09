import os
import re
import json
import tiktoken
from .utils import Retriever
from .template import *

class MedRAG:
    def __init__(
        self,
        llm_name="gemini-2.5-flash",
        rag=True,
        corpus_name="FreeCorpus",
        db_dir="./corpus",
        cache_dir=None,
        corpus_cache=False,
        HNSW=False
    ):
        self.llm_name = llm_name
        self.rag = rag
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None

        # -----------------------------
        # Setup retriever
        # -----------------------------
        if rag:
            chunk_dir = os.path.join(db_dir, corpus_name, "chunk")
            if os.path.exists(chunk_dir):
                self.retrieval_system = Retriever(chunk_dir=chunk_dir)
            else:
                self.retrieval_system = None
        else:
            self.retrieval_system = None

        # -----------------------------
        # Templates for plain English answers
        # -----------------------------
        self.templates = {
            "medrag_system": simple_medrag_system,
            "medrag_prompt": simple_medrag_prompt
        }

        # -----------------------------
        # Gemini setup
        # -----------------------------
        self.model = None
        try:
            import google.generativeai as genai
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(
                    model_name=self.llm_name.split("/")[-1],
                    generation_config={"temperature": 0, "max_output_tokens": 2048}
                )
        except Exception:
            self.model = None  # fallback if Gemini not available

        # -----------------------------
        # Tokenizer
        # -----------------------------
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        if "1.5" in self.llm_name.lower():
            self.max_length = 1048576
            self.context_length = 1040384
        else:
            self.max_length = 30720
            self.context_length = 28672

    # -----------------------------
    # Gemini generation
    # -----------------------------
    def generate(self, messages, **kwargs):
        if not self.model:
            # fallback to returning user message if Gemini not available
            return messages[1]["content"]
        user_input = messages[0]["content"] + "\n\n" + messages[1]["content"]
        response = self.model.generate_content(user_input, **kwargs)
        return response.candidates[0].content.parts[0].text

    # -----------------------------
    # Main MedRAG flow
    # -----------------------------
    import os
import re
import json
import tiktoken
from .utils import Retriever
from .template import *

class MedRAG:
    def __init__(
        self,
        llm_name="gemini-2.5-flash",
        rag=True,
        corpus_name="FreeCorpus",
        db_dir="./corpus",
        cache_dir=None,
        corpus_cache=False,
        HNSW=False
    ):
        self.llm_name = llm_name
        self.rag = rag
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None

        # -----------------------------
        # Setup retriever
        # -----------------------------
        if rag:
            chunk_dir = os.path.join(db_dir, corpus_name, "chunk")
            if os.path.exists(chunk_dir):
                self.retrieval_system = Retriever(chunk_dir=chunk_dir)
            else:
                self.retrieval_system = None
        else:
            self.retrieval_system = None

        # -----------------------------
        # Templates for plain English answers
        # -----------------------------
        self.templates = {
            "medrag_system": simple_medrag_system,
            "medrag_prompt": simple_medrag_prompt
        }

        # -----------------------------
        # Gemini setup
        # -----------------------------
        self.model = None
        try:
            import google.generativeai as genai
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(
                    model_name=self.llm_name.split("/")[-1],
                    generation_config={"temperature": 0, "max_output_tokens": 2048}
                )
        except Exception:
            self.model = None  # fallback if Gemini not available

        # -----------------------------
        # Tokenizer
        # -----------------------------
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        if "1.5" in self.llm_name.lower():
            self.max_length = 1048576
            self.context_length = 1040384
        else:
            self.max_length = 30720
            self.context_length = 28672

    # -----------------------------
    # Gemini generation
    # -----------------------------
    def generate(self, messages, **kwargs):
        if not self.model:
            # fallback to returning user message if Gemini not available
            return messages[1]["content"]
        user_input = messages[0]["content"] + "\n\n" + messages[1]["content"]
        response = self.model.generate_content(user_input, **kwargs)
        return response.candidates[0].content.parts[0].text

    # -----------------------------
    # Main MedRAG flow
    # -----------------------------
    def medrag_answer(self, question, k=5):
    # -----------------------------
    # Skip retrieval if k <= 0
    # -----------------------------
        retrieved_snippets = []
        context_text = ""

        if self.rag and self.retrieval_system and k > 0:
            retrieved_snippets = self.retrieval_system.get_relevant_documents(question, k=k)
            context_text = "\n".join([f"{doc['title']}: {doc['content']}" for doc in retrieved_snippets])

    # -----------------------------
    # Build prompt and generate
    # -----------------------------
        prompt_medrag = self.templates["medrag_prompt"].render(
            context=context_text, question=question
        )
        messages = [
            {"role": "system", "content": self.templates["medrag_system"]},
            {"role": "user", "content": prompt_medrag}
        ]
        ans = self.generate(messages)
        ans = re.sub(r"\s+", " ", ans).strip()

    # -----------------------------
    # Clean answer: remove ** and fallback text
    # -----------------------------
        if ans:
            ans = re.sub(r"\*\*(.*?)\*\*", r"\1", ans)
            ans = re.sub(r'^While no specific documents were provided,?\s*', '', ans)
            ans = ans.strip(" :;")
        return ans