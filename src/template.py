from liquid import Template

# -----------------------------
# Simple MedRAG templates
# -----------------------------

# Simple system prompt when using RAG
simple_medrag_system = '''You are a helpful medical expert, and your task is to answer a medical question using the relevant documents if available. Provide a concise, plain English answer.'''

# Prompt template for questions using retrieved documents
simple_medrag_prompt = Template('''
Here are the relevant documents (if any):
{{context}}

Question:
{{question}}
''')

# Simple system prompt when not using documents (direct LLM answer)
i_medrag_system = '''You are a helpful medical assistant, and your task is to answer the given question in plain English.'''

# Prompt template for direct question answering
i_medrag_prompt = Template('''
Question:
{{question}}
''')

# Optional follow-up instructions (if needed)
follow_up_instruction_ask = '''Analyze all the provided information and generate {} concise, context-specific questions to search for additional information. Each query should be simple and focused.'''

follow_up_instruction_answer = '''Analyze all the provided information step-by-step, then provide a concise, plain English answer.'''