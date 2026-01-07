import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# ---------- CONFIG ----------
genai.configure(api_key="paste_your_API_KEY")
model = genai.GenerativeModel("gemini-3-flash-preview")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- LOAD PDF ----------
def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

pdf_text = load_pdf_text(r"your_rag_file_path as pdf")
chunks = pdf_text.split("\n\n")

# ---------- EMBEDDINGS + VECTOR STORE ----------
embeddings = embedder.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def retrieve_context(query):
    q_emb = embedder.encode([query])
    distances, ids = index.search(q_emb, k=3)
    results = [chunks[i] for i in ids[0]]
    return "\n".join(results)

# ---------- STREAMLIT APP ----------
st.title("Event Management Planning & Operations – RAG Bot")

user_query = st.text_input("Ask something about event planning")

if st.button("Get Answer") and user_query:
    context = retrieve_context(user_query)
    prompt = f"""
You are an informational event planning assistant.
Answer only based on the context below.
Do not book venues or perform actions.

Context:
{context}

Question:
{user_query}
"""

    response = model.generate_content(prompt)
    st.write(response.text)
