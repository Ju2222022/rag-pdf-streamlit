import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from tempfile import NamedTemporaryFile

# === CONFIG ===
EMBED_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL)

def pdf_to_chunks(pdf_file, chunk_size=500):
    doc = fitz.open(pdf_file)
    full_text = "\n".join([page.get_text() for page in doc])
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

def embed_chunks(chunks):
    return model.encode(chunks)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

st.title("📘 Assistant conformité (PDF + IA)")
st.write("Chargez un PDF réglementaire, posez une question, et obtenez une réponse basée sur le contenu.")

uploaded_files = st.file_uploader("📤 Chargez un ou plusieurs fichiers PDF", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []

    for f in uploaded_files:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

        st.info(f"📄 Lecture de : {f.name}")
        chunks = pdf_to_chunks(tmp_path)
        all_chunks.extend(chunks)

    st.success(f"✅ {len(uploaded_files)} fichiers lus. {len(all_chunks)} morceaux extraits.")

    embeddings = embed_chunks(all_chunks)
    index = build_faiss_index(np.array(embeddings))

    embeddings = embed_chunks(chunks)
    index = build_faiss_index(np.array(embeddings))

    st.success(f"✅ Index de {len(chunks)} morceaux construit.")

    question = st.text_input("❓ Posez votre question liée au PDF :")

    if question:
        question_vec = model.encode([question])
        D, I = index.search(np.array(question_vec).astype("float32"), k=3)
        context = "\n---\n".join([all_chunks[i] for i in I[0]])
        
        st.markdown("### 🧠 Contexte extrait :")
        st.write(context)

        st.markdown("⚠️ Cette version ne génère pas encore de réponse résumée (LLM). Tu veux qu’on l’ajoute ?")
