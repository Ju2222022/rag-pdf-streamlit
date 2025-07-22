import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
import torch
from tempfile import NamedTemporaryFile
from sentence_transformers import SentenceTransformer

# === CONFIG ===
EMBED_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL)
model.to(torch.device("cpu"))  # 🧠 Forcer l'utilisation du CPU

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

st.title("📘 Assistant conformité (multi-PDF + IA)")
st.write("Chargez un ou plusieurs fichiers PDF réglementaires, posez une question, et obtenez une réponse basée sur leur contenu.")

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

    question = st.text_input("❓ Posez votre question liée aux documents :")

    if question:
        question_vec = model.encode([question])
        D, I = index.search(np.array(question_vec).astype("float32"), k=3)
        context = "\n---\n".join([all_chunks[i] for i in I[0]])
        
        st.markdown("### 🧠 Contexte extrait :")
        st.write(context)

        st.markdown("⚠️ Cette version ne génère pas encore de réponse résumée avec un LLM. Souhaitez-vous qu’on l’ajoute ?")

import json

# === Sauvegarde de l’index FAISS ===
faiss.write_index(index, "index.faiss")
st.success("💾 Index FAISS sauvegardé sous 'index.faiss'.")

# === Sauvegarde des textes correspondants ===
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
st.success("💾 Texte sauvegardé sous 'chunks.json'.")


