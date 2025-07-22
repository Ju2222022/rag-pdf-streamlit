import os
import fitz  # PyMuPDF
import faiss
import json
import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
EMBED_MODEL = "all-MiniLM-L6-v2"
PDF_FOLDER = "pdf_EU_elec_regulations"

# Forcer CPU (important pour Streamlit Cloud)
model = SentenceTransformer(EMBED_MODEL)
model.to(torch.device("cpu"))

# === FONCTIONS ===
def pdf_to_chunks(filepath, chunk_size=500):
    doc = fitz.open(filepath)
    full_text = "\n".join([page.get_text() for page in doc])
    return [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

def embed_chunks(chunks):
    return model.encode(chunks)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

# === INTERFACE STREAMLIT ===
st.title("📘 Génération de l'index RAG à partir de PDF")
st.write("Sélectionnez jusqu’à 3 fichiers PDF réglementaires à intégrer dans un index vectoriel.")

# 📂 Liste les fichiers PDF du dossier
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

selected_files = st.multiselect(
    "📄 Choisissez jusqu’à 3 fichiers PDF",
    options=pdf_files,
    default=pdf_files[:1]
)

if len(selected_files) > 3:
    st.warning("⚠️ Veuillez ne pas dépasser 3 fichiers à la fois pour éviter les limites de mémoire.")
    st.stop()

if selected_files:
    all_chunks = []
    for fname in selected_files:
        fpath = os.path.join(PDF_FOLDER, fname)
        st.info(f"📥 Lecture : {fname}")
        chunks = pdf_to_chunks(fpath)
        all_chunks.extend(chunks)

    st.success(f"✅ {len(selected_files)} fichier(s) chargé(s), {len(all_chunks)} morceaux de texte extraits.")

    embeddings = embed_chunks(all_chunks)
    index = build_faiss_index(np.array(embeddings))

    # 💾 Sauvegarde de l'index
    faiss.write_index(index, "index.faiss")
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    st.success("✅ Export terminé : fichiers `index.faiss` et `chunks.json` prêts à être téléchargés.")
    st.markdown("Accédez à ces fichiers via : **Manage app > Files > Download**")

import streamlit as st

with open("index.faiss", "rb") as f:
    st.download_button("⬇️ Télécharger index.faiss", f, file_name="index.faiss")

with open("chunks.json", "rb") as f:
    st.download_button("⬇️ Télécharger chunks.json", f, file_name="chunks.json")



