import streamlit as st
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
import os

# model SBERT
model = SentenceTransformer('clip-ViT-B-32')

# wczytanie obrazów i ich embeddings
image_folder = 'images'
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
images = [Image.open(p).convert("RGB").resize((224,224)) for p in image_paths]
image_embeddings = model.encode([img for img in images], convert_to_tensor=True)

def search(query, top_k=5):
    query_emb = model.encode([query], convert_to_tensor=True)
    sims = util.cos_sim(query_emb, image_embeddings)[0]
    top_results = torch.topk(sims, k=min(top_k, len(sims)))
    return [image_paths[i] for i in top_results.indices]

st.title("Wyszukiwarka obrazów")

query = st.text_input("Wpisz zapytanie", "")
top_k = st.slider("Liczba wyników", 1, 10, 5)

if query:
    results = search(query, top_k=top_k)
    for r in results:
        st.image(Image.open(r))
