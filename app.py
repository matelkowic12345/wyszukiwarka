import streamlit as st
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
import os

# --- Konfiguracja ---
image_folder = 'images'
embeddings_file = 'image_embeddings.pt'
paths_file = 'image_paths.pt'

# model SBERT / CLIP
model = SentenceTransformer('clip-ViT-B-32')

# --- Funkcja do generowania embeddings tylko raz ---
@st.cache_data
def generate_embeddings():
    image_paths = []
    images = []

    for f in os.listdir(image_folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(image_folder, f)
            try:
                img = Image.open(path).convert("RGB").resize((224,224))
                images.append(img)
                image_paths.append(path)
            except Exception as e:
                print(f"Nie udało się otworzyć {f}: {e}")

    image_embeddings = model.encode([img for img in images], convert_to_tensor=True)
    
    # zapisujemy embeddings i ścieżki, żeby nie generować ich za każdym razem
    torch.save(image_embeddings, embeddings_file)
    torch.save(image_paths, paths_file)

    return image_embeddings, image_paths

# --- Wczytanie embeddings ---
if os.path.exists(embeddings_file) and os.path.exists(paths_file):
    image_embeddings = torch.load(embeddings_file)
    image_paths = torch.load(paths_file)
else:
    image_embeddings, image_paths = generate_embeddings()

# --- Funkcja wyszukiwania ---
def search(query, top_k=5):
    query_emb = model.encode([query], convert_to_tensor=True)
    sims = util.cos_sim(query_emb, image_embeddings)[0]
    top_results = torch.topk(sims, k=min(top_k, len(sims)))
    return [image_paths[i] for i in top_results.indices]

# --- Streamlit UI ---
st.title("Wyszukiwarka obrazów (optymalizowana)")

query = st.text_input("Wpisz zapytanie", "")
top_k = st.slider("Liczba wyników", 1, 10, 5)

if query:
    results = search(query, top_k=top_k)
    if results:
        for r in results:
            st.image(Image.open(r))
    else:
        st.write("Brak wyników dla zapytania.")
