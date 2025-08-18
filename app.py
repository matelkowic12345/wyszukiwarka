import streamlit as st
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
import os

# Model CLIP z SentenceTransformers
model = SentenceTransformer('clip-ViT-B-32')

# Wczytanie obrazów i filtrowanie tylko poprawnych plików
image_folder = 'images'
image_paths = []

for f in os.listdir(image_folder):
    path = os.path.join(image_folder, f)
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            img = Image.open(path).convert("RGB").resize((224,224))
            image_paths.append(path)
        except Exception as e:
            print(f"Nie udało się otworzyć {f}: {e}")

# Wczytanie obrazów do listy
images = [Image.open(p).convert("RGB").resize((224,224)) for p in image_paths]

# Obliczenie embeddingów obrazów
image_embeddings = model.encode(images, convert_to_tensor=True)

# Funkcja wyszukiwania obrazów po zapytaniu
def search(query, top_k=5):
    query_emb = model.encode([query], convert_to_tensor=True)
    sims = util.cos_sim(query_emb, image_embeddings)[0]
    top_results = torch.topk(sims, k=min(top_k, len(sims)))
    return [image_paths[i] for i in top_results.indices]

# Streamlit UI
st.title("Wyszukiwarka obrazów")

query = st.text_input("Wpisz zapytanie", "")
top_k = st.slider("Liczba wyników", 1, 10, 5)

if query:
    results = search(query, top_k=top_k)
    if results:
        for r in results:
            st.image(Image.open(r))
    else:
        st.write("Brak wyników dla zapytania.")
