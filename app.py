import streamlit as st
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
import os
from googletrans import Translator  # pip install googletrans==4.0.0-rc1

# model SBERT (CLIP)
model = SentenceTransformer('clip-ViT-B-32')

# translator (PL -> EN)
translator = Translator()

# wczytanie obrazów i filtracja błędnych plików
image_folder = 'images'
image_paths = []
images = []

for f in os.listdir(image_folder):
    path = os.path.join(image_folder, f)
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            img = Image.open(path).convert("RGB").resize((224,224))
            images.append(img)
            image_paths.append(path)
        except Exception as e:
            print(f"Nie udało się otworzyć {f}: {e}")

# embeddingi obrazów
image_embeddings = model.encode(images, convert_to_tensor=True)

# funkcja wyszukiwania obrazów
def search(query, top_k=5):
    # tłumaczenie zapytania na angielski
    query_en = translator.translate(query, src='pl', dest='en').text
    query_emb = model.encode([query_en], convert_to_tensor=True)

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
