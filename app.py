import streamlit as st
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
import os

# model SBERT
model = SentenceTransformer('clip-ViT-B-32')

# wczytanie obrazów i ich embeddings
image_folder = 'images'
image_paths = []

# filtrujemy tylko obrazy i pomijamy błędne pliki
for f in os.listdir(image_folder):
    path = os.path.join(image_folder, f)
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # testowe otwarcie pliku
            img = Image.open(path).convert("RGB").resize((224, 224))
            image_paths.append(path)
        except Exception as e:
            print(f"Nie udało się otworzyć {f}: {e}")

# wczytanie obrazów do listy
images = [Image.open(p).convert("RGB").resize((224, 224)) for p in image_paths]
image_embeddings = model.encode([img for img in images], convert_to_tensor=True)

# funkcja wyszukiwania obrazów
def search(query, top_k=5, color_weight=0.3):
    query_en = translator.translate(query, src='pl', dest='en').text.lower()

    # wykrycie koloru w zapytaniu
    query_color = None
    for word in query_en.split():
        if word in color_dict:
            query_color = color_dict[word]
            break

    results = []
    for i, emb in enumerate(image_embeddings):
        # similarity CLIP
        sim = util.cos_sim(model.encode([query_en], convert_to_tensor=True), emb.unsqueeze(0))[0][0].item()

        # similarity koloru
        if query_color is not None:
            color_sim = color_similarity(query_color, get_dominant_color(images[i]))
            sim = (1 - color_weight) * sim + color_weight * color_sim

        results.append((i, sim))

    # sortowanie i top_k
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    return [image_paths[i] for i, s in results]

# przykładowe wyszukiwanie
query = "jadący samochód"
results = search(query, top_k=2)

from IPython.display import display
for r in results:
    display(Image.open(r))
