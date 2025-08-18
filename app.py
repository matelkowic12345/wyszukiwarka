import streamlit as st
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util
import os
from googletrans import Translator
from collections import Counter

# Inicjalizacja modelu i tłumacza
model = SentenceTransformer('clip-ViT-B-32')
translator = Translator()

# Folder z obrazami
image_folder = 'images'
image_paths = []

# Funkcje do kolorów
color_dict = {
    "czerwony": (255,0,0),
    "zielony": (0,255,0),
    "niebieski": (0,0,255),
    "żółty": (255,255,0),
    "czarny": (0,0,0),
    "biały": (255,255,255),
    "pomarańczowy": (255,165,0),
    "różowy": (255,192,203),
    "fioletowy": (128,0,128),
    "brązowy": (165,42,42),
    "szary": (128,128,128)
}

def get_dominant_color(image):
    image = image.resize((50,50))
    pixels = list(image.getdata())
    most_common = Counter(pixels).most_common(1)[0][0]
    return most_common

def color_similarity(c1, c2):
    # prosta odwrotność odległości euklidesowej w RGB
    dist = sum((a-b)**2 for a,b in zip(c1,c2))**0.5
    return 1 / (1 + dist)

# Wczytywanie obrazów
for f in os.listdir(image_folder):
    path = os.path.join(image_folder, f)
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            img = Image.open(path).convert("RGB").resize((224,224))
            image_paths.append(path)
        except Exception as e:
            print(f"Nie udało się otworzyć {f}: {e}")

images = [Image.open(p).convert("RGB").resize((224,224)) for p in image_paths]
image_embeddings = model.encode([img for img in images], convert_to_tensor=True)

# Funkcja wyszukiwania
def search(query, top_k=5, color_weight=0.3):
    # Tłumaczenie zapytania na angielski
    query_en = translator.translate(query, src='pl', dest='en').text.lower()

    # Wykrycie koloru w zapytaniu
    query_color = None
    for word in query.lower().split():
        if word in color_dict:
            query_color = color_dict[word]
            break

    results = []
    query_emb = model.encode([query_en], convert_to_tensor=True)
    for i, emb in enumerate(image_embeddings):
        # similarity CLIP
        sim = util.cos_sim(query_emb, emb.unsqueeze(0))[0][0].item()

        # similarity koloru
        if query_color is not None:
            color_sim = color_similarity(query_color, get_dominant_color(images[i]))
            sim = (1 - color_weight) * sim + color_weight * color_sim

        results.append((i, sim))

    # sortowanie i top_k
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    return [image_paths[i] for i, s in results]

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
