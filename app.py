import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

translation_dict = {
    "samochód": "car",
    "pies": "dog",
    "kot": "cat",
    "drzewo": "tree",
    "trawa": "grass",
    "dom": "house",
    "rower": "bike"
}

image_folder = 'images'
image_paths = []

for f in os.listdir(image_folder):
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(image_folder, f)
        try:
            img = Image.open(path).convert("RGB")
            image_paths.append(path)
        except Exception as e:
            print(f"Nie udało się otworzyć {f}: {e}")

images = [Image.open(p).convert("RGB") for p in image_paths]
image_inputs = processor(images=images, return_tensors="pt", padding=True)
with torch.no_grad():
    image_embeddings = model.get_image_features(**image_inputs)
image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

def search(query, top_k=5):
    # Mapowanie polskich słów na angielskie
    query_en = translation_dict.get(query.lower(), query.lower())
    text_inputs = processor(text=[query_en], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    
    sims = torch.matmul(text_embeddings, image_embeddings.T)[0]
    top_results = torch.topk(sims, k=min(top_k, len(sims)))
    
    return [image_paths[i] for i in top_results.indices]

st.title("Wyszukiwarka obrazów")

query = st.text_input("Wpisz zapytanie (polskie lub angielskie)", "")
top_k = st.slider("Liczba wyników", 1, 10, 5)

if query:
    results = search(query, top_k=top_k)
    if results:
        for r in results:
            st.image(Image.open(r))
    else:
        st.write("Brak wyników dla zapytania.")
