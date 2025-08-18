import streamlit as st
from PIL import Image
from my_search_module import search, image_paths, images  # zaimportuj swoje funkcje i zmienne

st.title("Mini wyszukiwarka obrazów")

query = st.text_input("Wpisz, czego szukasz:")

top_k = st.slider("Ile wyników pokazać?", 1, 10, 5)

if st.button("Szukaj"):
    if query.strip() == "":
        st.warning("Wpisz zapytanie")
    else:
        results = search(query, top_k=top_k)
        if len(results) == 0:
            st.info("Nie znaleziono wyników")
        else:
            for r in results:
                img = Image.open(r)
                st.image(img, use_column_width=True)
