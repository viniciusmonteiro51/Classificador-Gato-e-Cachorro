import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np


st.set_page_config(page_title="Classificador de Pets", page_icon="🐶🐱")

st.title("Classificador de Pets 🐶🐱")
st.markdown("[Link para o código](https://colab.research.google.com/drive/1wx17kG8Cb4iLZ8TjCdloM4FIsWRqxPz8?usp=sharing)")

st.write("Faça o upload de uma imagem para descobrir se é um cachorro ou um gato!")

MODEL_PATH = "modelo_cachorro_gato.h5"
model = load_model(MODEL_PATH)

def process_image(uploaded_image):
    img = Image.open(uploaded_image).convert('RGB')
    img = img.resize((128, 128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

uploaded_image = st.file_uploader("Envie uma imagem de um gato ou cachorro:", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:

    st.image(uploaded_image, caption="Imagem carregada", use_container_width=True)

    img_array = process_image(uploaded_image)

    prediction = model.predict(img_array)
    class_name = "Cachorro 🐶" if prediction >= 0.5 else "Gato 🐱"
    confidence = prediction[0][0] if prediction >= 0.5 else 1 - prediction[0][0]

    st.write(f"### Resultado: **{class_name}**")
    st.write(f"Confiança: **{confidence * 100:.2f}%**")
