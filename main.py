import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Configuração da página
st.set_page_config(page_title="Classificador de Pets", page_icon="🐶🐱")

# Título
st.title("Classificador de Pets 🐶🐱")
st.write("Faça o upload de uma imagem para descobrir se é um cachorro ou um gato!")

# Carregar o modelo treinado
MODEL_PATH = "modelo_cachorro_gato.h5"
model = load_model(MODEL_PATH)

# Função para processar a imagem
def process_image(uploaded_image):
    img = Image.open(uploaded_image).convert('RGB')  # Certificar que está em RGB
    img = img.resize((128, 128))  # Redimensionar para o tamanho usado no treinamento
    img_array = img_to_array(img)  # Converter para um array numpy
    img_array = img_array / 255.0  # Normalizar os valores para o intervalo [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar uma dimensão para batch
    return img_array

# Componente para upload da imagem
uploaded_image = st.file_uploader("Envie uma imagem de um gato ou cachorro:", type=["jpg", "jpeg", "png"])

# Inferência e exibição do resultado
if uploaded_image is not None:
    # Mostrar a imagem carregada
    st.image(uploaded_image, caption="Imagem carregada", use_container_width=True)

    # Processar a imagem
    img_array = process_image(uploaded_image)

    # Realizar a predição
    prediction = model.predict(img_array)
    class_name = "Cachorro 🐶" if prediction >= 0.5 else "Gato 🐱"
    confidence = prediction[0][0] if prediction >= 0.5 else 1 - prediction[0][0]

    # Exibir o resultado
    st.write(f"### Resultado: **{class_name}**")
    st.write(f"Confiança: **{confidence * 100:.2f}%**")
