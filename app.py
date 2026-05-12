import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Waste Classifier")
st.title("🌱 Visual Waste Classifier")

@st.cache_resource
def load_model():
    # Di server nanti ini pasti jalan karena versinya pas
    return tf.keras.models.load_model('model_waste_final.h5')

try:
    model = load_model()
    st.success("✅ Website Siap Digunakan!")
    
    file = st.file_uploader("Upload Foto Sampah", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, use_container_width=True)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        hasil = "ANORGANIK" if prediction[0] > 0.5 else "ORGANIK"
        st.header(f"Hasil: {hasil}")
except Exception as e:
    st.error(f"Error: {e}")