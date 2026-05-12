import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Fungsi sakti biar model gak rewel soal versi
def fix_layer(cls):
    return lambda **kwargs: cls(**{k: v for k, v in kwargs.items() if k not in ['batch_shape', 'optional', 'dtype']})

from tensorflow.keras.layers import InputLayer, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, ReLU

# PAKSA INPUT SHAPE DI SINI BIAR GAK ERROR 'SHAPE' LAGI
custom_objects = {
    'InputLayer': lambda **kwargs: InputLayer(input_shape=(224, 224, 3), **{k: v for k, v in kwargs.items() if k not in ['batch_shape', 'optional', 'dtype', 'input_shape']}),
    'Conv2D': fix_layer(Conv2D),
    'Dense': fix_layer(Dense),
    'Flatten': fix_layer(Flatten),
    'MaxPooling2D': fix_layer(MaxPooling2D),
    'BatchNormalization': fix_layer(BatchNormalization),
    'ReLU': fix_layer(ReLU)
}

st.title("🌱 Visual Waste Classifier")

@st.cache_resource
def load_model():
    # Load tanpa compile biar gak error optimizer
    return tf.keras.models.load_model('model_waste_final.h5', custom_objects=custom_objects, compile=False)

try:
    model = load_model()
    st.success("✅ AKHIRNYA! Website Siap Pakai.")
    
    file = st.file_uploader("Upload Foto Sampah Kamu", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption="Gambar Berhasil di-Upload", use_container_width=True)
        
        # PROSES GAMBAR
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype('float32')
        
        # Prediksi
        with st.spinner('Tunggu sebentar, lagi mikir...'):
            prediction = model.predict(img_array)
        
        # Tampilkan Hasil
        hasil = "ANORGANIK" if prediction[0] > 0.5 else "ORGANIK"
        st.subheader(f"Hasil Klasifikasi: {hasil}")
        st.info(f"Skor Keyakinan: {prediction[0][0]:.2f}")
            
except Exception as e:
    st.error(f"Dikit lagi Bro, ini kendalanya: {e}")
