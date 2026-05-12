import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Fungsi sakti biar model lama mau jalan di server baru
def fix_layer(cls):
    return lambda **kwargs: cls(**{k: v for k, v in kwargs.items() if k not in ['batch_shape', 'optional', 'dtype']})

from tensorflow.keras.layers import InputLayer, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, ReLU

custom_objects = {
    'InputLayer': fix_layer(InputLayer),
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
    # Tambahin compile=False biar gak ribet urusan optimizer
    return tf.keras.models.load_model('model_waste_final.h5', custom_objects=custom_objects, compile=False)

try:
    model = load_model()
    st.success("✅ Website Berhasil Online! Silakan Upload Foto.")
    
    file = st.file_uploader("Upload Foto Sampah", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption="Gambar yang diupload", use_container_width=True)
        
        # PROSES GAMBAR BIAR MODEL GAK ERROR 'SHAPE'
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype('float32')
        
        # Prediksi
        prediction = model.predict(img_array)
        
        # Tampilkan Hasil
        if prediction[0] > 0.5:
            st.header("Hasil: ANORGANIK")
        else:
            st.header("Hasil: ORGANIK")
            
except Exception as e:
    st.error(f"Ada kendala teknis: {e}")
