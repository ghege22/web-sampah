import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Fungsi sakti untuk menjinakkan error versi model
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
    return tf.keras.models.load_model('model_waste_final.h5', custom_objects=custom_objects, compile=False)

try:
    model = load_model()
    st.success("✅ Website Berhasil Online!")
    
    file = st.file_uploader("Upload Foto Sampah", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, use_container_width=True)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        pred = model.predict(img_array)
        hasil = "ANORGANIK" if pred[0] > 0.5 else "ORGANIK"
        st.header(f"Hasil: {hasil}")
except Exception as e:
    st.error(f"Sistem sedang sinkronisasi: {e}")
