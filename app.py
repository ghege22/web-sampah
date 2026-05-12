import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# --- CSS CUSTOM BIAR GANTENG ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI BYPASS LAYER (ANTI-ERROR) ---
def fix_layer(cls):
    return lambda **kwargs: cls(**{k: v for k, v in kwargs.items() if k not in ['batch_shape', 'optional', 'dtype']})

from tensorflow.keras.layers import InputLayer, Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, ReLU

custom_objects = {
    'InputLayer': lambda **kwargs: InputLayer(input_shape=(224, 224, 3), **{k: v for k, v in kwargs.items() if k not in ['batch_shape', 'optional', 'dtype', 'input_shape']}),
    'Conv2D': fix_layer(Conv2D),
    'Dense': fix_layer(Dense),
    'Flatten': fix_layer(Flatten),
    'MaxPooling2D': fix_layer(MaxPooling2D),
    'BatchNormalization': fix_layer(BatchNormalization),
    'ReLU': fix_layer(ReLU)
}

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_waste_final.h5', custom_objects=custom_objects, compile=False)

# --- TAMPILAN UTAMA ---
st.write("### ♻️ AI Waste Classifier")
st.write("Sistem Klasifikasi Sampah Otomatis Menggunakan Deep Learning")
st.divider()

try:
    model = load_model()
    
    uploaded_file = st.file_uploader("Pilih foto sampah...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        image = Image.open(uploaded_file).convert('RGB')
        
        with col1:
            st.image(image, caption="Foto Sampah", use_container_width=True)
        
        with col2:
            with st.spinner('Menganalisis...'):
                # Preprocessing
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0).astype('float32')
                
                # Prediksi
                prediction = model.predict(img_array)
                score = float(prediction[0][0])
                
                if score > 0.5:
                    label = "ANORGANIK"
                    persen = score * 100
                    warna = "#ff4b4b"
                    icon = "🥤"
                else:
                    label = "ORGANIK"
                    persen = (1 - score) * 100
                    warna = "#28a745"
                    icon = "🍎"

                # Box Hasil
                st.markdown(f"""
                    <div style="background-color: {warna}; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                        <h2 style='color: white;'>{icon} {label}</h2>
                        <h1 style='color: white; margin: 0;'>{persen:.1f}%</h1>
                        <p style='color: white;'>Tingkat Keyakinan</p>
                    </div>
                """, unsafe_allow_html=True)

    else:
        st.info("Silakan upload foto sampah untuk memulai.")

except Exception as e:
    st.error(f"Terjadi kesalahan teknis: {e}")

st.divider()
st.caption("Developed with ❤️ for Waste Management Education")
