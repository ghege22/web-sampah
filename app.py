import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Smart Waste AI",
    page_icon="♻️",
    layout="wide"
)

# --- CSS UNTUK TAMPILAN PREMIUM ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #e8f5e9, #ffffff);
    }
    .main-title {
        color: #2e7d32;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .info-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI BYPASS LAYER ---
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

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_waste_final.h5', custom_objects=custom_objects, compile=False)

# --- HEADER ---
st.markdown("<h1 class='main-title'>♻️ Smart Waste Classifier AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Solusi cerdas berbasis AI untuk memilah sampah Organik dan Anorganik secara otomatis.</p>", unsafe_allow_html=True)
st.divider()

# --- LAYOUT KOLOM ---
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.subheader("📸 Ambil atau Upload Foto")
    uploaded_file = st.file_uploader("Seret foto sampah ke sini...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Preview Foto", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.subheader("📊 Hasil Analisis")
    
    if uploaded_file:
        try:
            model = load_model()
            with st.spinner('Menganalisis jenis sampah...'):
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0).astype('float32')
                
                prediction = model.predict(img_array)
                score = float(prediction[0][0])
                
                if score > 0.5:
                    label, color, icon = "ANORGANIK", "#ff4b4b", "🥤"
                    confidence = score * 100
                else:
                    label, color, icon = "ORGANIK", "#28a745", "🍎"
                    confidence = (1 - score) * 100

                st.markdown(f"""
                    <div style="background-color: {color}; color: white; padding: 25px; border-radius: 15px; text-align: center; border: 3px solid rgba(255,255,255,0.3);">
                        <h1 style='color: white; font-size: 50px; margin-bottom: 0;'>{icon}</h1>
                        <h2 style='color: white; margin: 0;'>{label}</h2>
                        <h1 style='color: white; font-size: 60px; margin-top: 10px;'>{confidence:.1f}%</h1>
                        <p style='color: white; font-weight: bold;'>Tingkat Keyakinan AI</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.progress(confidence / 100)
                st.write(f"Saran: Segera buang ke tempat sampah **{label.lower()}** terdekat!")
                
        except Exception as e:
            st.error(f"Terjadi kendala: {e}")
    else:
        st.info("Menunggu foto untuk dianalisis...")
    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER ---
st.divider()
st.markdown("<p style='text-align: center; color: #888;'>&copy; 2026 Smart Waste Management System | Built with ❤️ for the Environment</p>", unsafe_allow_html=True)
