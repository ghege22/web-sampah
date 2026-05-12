import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Smart Waste AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS SUPER PREMIUM & MODERN ---
st.markdown("""
    <style>
    /* Import Font Google */
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,600;14..32,700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background dengan efek animasi gradien */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f0e8 100%);
        animation: gradientShift 10s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background: linear-gradient(135deg, #f5f7fa 0%, #e8f0e8 100%); }
        50% { background: linear-gradient(135deg, #e8f0e8 0%, #d4e4d4 100%); }
        100% { background: linear-gradient(135deg, #f5f7fa 0%, #e8f0e8 100%); }
    }
    
    /* Animasi fade in untuk konten */
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Header Premium */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        animation: slideDown 0.6s ease-out;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card Premium dengan efek hover */
    .premium-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 20px 35px -10px rgba(0,0,0,0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255,255,255,0.3);
        margin-bottom: 1.5rem;
    }
    
    .premium-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 45px -15px rgba(0,0,0,0.2);
    }
    
    /* Upload Area Styling */
    .stFileUploader > div {
        border: 3px dashed #2e7d32;
        border-radius: 20px;
        background: rgba(46, 125, 50, 0.05);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #1b5e20;
        background: rgba(46, 125, 50, 0.1);
        transform: scale(1.02);
    }
    
    /* Custom Button */
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(46, 125, 50, 0.3);
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #28a745, #2e7d32);
        border-radius: 10px;
    }
    
    /* Divider Premium */
    .premium-divider {
        background: linear-gradient(90deg, transparent, #2e7d32, #1b5e20, #2e7d32, transparent);
        height: 3px;
        border-radius: 3px;
        margin: 2rem 0;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2e7d32;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1b5e20;
    }
    
    /* Info Text */
    .info-text {
        color: #666;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 1rem;
    }
    
    /* Badge untuk kategori */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .premium-card {
            padding: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI BYPASS LAYER (TIDAK DIUBAH - KRUSIAL) ---
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

# --- HEADER PREMIUM ---
st.markdown('<div class="fade-in">', unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>♻️ Smart Waste Classifier AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>✨ Solusi cerdas berbasis AI untuk memilah sampah Organik dan Anorganik secara otomatis ✨</p>", unsafe_allow_html=True)

# Stats Preview
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div style="text-align: center;">
            <p style="font-size: 2rem; margin: 0;">🎯</p>
            <p style="font-weight: 600; color: #2e7d32;">Akurasi Tinggi</p>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div style="text-align: center;">
            <p style="font-size: 2rem; margin: 0;">⚡</p>
            <p style="font-weight: 600; color: #2e7d32;">Deteksi Cepat</p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div style="text-align: center;">
            <p style="font-size: 2rem; margin: 0;">🌍</p>
            <p style="font-weight: 600; color: #2e7d32;">Ramah Lingkungan</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

# --- LAYOUT KOLOM UTAMA ---
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="premium-card fade-in">', unsafe_allow_html=True)
    st.subheader("📸 Upload & Preview")
    st.markdown("💡 **Tips:** Upload foto sampah dengan pencahayaan yang baik untuk hasil optimal")
    
    uploaded_file = st.file_uploader(
        "Klik atau seret file foto ke sini", 
        type=["jpg", "jpeg", "png"],
        help="Format yang didukung: JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="📷 Preview Foto", use_container_width=True)
        
        # Tombol refresh dengan efek
        if st.button("🔄 Ganti Foto", use_container_width=True):
            st.rerun()
    else:
        # Placeholder animasi
        st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <p style="font-size: 4rem; margin: 0;">📤</p>
                <p style="color: #888;">Belum ada foto yang dipilih</p>
                <p style="color: #aaa; font-size: 0.85rem;">Upload foto sampah untuk memulai analisis</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tambahan informasi edukasi
    with st.expander("🌱 Edukasi Pemilahan Sampah", expanded=False):
        st.markdown("""
        **🗑️ Sampah Organik:**
        - Sisa makanan, sayur, buah
        - Daun kering, ranting
        - Dapat diurai oleh alam (kompos)
        
        **♻️ Sampah Anorganik:**
        - Plastik, botol, kemasan
        - Kaleng, kaca, logam
        - Perlu didaur ulang
        """)

with right_col:
    st.markdown('<div class="premium-card fade-in">', unsafe_allow_html=True)
    st.subheader("🤖 Hasil Analisis AI")
    
    if uploaded_file:
        try:
            model = load_model()
            
            with st.spinner('🧠 AI sedang menganalisis foto Anda...'):
                time.sleep(0.5)  # Animasi kecil
                
                # Proses gambar
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0).astype('float32')
                
                # Prediksi
                prediction = model.predict(img_array)
                score = float(prediction[0][0])
                
                # Tentukan hasil
                if score > 0.5:
                    label, color, icon, bg_color = "ANORGANIK", "#ff6b6b", "🥤", "#fff5f5"
                    confidence = score * 100
                    saran = "Masukkan ke tempat sampah **Anorganik** untuk didaur ulang"
                else:
                    label, color, icon, bg_color = "ORGANIK", "#51cf66", "🍎", "#f0fff4"
                    confidence = (1 - score) * 100
                    saran = "Masukkan ke tempat sampah **Organik** untuk dijadikan kompos"
                
                # Animasi loading progress
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.01)
                
                # Tampilkan hasil dengan animasi
                st.markdown(f"""
                    <div style="background: {bg_color}; border-radius: 20px; padding: 2rem; text-align: center; border: 2px solid {color}; animation: fadeIn 0.5s ease-out;">
                        <div style="font-size: 4rem; margin-bottom: 0.5rem;">{icon}</div>
                        <div style="background: {color}; display: inline-block; padding: 0.5rem 1.5rem; border-radius: 50px; margin-bottom: 1rem;">
                            <span style="color: white; font-weight: 700; font-size: 1.2rem;">{label}</span>
                        </div>
                        <div style="font-size: 2.5rem; font-weight: 800; color: {color}; margin: 1rem 0;">
                            {confidence:.1f}%
                        </div>
                        <div style="color: #555; margin-top: 1rem;">
                            <span style="font-weight: 600;">🎯 Tingkat Keyakinan AI</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Saran dengan ikon
                st.success(f"💡 **Saran:** {saran}")
                
                # Tips tambahan
                st.caption("✨ AI ini terus belajar untuk membantu menjaga lingkungan kita!")
                
        except Exception as e:
            st.error(f"⚠️ Terjadi kendala teknis: {e}")
            st.info("Pastikan file model 'model_waste_final.h5' tersedia di direktori yang sama")
    else:
        # Tampilan kosong yang menarik
        st.markdown("""
            <div style="text-align: center; padding: 3rem 1rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🖼️</div>
                <h3 style="color: #555;">Belum Ada Foto</h3>
                <p style="color: #888;">Upload foto sampah di sebelah kiri untuk melihat hasil analisis AI</p>
                <div style="margin-top: 2rem;">
                    <span class="badge" style="background: #e8f5e9; color: #2e7d32;">📸 Organik</span>
                    <span class="badge" style="background: #ffebee; color: #d32f2f;">🥤 Anorganik</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

# --- FOOTER PREMIUM ---
col_f1, col_f2, col_f3 = st.columns([1, 2, 1])
with col_f2:
    st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <p style="color: #888; margin: 0;">
                🌿 <strong>Smart Waste Management System</strong> 🌿<br>
                Dibuat dengan ❤️ untuk lingkungan yang lebih baik
            </p>
            <p style="color: #aaa; font-size: 0.8rem; margin-top: 0.5rem;">
                © 2026 | Powered by TensorFlow & Streamlit
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
