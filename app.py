import streamlit as st
import pickle

# ==========================================
# Fungsi untuk Memuat Pipeline
# ==========================================
@st.cache_resource
def load_pipeline():
    with open('sentiment_pipeline.pkl', 'rb') as file:
        return pickle.load(file)

# Memuat pipeline
pipeline = load_pipeline()

# ==========================================
# Judul Aplikasi
# ==========================================
st.title("Analisis Sentimen Positif & Negatif")
st.write("""
Aplikasi ini menggunakan model SVM yang telah dilatih untuk menganalisis sentimen teks.
Hanya mendukung dua sentimen: *Positif* dan *Negatif*.
""")

# ==========================================
# Input Teks dari Pengguna
# ==========================================
user_input = st.text_area("Masukkan teks untuk analisis sentimen:", "")

# Tombol untuk Prediksi
if st.button("Analisis Sentimen"):
    if user_input.strip():
        # Prediksi sentimen
        prediction = pipeline.predict([user_input])[0]

        # Menampilkan hasil
        st.write("*Kalimat:*", user_input)
        if prediction == 'positif':
            st.success("*Prediksi Sentimen:* Positif 😊")
        elif prediction == 'negatif':
            st.error("*Prediksi Sentimen:* Negatif 😟")
    else:
        st.warning("Teks tidak boleh kosong!")

# ==========================================
# Sidebar Informasi
# ==========================================
st.sidebar.title("Tentang Aplikasi")
st.sidebar.info("""
Aplikasi ini dirancang untuk mendeteksi sentimen positif dan negatif pada teks.
Model SVM telah dilatih dengan pipeline preprocessing, TF-IDF, dan SVM.
""")
