import streamlit as st
import pickle

# Load model pipeline
@st.cache_resource
def load_model():
    with open("sentiment_analysis_pipeline.pkl", "rb") as file:
        pipeline = pickle.load(file)
    return pipeline

pipeline = load_model()

# Halaman utama Streamlit
st.title("Sentiment Analysis App")
st.write("Masukkan teks di bawah ini untuk menganalisis sentimen (positif atau negatif).")

# Input teks dari pengguna
user_input = st.text_area("Masukkan teks Anda:", "")

# Tombol prediksi
if st.button("Analisis Sentimen"):
    if user_input.strip() == "":
        st.warning("Harap masukkan teks untuk dianalisis.")
    else:
        # Prediksi sentimen
        prediction = pipeline.predict([user_input])[0]
        
        # Tampilkan hasil
        if prediction == "positive":
            st.success("Hasil Sentimen: **Positif** 😊")
        else:
            st.error("Hasil Sentimen: **Negatif** 😞")
