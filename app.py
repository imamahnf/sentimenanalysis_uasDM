import streamlit as st
import pickle

# Memuat pipeline dari file pickle
with open("deployment_pipeline.pkl", "rb") as file:
    pipeline = pickle.load(file)

# Menambahkan CSS kustom untuk Streamlit
st.markdown("""
    <style>
        body {
            font-family: 'Poppins', Arial, sans-serif;
            background: #f0f8ff;
            padding: 20px;
        }

        h1 {
            color: #007bff;
            text-align: center;
            margin-bottom: 20px;
        }

        .stButton > button {
            background-color: #28a745;
            color: white;
            border-radius: 5px;
            font-size: 16px;
            padding: 10px 20px;
        }

        .stButton > button:hover {
            background-color: #218838;
        }
    </style>
""", unsafe_allow_html=True)

# Header aplikasi
st.title("Aplikasi Prediksi Sentimen")

# Memilih jenis input
tab = st.radio("Pilih Jenis Input", ["Teks", "Dokumen"])

if tab == "Teks":
    # Input teks untuk analisis
    input_teks = st.text_area("Masukkan teks untuk analisis sentimen:", height=150)
    if st.button("Prediksi Sentimen"):
        if input_teks.strip():
            # Prediksi sentimen menggunakan pipeline
            hasil_prediksi = pipeline.predict([input_teks])[0]
            if hasil_prediksi == "positive":
                st.success("Hasil Prediksi: Sentimen Positif 😊")
            else:
                st.error("Hasil Prediksi: Sentimen Negatif 😞")
        else:
            st.warning("Harap masukkan teks terlebih dahulu.")
elif tab == "Dokumen":
    # Input file dokumen
    file = st.file_uploader("Unggah file dokumen (.txt)", type=["txt"])
    if file is not None:
        file_content = file.read().decode("utf-8")
        if st.button("Prediksi Sentimen Dokumen"):
            if file_content.strip():
                # Prediksi sentimen untuk seluruh dokumen
                hasil_prediksi = pipeline.predict([file_content])[0]
                if hasil_prediksi == "positive":
                    st.success("Hasil Prediksi: Sentimen Positif 😊")
                else:
                    st.error("Hasil Prediksi: Sentimen Negatif 😞")
            else:
                st.warning("Dokumen kosong. Harap unggah file yang valid.")
