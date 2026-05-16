import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
import faiss
import numpy as np
import re
import os # Tambahkan modul os untuk memanipulasi environment variabel

# ==========================================
# 1. KONFIGURASI HALAMAN & UTILITY
# ==========================================
st.set_page_config(page_title="AI Culinary Assistant", page_icon="🍳", layout="wide")

st.title("🍳 AI Culinary Assistant Dashboard")
st.markdown("Cari resep masakan Indonesia berdasarkan bahan sisa di kulkas menggunakan Kecerdasan Buatan (NLP & Vector Search).")
st.divider()

# ==========================================
# 2. SISTEM MEMUAT DATA & MODEL AI (CACHED)
# ==========================================
@st.cache_resource
def load_models_and_data():
    HF_TOKEN = "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"
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    # 2. Memuat Dataset Gated (Pustaka akan otomatis membaca variabel OS di atas)
    dataset = load_dataset("junwatu/indonesian-recipes", split="train")
    df = dataset.to_pandas()
    
    # Deteksi otomatis nama kolom instruksi/langkah memasak
    kolom_instruksi = 'instructions'
    if 'instructions' not in df.columns:
        for col in df.columns:
            if col in ['steps', 'directions', 'step', 'cara', 'langkah', 'instructions']:
                kolom_instruksi = col
                break
                
    # Sinkronisasi nama kolom ke 'instructions'
    if kolom_instruksi != 'instructions':
        df = df.rename(columns={kolom_instruksi: 'instructions'})
        
    # Memastikan kolom penting bertipe string dan membersihkan nilai kosong (NaN / float error)
    df['ingredients'] = df['ingredients'].astype(str).replace(['nan', 'None', ''], 'tidak ada bahan')
    df['title'] = df['title'].astype(str)
    df['instructions'] = df['instructions'].astype(str)
    
    # Mengambil 30 data pertama untuk optimasi performa server cloud gratis
    df_sample = df.head(30).copy()
    
    # B. Fitur 3: Load Model Zero-Shot untuk Klasifikasi Kategori Otomatis
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    kandidat_label = ["makanan berkuah sup soto", "makanan gorengan kering tumis", "kue makanan manis dessert"]
    
    kategori_terdeteksi = []
    for index, row in df_sample.iterrows():
        teks_konteks = f"{row['title']} {row['ingredients']}"
        hasil = classifier(teks_konteks, candidate_labels=kandidat_label)
        label_teratas = hasil['labels'][0]
        
        if "berkuah" in label_teratas: 
            kategori = "Berkuah"
        elif "gorengan" in label_teratas: 
            kategori = "Goreng/Kering"
        else: 
            kategori = "Kue & Manis"
        kategori_terdeteksi.append(kategori)
        
    df_sample['kategori_ai'] = kategori_terdeteksi
    
    # C. Fitur 1: Load Model Embedding untuk Pencarian Semantik
    model_embed = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    df_sample['clean_ingredients'] = df_sample['ingredients'].str.replace('\n', ' ', regex=False)
    
    # Membuat representasi vektor koordinat bahan makanan
    vektor_bahan = model_embed.encode(df_sample['clean_ingredients'].tolist(), show_progress_bar=False)
    
    # D. Membangun Database Vektor FAISS
    dimensi = vektor_bahan.shape[1]
    index_faiss = faiss.IndexFlatL2(dimensi)
    index_faiss.add(np.array(vektor_bahan))
    
    return df_sample, model_embed, index_faiss

# Menampilkan indikator loading saat pertama kali aplikasi dijalankan
with st.spinner("🤖 Sedang menginisialisasi AI dan memuat database resep... (Mohon tunggu sekitar 1-2 menit)"):
    df_sample, model_embed, index_faiss = load_models_and_data()

# ==========================================
# 3. FITUR 4: LOGIKA PORSI & FILTER ALERGI
# ==========================================
def sesuaikan_porsi_dan_alergi(teks_bahan, faktor_porsi, list_alergi=[]):
    # Cek apakah ada bahan pemicu alergi di dalam resep
    if list_alergi:
        for alergi in list_alergi:
            if alergi.strip().lower() in teks_bahan.lower():
                return None  # Mengembalikan None jika mengandung bahan alergi
                
    # Fungsi pembantu regex untuk mengalikan angka takaran bahan makanan
    def ganti_angka(match):
        angka = float(match.group())
        return str(round(angka * faktor_porsi, 2))
        
    return re.sub(r'\d+(\.\d+)?', ganti_angka, teks_bahan)

# ==========================================
# 4. SIDEBAR PANEL INTERAKTIF (INPUT USER)
# ==========================================
st.sidebar.header("🎛️ Panel Kontrol Pencarian")

kueri_bahan = st.sidebar.text_input("🛒 Bahan sisa di kulkas:", placeholder="Contoh: ayam dan cabai")
porsi_diinginkan = st.sidebar.slider("👥 Kelipatan Porsi Masakan:", min_value=1, max_value=10, value=1)
input_alergi = st.sidebar.text_input("⚠️ Bahan Alergi (Pisahkan dengan koma):", placeholder="Contoh: kecap, udang")

# Mengolah teks input alergi menjadi list bersih
list_alergi = [x.strip() for x in input_alergi.split(",")] if input_alergi else []

kategori_filter = st.sidebar.selectbox("🍽️ Filter Jenis Makanan (AI):", ["Semua", "Berkuah", "Goreng/Kering", "Kue & Manis"])

# ==========================================
# 5. HALAMAN UTAMA DASHBOARD DISPLAY & ENGINE
# ==========================================
if st.sidebar.button("🚀 Cari Resep Sekarang", type="primary"):
    if kueri_bahan:
        st.subheader(f"🔍 Hasil Analisis Pencarian Semantik untuk: '{kueri_bahan}'")
        
        # Jalankan FAISS Search berdasarkan kemiripan makna kueri bahan sisa
        kueri_vektor = model_embed.encode([kueri_bahan])
        jarak, indeks = index_faiss.search(np.array(kueri_vektor), len(df_sample))
        
        hasil_ditemukan = 0
        
        # Melakukan perulangan hasil pencarian terdekat
        for idx in indeks[0]:
            resep = df_sample.iloc[idx]
            
            # Aplikasi Filter Kategori AI (Fitur 3)
            if kategori_filter != "Semua" and resep['kategori_ai'] != kategori_filter:
                continue
                
            # Aplikasi Filter Alergi & Perubahan Jumlah Porsi (Fitur 4)
            bahan_termodifikasi = sesuaikan_porsi_dan_alergi(resep['ingredients'], porsi_diinginkan, list_alergi)
            if bahan_termodifikasi is None:
                continue  # Lewati resep ini jika terkena filter alergi
                
            hasil_ditemukan += 1
            
            # Tampilan visualisasi kartu resep (Card Layout)
            with st.expander(f"⭐ REKOMENDASI #{hasil_ditemukan}: {resep['title'].upper()} ({resep['kategori_ai']})", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**📋 Bahan-Bahan (Telah Disesuaikan ke porsi {porsi_diinginkan}x):**")
                    st.text(bahan_termodifikasi)
                with col2:
                    st.markdown("**📖 Langkah-Langkah Memasak:**")
                    st.text(resep['instructions'])
            
            if hasil_ditemukan == 2:  # Batasi hanya menampilkan maksimal 2 hasil terbaik
                break
                
        if hasil_ditemukan == 0:
            st.error("❌ Maaf, tidak ada resep yang cocok dengan kriteria bahan, jenis makanan, atau batasan alergi Anda.")
    else:
        st.warning("⚠️ Silakan masukkan bahan makanan terlebih dahulu di kolom sidebar sebelah kiri!")
else:
    # Tampilan panduan awal sebelum pengguna menekan tombol cari
    st.info("💡 Petunjuk Penggunaan: Isi bahan makanan yang Anda miliki di panel sidebar sebelah kiri, tentukan porsi serta pantangan alergi Anda, lalu klik tombol 'Cari Resep Sekarang'!")
    
    # Menampilkan Visualisasi Data Insight Koleksi Resep Masakan
    st.subheader("📊 Statistik Menu Kuliner Saat Ini (Analisis Klasifikasi AI)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Daftar Menu Masakan Tersedia (Sampel Data)**")
        st.dataframe(
            df_sample[['title', 'kategori_ai']].rename(
                columns={'title': 'Nama Masakan', 'kategori_ai': 'Kategori Prediksi AI'}
            ), 
            use_container_width=True,
            hide_index=True
        )
    with col2:
        st.markdown("**Penyebaran Kategori Kuliner Hasil Analisis AI**")
        kategori_counts = df_sample['kategori_ai'].value_counts()
        st.bar_chart(kategori_counts)
