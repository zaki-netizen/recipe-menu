import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd
import faiss
import numpy as np
import re

# ==========================================
# 1. KONFIGURASI HALAMAN & CACHE STREAMLIT
# ==========================================
st.set_page_config(page_title="AI Culinary Assistant", page_icon="🍳", layout="wide")

st.title("🍳 AI Culinary Assistant Dashboard")
st.markdown("Cari resep masakan Indonesia berdasarkan bahan sisa di kulkas menggunakan Kecerdasan Buatan (NLP & Vector Search).")
st.hr()

# Kita gunakan st.cache_resource agar model AI tidak di-load ulang setiap kali user mengklik tombol (bikin lemot)
@st.cache_resource
def load_models_and_data():
    # Gunakan token Hugging Face Anda (Disarankan pakai Streamlit Secrets untuk keamanan)
    # Jika deploy lokal, Anda bisa ganti langsung dengan string token Anda hf_...
    HF_TOKEN = st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else "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"
    
    # Load Dataset
    dataset = load_dataset("junwatu/indonesian-recipes", split="train", token=HF_TOKEN)
    df = dataset.to_pandas()
    df = df.dropna(subset=['title', 'ingredients', 'instructions']).reset_index(drop=True)
    df_sample = df.head(30).copy() # Gunakan sampel untuk efisiensi resource cloud gratisan
    
    # Load Model Zero-Shot untuk Fitur 3 (Klasifikasi)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    kandidat_label = ["makanan berkuah sup soto", "makanan gorengan kering tumis", "kue makanan manis dessert"]
    
    kategori_terdeteksi = []
    for index, row in df_sample.iterrows():
        teks_konteks = f"{row['title']} {row['ingredients']}"
        hasil = classifier(teks_konteks, candidate_labels=kandidat_label)
        label_teratas = hasil['labels'][0]
        if "berkuah" in label_teratas: kategori = "Berkuah"
        elif "gorengan" in label_teratas: kategori = "Goreng/Kering"
        else: kategori = "Kue & Manis"
        kategori_terdeteksi.append(kategori)
    df_sample['kategori_ai'] = kategori_terdeteksi
    
    # Load Model Embedding untuk Fitur 1 (Pencarian Semantik)
    model_embed = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    df_sample['clean_ingredients'] = df_sample['ingredients'].astype(str).str.replace('\n', ' ', regex=False)
    df_sample['clean_ingredients'] = df_sample['clean_ingredients'].replace(['nan', '', 'None'], 'tidak ada bahan')
    vektor_bahan = model_embed.encode(df_sample['clean_ingredients'].tolist(), show_progress_bar=False)
    
    # Build FAISS
    dimensi = vektor_bahan.shape[1]
    index_faiss = faiss.IndexFlatL2(dimensi)
    index_faiss.add(np.array(vektor_bahan))
    
    return df_sample, model_embed, index_faiss

with st.spinner("🤖 Sedang menginisialisasi AI dan memuat database resep... (Mohon tunggu sebentar)"):
    df_sample, model_embed, index_faiss = load_models_and_data()

# ==========================================
# 2. LOGIKA FITUR 4 (PORSI & ALERGI)
# ==========================================
def sesuaikan_porsi_dan_alergi(teks_bahan, faktor_porsi, list_alergi=[]):
    if list_alergi:
        for alergi in list_alergi:
            if alergi.strip().lower() in teks_bahan.lower():
                return None
    def ganti_angka(match):
        angka = float(match.group())
        return str(round(angka * faktor_porsi, 2))
    return re.sub(r'\d+(\.\d+)?', ganti_angka, teks_bahan)

# ==========================================
# 3. SIDEBAR PANEL (INPUT USER)
# ==========================================
st.sidebar.header("🎛️ Panel Kontrol Pencarian")

kueri_bahan = st.sidebar.text_input("🛒 Bahan sisa di kulkas:", placeholder="Contoh: ayam dan cabai")
porsi_diinginkan = st.sidebar.slider("👥 Kelipatan Porsi Masakan:", min_value=1, max_value=10, value=1)
input_alergi = st.sidebar.text_input("⚠️ Bahan Alergi (Pisahkan dengan koma):", placeholder="Contoh: kecap, udang")
list_alergi = [x.strip() for x in input_alergi.split(",")] if input_alergi else []

kategori_filter = st.sidebar.selectbox("🍽️ Filter Jenis Makanan (AI):", ["Semua", "Berkuah", "Goreng/Kering", "Kue & Manis"])

# ==========================================
# 4. MAIN DASHBOARD DISPLAY & SEARCH LOGIC
# ==========================================
if st.sidebar.button("🚀 Cari Resep Sekarang", type="primary"):
    if kueri_bahan:
        st.subheader(f"🔍 Hasil Analisis Pencarian untuk: '{kueri_bahan}'")
        
        # Jalankan FAISS Search Semantik
        kueri_vektor = model_embed.encode([kueri_bahan])
        jarak, indeks = index_faiss.search(np.array(kueri_vektor), len(df_sample))
        
        hasil_ditemukan = 0
        
        # Tampilkan hasil dalam bentuk grid/kolom Streamlit
        for idx in indeks[0]:
            resep = df_sample.iloc[idx]
            
            # Filter Kategori AI
            if kategori_filter != "Semua" and resep['kategori_ai'] != kategori_filter:
                continue
                
            # Filter Alergi & Hitung Porsi
            bahan_termodifikasi = sesuaikan_porsi_dan_alergi(resep['ingredients'], porsi_diinginkan, list_alergi)
            if bahan_termodifikasi is None:
                continue
                
            hasil_ditemukan += 1
            
            # Tampilan Card Resep Masakan
            with st.expander(f"⭐ REKOMENDASI #{hasil_ditemukan}: {resep['title'].upper()} ({resep['kategori_ai']})", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**📋 Bahan-Bahan (Dikalikan {porsi_diinginkan}x):**")
                    # Tampilkan baris baru dengan rapi
                    st.text(bahan_termodifikasi)
                with col2:
                    st.markdown("**📖 Langkah Memasak:**")
                    st.text(resep['instructions'])
            
            if hasil_ditemukan == 2: # Batasi 2 hasil terbaik
                break
                
        if hasil_ditemukan == 0:
            st.error("❌ Maaf, tidak ada resep yang cocok dengan kriteria atau filter alergi Anda.")
    else:
        st.warning("⚠️ Silakan masukkan bahan makanan terlebih dahulu di kolom sidebar!")
else:
    # Tampilan awal Dashboard sebelum tombol ditekan
    st.info("💡 Petunjuk: Isi bahan makanan yang Anda miliki di sidebar sebelah kiri, sesuaikan porsi dan pantangan alergi Anda, lalu klik tombol 'Cari Resep Sekarang'!")
    
    # Menampilkan visualisasi data mini di Dashboard
    st.subheader("📊 Statistik Koleksi Resep Saat Ini (Analisis Otomatis AI)")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_sample[['title', 'kategori_ai']].rename(columns={'title': 'Nama Masakan', 'kategori_ai': 'Kategori AI'}), use_container_width=True)
    with col2:
        # Menampilkan jumlah menu per kategori
        kategori_counts = df_sample['kategori_ai'].value_counts()
        st.bar_chart(kategori_counts)