import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from huggingface_hub import login

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="AI Culinary Assistant", page_icon="🍳", layout="wide")

st.title("🍳 AI Culinary Assistant Dashboard")
st.markdown("Cari resep masakan Indonesia berdasarkan bahan sisa menggunakan Kecerdasan Buatan (NLP & Vector Search).")
st.divider()

# ==========================================
# 2. SISTEM MEMUAT DATA & MODEL (CACHED)
# ==========================================
@st.cache_resource
def load_models_and_data():
    # Otentikasi Environment
    os.environ["HF_TOKEN"] = HF_TOKEN
    try:
        login(token=HF_TOKEN)
    except:
        pass

    # A. Memuat Dataset
    # Mengunduh dataset resep masakan Indonesia
    try:
        dataset = load_dataset("junwatu/indonesian-recipes", split="train")
        df = dataset.to_pandas()
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return None, None, None
    
    # Deteksi otomatis nama kolom instruksi (antisipasi perubahan nama kolom di dataset)
    kolom_instruksi = 'instructions'
    for col in ['steps', 'directions', 'cara', 'langkah', 'instructions']:
        if col in df.columns:
            kolom_instruksi = col
            break
    
    if kolom_instruksi != 'instructions':
        df = df.rename(columns={kolom_instruksi: 'instructions'})

    # Membersihkan Data (PENTING: Mengubah NaN/Float menjadi String)
    df['ingredients'] = df['ingredients'].astype(str).replace(['nan', 'None', ''], 'bahan tidak disebutkan')
    df['title'] = df['title'].astype(str)
    df['instructions'] = df['instructions'].astype(str)
    
    # Mengambil 30 data pertama untuk stabilitas RAM server
    df_sample = df.head(30).copy()
    
    # B. Fitur Klasifikasi (Zero-Shot) - Menggunakan model ringan DistilBERT
    # Membantu mengategorikan jenis masakan secara otomatis
    classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
    labels = ["makanan berkuah", "makanan gorengan", "kue dessert manis"]
    
    kategori_list = []
    for _, row in df_sample.iterrows():
        konteks = f"{row['title']} {row['ingredients']}"
        res = classifier(konteks, candidate_labels=labels)
        top_label = res['labels'][0]
        if "berkuah" in top_label: kategori = "Berkuah"
        elif "gorengan" in top_label: kategori = "Goreng/Kering"
        else: kategori = "Kue & Manis"
        kategori_list.append(kategori)
    
    df_sample['kategori_ai'] = kategori_list
    
    # C. Fitur Pencarian Semantik (FAISS)
    # Menggunakan model embedding multibahasa
    model_embed = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    df_sample['clean_text'] = df_sample['ingredients'].str.replace('\n', ' ', regex=False)
    embeddings = model_embed.encode(df_sample['clean_text'].tolist(), show_progress_bar=False)
    
    # Membangun Index FAISS untuk pencarian cepat
    dimensi = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimensi)
    index.add(np.array(embeddings))
    
    return df_sample, model_embed, index

# Menjalankan proses loading dengan indikator visual
with st.spinner("🤖 Menginisialisasi AI dan memuat database... Harap tunggu 1-2 menit."):
    df_sample, model_embed, index_faiss = load_models_and_data()

# ==========================================
# 3. FUNGSI LOGIKA (PORSI & ALERGI)
# ==========================================
def process_recipe(ingredients, porsi_factor, allergens_list):
    # Cek Filter Alergi
    if allergens_list:
        for a in allergens_list:
            if a.strip().lower() in ingredients.lower():
                return None # Skip resep ini
    
    # Fungsi Regex untuk mengalikan angka takaran porsi
    def multiply(match):
        try:
            val = float(match.group())
            return str(round(val * porsi_factor, 2))
        except:
            return match.group()
    
    return re.sub(r'\d+(\.\d+)?', multiply, ingredients)

# ==========================================
# 4. ANTARMUKA SIDEBAR (INPUT)
# ==========================================
st.sidebar.header("🎛️ Pengaturan Pencarian")
query = st.sidebar.text_input("🛒 Bahan sisa yang Anda miliki:", placeholder="Misal: ayam, telur, cabai")
porsi = st.sidebar.slider("👥 Skala Porsi (Kelipatan):", 1, 10, 1)
alergi_input = st.sidebar.text_input("⚠️ Pantangan Alergi (Pisahkan dengan koma):", placeholder="Misal: udang, kecap")
filter_kat = st.sidebar.selectbox("🍽️ Filter Jenis Masakan (AI):", ["Semua", "Berkuah", "Goreng/Kering", "Kue & Manis"])

# Memproses input alergi menjadi list
alergi_list = [x.strip() for x in alergi_input.split(",")] if alergi_input else []

# ==========================================
# 5. HALAMAN UTAMA & LOGIKA PENCARIAN
# ==========================================
if st.sidebar.button("🚀 Cari Resep Sekarang", type="primary"):
    if query and df_sample is not None:
        st.subheader(f"🔍 Hasil Pencarian untuk: '{query}'")
        
        # Pencarian kemiripan makna menggunakan FAISS
        q_vec = model_embed.encode([query])
        dist, indices = index_faiss.search(np.array(q_vec), len(df_sample))
        
        found_count = 0
        for idx in indices[0]:
            row = df_sample.iloc[idx]
            
            # 1. Filter berdasarkan Kategori AI
            if filter_kat != "Semua" and row['kategori_ai'] != filter_kat:
                continue
            
            # 2. Proses Logika Porsi & Alergi
            mod_ingredients = process_recipe(row['ingredients'], porsi, alergi_list)
            
            if mod_ingredients:
                found_count += 1
                with st.expander(f"⭐ REKOMENDASI #{found_count}: {row['title'].upper()} ({row['kategori_ai']})", expanded=True):
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.write("**📋 Bahan-Bahan (Porsi x{}):**".format(porsi))
                        st.text(mod_ingredients)
                    with col_right:
                        st.write("**📖 Cara Memasak:**")
                        st.text(row['instructions'])
                
                # Batasi hanya menampilkan 3 hasil terbaik untuk efisiensi
                if found_count == 3: 
                    break
        
        if found_count == 0:
            st.error("❌ Tidak ditemukan resep yang cocok dengan kriteria atau batasan alergi Anda.")
    else:
        st.warning("⚠️ Masukkan bahan masakan yang Anda miliki di panel sidebar!")
else:
    # Tampilan awal Dashboard
    st.info("💡 Masukkan bahan makanan di sidebar kiri untuk mendapatkan rekomendasi resep berbasis AI.")
    
    if df_sample is not None:
        st.subheader("📊 Statistik Menu Tersedia (Analisis AI)")
        ca, cb = st.columns(2)
        with ca:
            st.dataframe(df_sample[['title', 'kategori_ai']].rename(columns={'title': 'Nama Resep', 'kategori_ai': 'Kategori'}), use_container_width=True, hide_index=True)
        with cb:
            st.markdown("**Distribusi Jenis Masakan**")
            st.bar_chart(df_sample['kategori_ai'].value_counts())
