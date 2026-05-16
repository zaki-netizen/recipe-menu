import os
import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from huggingface_hub import login
import pandas as pd
import faiss
import numpy as np
import re

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="AI Culinary Assistant", page_icon="🍳", layout="wide")

st.title("🍳 AI Culinary Assistant Dashboard")
st.markdown("Cari resep masakan Indonesia berdasarkan bahan sisa menggunakan Kecerdasan Buatan.")
st.divider()

# ==========================================
# 2. SISTEM MEMUAT DATA & MODEL (CACHED)
# ==========================================
@st.cache_resource
def load_models_and_data():
    # Token Hugging Face
    HF_TOKEN = "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"
    
    # Otentikasi Global (Penting untuk Streamlit Cloud)
    os.environ["HF_TOKEN"] = HF_TOKEN
    try:
        login(token=HF_TOKEN)
    except:
        pass

    # A. Memuat Dataset
    # Mengunduh dataset resep masakan Indonesia
    dataset = load_dataset("junwatu/indonesian-recipes", split="train")
    df = dataset.to_pandas()
    
    # Deteksi kolom instruksi secara otomatis
    kolom_instruksi = 'instructions'
    for col in ['steps', 'directions', 'cara', 'langkah', 'instructions']:
        if col in df.columns:
            kolom_instruksi = col
            break
    
    if kolom_instruksi != 'instructions':
        df = df.rename(columns={kolom_instruksi: 'instructions'})

    # Pembersihan Data (Mencegah error Float/NaN)
    df['ingredients'] = df['ingredients'].astype(str).replace(['nan', 'None', ''], 'tidak ada bahan')
    df['title'] = df['title'].astype(str)
    df['instructions'] = df['instructions'].astype(str)
    
    # Gunakan 30 sampel data agar RAM tidak penuh di server gratis
    df_sample = df.head(30).copy()
    
    # B. Fitur Klasifikasi (Zero-Shot) - Menggunakan model ringan DistilBERT
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
    model_embed = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    df_sample['clean_text'] = df_sample['ingredients'].str.replace('\n', ' ', regex=False)
    embeddings = model_embed.encode(df_sample['clean_text'].tolist(), show_progress_bar=False)
    
    # Bangun Index FAISS
    dimensi = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimensi)
    index.add(np.array(embeddings))
    
    return df_sample, model_embed, index

# Tampilkan Spinner saat Loading
with st.spinner("🤖 Menginisialisasi AI... Harap tunggu sebentar."):
    df_sample, model_embed, index_faiss = load_models_and_data()

# ==========================================
# 3. FUNGSI LOGIKA (PORSI & ALERGI)
# ==========================================
def process_recipe(ingredients, porsi, allergens):
    # Cek Alergi
    if allergens:
        for a in allergens:
            if a.strip().lower() in ingredients.lower():
                return None
    
    # Kalkulasi Porsi (Regex)
    def multiply(match):
        val = float(match.group())
        return str(round(val * porsi, 2))
    
    return re.sub(r'\d+(\.\d+)?', multiply, ingredients)

# ==========================================
# 4. ANTARMUKA SIDEBAR
# ==========================================
st.sidebar.header("🎛️ Pengaturan")
query = st.sidebar.text_input("🛒 Bahan yang dimiliki:", placeholder="Misal: ayam, santan")
porsi = st.sidebar.slider("👥 Jumlah Porsi (Kelipatan):", 1, 10, 1)
alergi_input = st.sidebar.text_input("⚠️ Pantangan Alergi (Koma):", placeholder="Misal: kacang, kecap")
filter_kat = st.sidebar.selectbox("🍽️ Jenis Masakan:", ["Semua", "Berkuah", "Goreng/Kering", "Kue & Manis"])

alergi_list = [x.strip() for x in alergi_input.split(",")] if alergi_input else []

# ==========================================
# 5. HALAMAN UTAMA & HASIL
# ==========================================
if st.sidebar.button("🚀 Cari Resep", type="primary"):
    if query:
        # Search FAISS
        q_vec = model_embed.encode([query])
        D, I = index_faiss.search(np.array(q_vec), len(df_sample))
        
        found = 0
        for idx in I[0]:
            row = df_sample.iloc[idx]
            
            # Filter Kategori
            if filter_kat != "Semua" and row['kategori_ai'] != filter_kat:
                continue
            
            # Proses Porsi & Alergi
            mod_ingredients = process_recipe(row['ingredients'], porsi, alergi_list)
            
            if mod_ingredients:
                found += 1
                with st.expander(f"✅ {row['title'].upper()} ({row['kategori_ai']})"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Bahan (Disesuaikan):**")
                        st.text(mod_ingredients)
                    with c2:
                        st.write("**Cara Masak:**")
                        st.text(row['instructions'])
                
                if found == 3: break # Tampilkan 3 terbaik
        
        if found == 0:
            st.error("Tidak ditemukan resep yang cocok dengan kriteria Anda.")
    else:
        st.warning("Masukkan bahan masakan di sidebar!")
else:
    st.info("Gunakan panel di kiri untuk mulai mencari resep.")
    
    # Visualisasi Mini
    st.subheader("📊 Koleksi Resep Berdasarkan AI")
    col_a, col_b = st.columns(2)
    with col_a:
        st.dataframe(df_sample[['title', 'kategori_ai']], use_container_width=True)
    with col_b:
        st.bar_chart(df_sample['kategori_ai'].value_counts())
