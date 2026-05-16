import streamlit as st
import pandas as pd
import numpy as np

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Indonesian Recipe AI",
    page_icon="🍜",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.recipe-card {
    background-color: #1f1f1f;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 16px;
    border: 1px solid #333333;
}

.recipe-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 10px;
}

.recipe-section {
    margin-top: 12px;
    line-height: 1.7;
}

.small-text {
    color: #bdbdbd;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# CHECK HF TOKEN
# ======================================================

if "HF_TOKEN" not in st.secrets:
    st.error("HF_TOKEN belum ditambahkan di Streamlit Secrets.")
    st.stop()

HF_TOKEN = st.secrets["HF_TOKEN"]

# ======================================================
# LOAD DATASET
# ======================================================

@st.cache_data(show_spinner=False)
def load_recipes():

    dataset = load_dataset(
        "junwatu/indonesian-recipes",
        split="train[:2000]",
        token=HF_TOKEN
    )

    df = dataset.to_pandas()

    return df

# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model

# ======================================================
# PREPARE TEXT
# ======================================================

@st.cache_data(show_spinner=False)
def prepare_text(df):

    combined_texts = []

    for _, row in df.iterrows():

        ingredients = " ".join(row["ingredients"])
        steps = " ".join(row["steps"])

        text = f"""
        {row['title']}
        {ingredients}
        {steps}
        """

        combined_texts.append(text)

    return combined_texts

# ======================================================
# CREATE EMBEDDINGS
# ======================================================

@st.cache_data(show_spinner=False)
def create_embeddings(texts):

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embeddings

# ======================================================
# INITIALIZATION
# ======================================================

with st.spinner("Loading dataset & AI model..."):

    df = load_recipes()

    model = load_model()

    combined_texts = prepare_text(df)

    embeddings = create_embeddings(combined_texts)

# ======================================================
# HEADER
# ======================================================

st.title("🍜 Indonesian Recipe AI")

st.caption(
    "Cari resep makanan Indonesia menggunakan semantic search AI"
)

# ======================================================
# SEARCH INPUT
# ======================================================

query = st.text_input(
    "Cari resep makanan",
    placeholder="Contoh: ayam pedas, nasi goreng, mie kuah..."
)

# ======================================================
# SEARCH PROCESS
# ======================================================

if query:

    with st.spinner("Mencari resep terbaik..."):

        # Encode query
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True
        )

        # Similarity
        similarities = cosine_similarity(
            query_embedding,
            embeddings
        )[0]

        # Top 5 results
        top_indices = np.argsort(similarities)[-5:][::-1]

        st.subheader("Hasil Pencarian")

        for idx in top_indices:

            row = df.iloc[idx]

            title = row["title"]

            ingredients = row["ingredients"][:10]
            steps = row["steps"][:5]

            similarity_score = similarities[idx]

            st.markdown(f"""
            <div class="recipe-card">

                <div class="recipe-title">
                    {title}
                </div>

                <div class="recipe-section">
                    <b>🧂 Bahan:</b>
                    <br>
                    {'<br>'.join(ingredients)}
                </div>

                <div class="recipe-section">
                    <b>👨‍🍳 Langkah:</b>
                    <br>
                    {'<br>'.join(steps)}
                </div>

                <div class="recipe-section small-text">
                    Similarity Score: {similarity_score:.2f}
                </div>

            </div>
            """, unsafe_allow_html=True)

# ======================================================
# DEFAULT DISPLAY
# ======================================================

else:

    st.subheader("🍽️ Rekomendasi Resep")

    sample_df = df.sample(6)

    cols = st.columns(2)

    for i, (_, row) in enumerate(sample_df.iterrows()):

        with cols[i % 2]:

            st.markdown(f"""
            <div class="recipe-card">

                <div class="recipe-title">
                    {row['title']}
                </div>

                <div class="recipe-section small-text">
                    {len(row['ingredients'])} bahan
                    •
                    {len(row['steps'])} langkah
                </div>

            </div>
            """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.caption(
    "Powered by Streamlit + HuggingFace + Sentence Transformers"
)
