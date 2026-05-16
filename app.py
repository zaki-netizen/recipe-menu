import streamlit as st
import pandas as pd
import numpy as np

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Indonesian Recipe AI",
    page_icon="🍜",
    layout="wide"
)

# ==========================================
# CHECK TOKEN
# ==========================================

if "HF_TOKEN" not in st.secrets:
    st.error("HF_TOKEN belum ditambahkan di Streamlit Secrets.")
    st.stop()

HF_TOKEN = st.secrets["HF_TOKEN"]

# ==========================================
# CSS
# ==========================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.recipe-card {
    background: #1f1f1f;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 16px;
    border: 1px solid #333;
}

.recipe-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 10px;
}

.small {
    color: #cccccc;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data(show_spinner=False)
def load_recipes():

    dataset = load_dataset(
        "junwatu/indonesian-recipes",
        split="train[:1000]",
        token=HF_TOKEN
    )

    return dataset.to_pandas()

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

# ==========================================
# PREPARE TEXT
# ==========================================

@st.cache_data(show_spinner=False), hash_funcs={pd.DataFrame: lambda _: None}
def prepare_text(df):

    texts = []

    for _, row in df.iterrows():

        ingredients = " ".join(row["ingredients"])
        steps = " ".join(row["steps"])

        combined = f"""
        {row['title']}
        {ingredients}
        {steps}
        """

        texts.append(combined)

    return texts

# ==========================================
# CREATE EMBEDDINGS
# ==========================================

@st.cache_resource
def create_embeddings(texts):

    return model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

# ==========================================
# INITIALIZATION
# ==========================================

with st.spinner("Loading AI & dataset..."):

    df = load_recipes()

    model = load_model()

    combined_texts = prepare_text(df)

    embeddings = create_embeddings(combined_texts)

# ==========================================
# HEADER
# ==========================================

st.title("🍜 Indonesian Recipe AI")

st.caption(
    "Cari resep makanan Indonesia dengan semantic search AI"
)

# ==========================================
# SEARCH BOX
# ==========================================

query = st.text_input(
    "Cari makanan",
    placeholder="contoh: ayam pedas, mie goreng, soto..."
)

# ==========================================
# SEARCH
# ==========================================

if query:

    with st.spinner("Mencari resep..."):

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True
        )

        similarities = cosine_similarity(
            query_embedding,
            embeddings
        )[0]

        top_indices = np.argsort(similarities)[-5:][::-1]

        st.subheader("Hasil Pencarian")

        for idx in top_indices:

            row = df.iloc[idx]

            st.markdown(f"""
            <div class="recipe-card">

                <div class="recipe-title">
                    {row['title']}
                </div>

                <div class="small">
                    Similarity Score:
                    {similarities[idx]:.2f}
                </div>

                <br>

                <b>🧂 Bahan:</b>

                <ul>
                    {''.join([f"<li>{x}</li>" for x in row['ingredients'][:8]])}
                </ul>

                <b>👨‍🍳 Langkah:</b>

                <ol>
                    {''.join([f"<li>{x}</li>" for x in row['steps'][:5]])}
                </ol>

            </div>
            """, unsafe_allow_html=True)

# ==========================================
# DEFAULT VIEW
# ==========================================

else:

    st.subheader("🍽️ Sample Recipes")

    sample_df = df.sample(6)

    cols = st.columns(2)

    for i, (_, row) in enumerate(sample_df.iterrows()):

        with cols[i % 2]:

            st.markdown(f"""
            <div class="recipe-card">

                <div class="recipe-title">
                    {row['title']}
                </div>

                <div class="small">
                    {len(row['ingredients'])} bahan
                    •
                    {len(row['steps'])} langkah
                </div>

            </div>
            """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Powered by Streamlit + HuggingFace + Sentence Transformers"
)
