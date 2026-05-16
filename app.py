import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Recipe Menu AI",
    page_icon="🍜",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.stTextInput input {
    border-radius: 12px;
}

.recipe-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e1e1e;
    margin-bottom: 15px;
    border: 1px solid #333;
}

.recipe-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 10px;
}

.recipe-desc {
    color: #cccccc;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("recipes.csv")

        # Pastikan kolom tersedia
        required_columns = ["recipe", "description"]

        for col in required_columns:
            if col not in df.columns:
                st.error(f"Kolom '{col}' tidak ditemukan di recipes.csv")
                st.stop()

        # Hapus data kosong
        df = df.dropna(subset=["recipe", "description"])

        return df

    except FileNotFoundError:
        st.error("File recipes.csv tidak ditemukan.")
        st.stop()

    except Exception as e:
        st.error(f"Error membaca data: {e}")
        st.stop()


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# CREATE EMBEDDINGS
# =========================
@st.cache_data(show_spinner=False)
def create_embeddings(recipe_texts):

    embeddings = model.encode(
        recipe_texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embeddings


# =========================
# INITIALIZATION
# =========================
with st.spinner("Memuat data dan model AI..."):
    df = load_data()
    model = load_model()
    embeddings = create_embeddings(df["recipe"].tolist())


# =========================
# HEADER
# =========================
st.title("🍜 Recipe Menu AI")
st.caption("Cari rekomendasi resep makanan dengan AI semantic search")

# =========================
# SEARCH INPUT
# =========================
query = st.text_input(
    "Cari resep makanan",
    placeholder="Contoh: ayam pedas, mie kuah, nasi goreng..."
)

# =========================
# SEARCH PROCESS
# =========================
if query:

    with st.spinner("Mencari resep terbaik..."):

        # Encode query
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True
        )

        # Hitung similarity
        similarities = cosine_similarity(
            query_embedding,
            embeddings
        )[0]

        # Ambil top 5
        top_indices = np.argsort(similarities)[-5:][::-1]

        st.subheader("Hasil Pencarian")

        for idx in top_indices:

            recipe_name = df.iloc[idx]["recipe"]
            recipe_desc = df.iloc[idx]["description"]
            score = similarities[idx]

            st.markdown(f"""
            <div class="recipe-card">
                <div class="recipe-title">
                    {recipe_name}
                </div>

                <div class="recipe-desc">
                    {recipe_desc}
                </div>

                <br>

                <b>Similarity:</b> {score:.2f}
            </div>
            """, unsafe_allow_html=True)

# =========================
# DEFAULT DISPLAY
# =========================
else:

    st.subheader("🍽️ Rekomendasi Menu")

    sample_data = df.head(6)

    cols = st.columns(2)

    for i, (_, row) in enumerate(sample_data.iterrows()):

        with cols[i % 2]:

            st.markdown(f"""
            <div class="recipe-card">
                <div class="recipe-title">
                    {row['recipe']}
                </div>

                <div class="recipe-desc">
                    {row['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.divider()

st.caption("Powered by Streamlit + Sentence Transformers")
