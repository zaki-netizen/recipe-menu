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
# CHECK HF TOKEN
# ======================================================

if "HF_TOKEN" not in st.secrets:
    st.error("HF_TOKEN belum ditambahkan di Streamlit Secrets.")
    st.stop()

HF_TOKEN = st.secrets["HF_TOKEN"]

# ======================================================
# SIMPLE CSS
# ======================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stTextInput input {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD DATASET
# ======================================================

@st.cache_data(show_spinner=False)
def load_recipes():

    dataset = load_dataset(
        "junwatu/indonesian-recipes",
        split="train[:500]",
        token=HF_TOKEN
    )

    df = dataset.to_pandas()

    return df

# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

# ======================================================
# PREPARE SEARCH TEXT
# ======================================================

@st.cache_data(
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda _: None}
)
def prepare_text(df):

    texts = []

    for _, row in df.iterrows():

        ingredients = " ".join(
            map(str, row["ingredients"])
        )

        steps = " ".join(
            map(str, row["steps"])
        )

        combined = f"""
        {row['title']}
        {ingredients}
        {steps}
        """

        texts.append(combined)

    return texts

# ======================================================
# CREATE EMBEDDINGS
# ======================================================

@st.cache_resource
def create_embeddings(texts):

    return model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

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
    "Cari makanan",
    placeholder="contoh: ayam pedas, soto, mie goreng..."
)

# ======================================================
# SEARCH PROCESS
# ======================================================

if query:

    with st.spinner("Mencari resep terbaik..."):

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True
        )

        similarities = cosine_similarity(
            query_embedding,
            embeddings
        )[0]

        top_indices = np.argsort(
            similarities
        )[-5:][::-1]

        st.subheader("🔍 Hasil Pencarian")

        for idx in top_indices:

            row = df.iloc[idx]

            with st.container(border=True):

                st.subheader(row["title"])

                st.caption(
                    f"Similarity Score: "
                    f"{similarities[idx]:.2f}"
                )

                st.markdown("### 🧂 Bahan")

                for item in row["ingredients"][:8]:
                    st.write(f"• {item}")

                st.markdown("### 👨‍🍳 Langkah")

                for i, step in enumerate(
                    row["steps"][:5],
                    start=1
                ):
                    st.write(f"{i}. {step}")

# ======================================================
# DEFAULT DISPLAY
# ======================================================

else:

    st.subheader("🍽️ Sample Recipes")

    sample_df = df.sample(6)

    cols = st.columns(2)

    for i, (_, row) in enumerate(
        sample_df.iterrows()
    ):

        with cols[i % 2]:

            with st.container(border=True):

                st.subheader(row["title"])

                st.caption(
                    f"{len(row['ingredients'])} bahan • "
                    f"{len(row['steps'])} langkah"
                )

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.caption(
    "Powered by Streamlit + HuggingFace + Sentence Transformers"
)
