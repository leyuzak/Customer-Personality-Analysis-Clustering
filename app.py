import joblib
import pandas as pd
import streamlit as st

# ---------------- config ----------------
APP_TITLE = "customer segmentation (kmeans)"
MODEL_PATH = "kmeans_pipeline.joblib"

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ---------------- minimal ui / css ----------------
st.markdown(
    """
<style>
section[data-testid="stSidebar"] {display:none !important;}
.block-container {max-width: 1100px; padding-top: 2rem;}
h1,h2,h3,label,p,span,div {letter-spacing:0.2px;}
.stButton>button {
    border-radius:14px;
    padding:0.7rem 1.1rem;
    font-weight:700;
}
div[data-testid="stDataFrame"] {
    border-radius:14px;
    overflow:hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- load model ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"model yüklenemedi: {e}")
    st.stop()

# ---------------- ui ----------------
st.markdown(f"## {APP_TITLE}")

tabs = st.tabs(["manuel", "csv"])

# ========== MANUAL ==========
with tabs[0]:
    st.markdown("#### tek müşteri girdisi")

    # eğitimden sonra pipeline'ın beklediği kolonlar
    feature_cols = pipeline.feature_names_in_

    left, right = st.columns(2, gap="large")
    values = {}

    for i, col in enumerate(feature_cols):
        target = left if i % 2 == 0 else right

        values[col] = target.number_input(
            col.lower(),
            value=0.0,
            step=0.1,
            format="%.6f",
        )

    df = pd.DataFrame([values])

    st.markdown("##### girdi")
    st.dataframe(df, use_container_width=True)

    if st.button("segmenti tahmin et", type="primary"):
        try:
            cluster = int(pipeline.predict(df)[0])
            st.success(f"segment: **{cluster}**")
        except Exception as e:
            st.error(f"hata: {e}")

# ========== CSV ==========
with tabs[1]:
    st.markdown("#### csv ile tahmin")
    st.caption("csv kolonları eğitimdeki kolonlarla birebir aynı olmalı.")

    file = st.file_uploader("csv yükle", type=["csv"])

    if file is not None:
        try:
            df = pd.read_csv(file)

            missing = [c for c in pipeline.feature_names_in_ if c not in df.columns]
            if missing:
                st.error(f"eksik kolonlar: {missing}")
                st.stop()

            df = df[pipeline.feature_names_in_]

            st.markdown("##### önizleme")
            st.dataframe(df.head(50), use_container_width=True)

            if st.button("tahmin et", type="primary"):
                df["cluster"] = pipeline.predict(df)
                st.success("tamamlandı")

                st.dataframe(df.head(100), use_container_width=True)

                st.download_button(
                    "sonuç csv indir",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="customer_segments.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"hata: {e}")
