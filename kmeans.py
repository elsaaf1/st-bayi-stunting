# kmeans.py
import io
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# -----------------------
# Helper functions
# -----------------------
@st.cache_data
def read_data(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    # Format kolom tanggal lahir agar tidak menampilkan waktu
    for col in df.columns:
        if "tgl" in col.lower() and "lahir" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col]).dt.date
            except Exception:
                pass
    return df


def prepare_scaled_matrix(df, feature_cols):
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def fit_kmeans_on_matrix(X_scaled, n_clusters, random_state=42):
    """Menjalankan KMeans pada matrix ter-scaling, tanpa expose n_init."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    return labels, km, inertia


def fixed_mapping(clusters):
    """Pemetaan fixed cluster → label."""
    mapping = {0: "Stunting", 1: "Normal", 2: "Overweight"}
    return [mapping.get(c, f"Lainnya_{c}") for c in clusters]


def compute_elbow(X_scaled, k_max=10, random_state=42):
    """Hitung inertia untuk k=1..k_max (elbow plot)."""
    inertias = []
    ks = list(range(1, k_max + 1))
    for k in ks:
        try:
            _, km, inertia = fit_kmeans_on_matrix(X_scaled, k, random_state=random_state)
            inertias.append(inertia)
        except Exception:
            inertias.append(np.nan)
    return ks, inertias


def compute_scores_over_k(X_scaled, k_min=2, k_max=10, random_state=42):
    """Hitung silhouette & davies untuk k in [k_min..k_max]."""
    ks = []
    silhouettes = []
    davies = []
    for k in range(k_min, k_max + 1):
        try:
            labels, _, _ = fit_kmeans_on_matrix(X_scaled, k, random_state=random_state)
            # silhouette requires >1 cluster and at least one sample per cluster
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(X_scaled, labels)
                db = davies_bouldin_score(X_scaled, labels)
            else:
                sil, db = np.nan, np.nan
        except Exception:
            sil, db = np.nan, np.nan
        ks.append(k)
        silhouettes.append(sil)
        davies.append(db)
    return ks, silhouettes, davies


# -----------------------
# Streamlit Page
# -----------------------
def show():
    st.title("Metode K-Means — Analisa BMI / Berat / Tinggi")
    st.markdown("""
    Gunakan **K-Means** untuk mengelompokkan data anak berdasarkan kolom:
    **BMI**, **Berat**, dan **Tinggi**.
    """)
 
    uploaded = st.file_uploader("📂 Unggah file data (Excel/CSV)", type=["xlsx", "xls", "csv"])

    if uploaded is None:
        st.info("Silakan unggah file untuk memulai analisis.")
        return

    df = read_data(uploaded)
    st.subheader("Preview Data")
    st.dataframe(df.head(15), use_container_width=True)

    # Deteksi otomatis kolom fitur
    expected_cols = ["Berat", "Tinggi", "BMI"]
    feature_cols = [col for col in expected_cols if col in df.columns]

    if len(feature_cols) < 2:
        st.error("❌ Kolom yang dibutuhkan tidak lengkap. Pastikan ada kolom 'Berat', 'Tinggi', dan 'BMI'.")
        return

    # Bersihkan data (hanya baris yang punya semua fitur)
    df_work = df.dropna(subset=feature_cols).copy()
    if df_work.empty:
        st.error("Tidak ada data valid setelah pembersihan NA.")
        return

    # Persiapkan matriks ter-scaling
    X_scaled, scaler = prepare_scaled_matrix(df_work, feature_cols)

    # -----------------------
    # Elbow + Silhouette + Davies
    # -----------------------
    k_max = 10 if X_scaled.shape[0] >= 10 else max(3, X_scaled.shape[0] - 1)
    ks_elbow, inertias = compute_elbow(X_scaled, k_max=k_max)
    ks_scores, silhouettes, davies = compute_scores_over_k(X_scaled, k_min=2, k_max=k_max)

    st.subheader("🔎 Evaluasi Cluster untuk Berbagai Nilai k")

    # Elbow plot (inertia)
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=ks_elbow, y=inertias, mode="lines+markers", name="Inertia"))
    fig_elbow.update_layout(title="Elbow Plot (Inertia)", xaxis_title="Jumlah Cluster (k)", yaxis_title="Inertia")
    st.plotly_chart(fig_elbow, use_container_width=True)

    # Silhouette plot
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(x=ks_scores, y=silhouettes, mode="lines+markers", name="Silhouette Score"))
    fig_sil.update_layout(title="Silhouette Score per k (lebih tinggi lebih baik)",
                          xaxis_title="Jumlah Cluster (k)", yaxis_title="Silhouette Score")
    st.plotly_chart(fig_sil, use_container_width=True)

    # Davies–Bouldin plot
    fig_db = go.Figure()
    fig_db.add_trace(go.Scatter(x=ks_scores, y=davies, mode="lines+markers", name="Davies–Bouldin Index"))
    fig_db.update_layout(title="Davies–Bouldin Index per k (lebih rendah lebih baik)",
                         xaxis_title="Jumlah Cluster (k)", yaxis_title="Davies–Bouldin Index")
    st.plotly_chart(fig_db, use_container_width=True)

    # -----------------------
    # Jalankan KMeans fixed k=3
    # -----------------------
    fixed_k = 3
    if X_scaled.shape[0] < fixed_k:
        st.error(f"Jumlah sampel ({X_scaled.shape[0]}) kurang dari k={fixed_k}. Tidak dapat menjalankan K-Means.")
        return

    labels_3, km3, inertia3 = fit_kmeans_on_matrix(X_scaled, fixed_k)
    df_work["Cluster"] = labels_3
    df_work["Label"] = fixed_mapping(labels_3)

    # Hitung metrik untuk k=3
    sil_3, db_3 = None, None
    try:
        if len(np.unique(labels_3)) > 1:
            sil_3 = silhouette_score(X_scaled, labels_3)
            db_3 = davies_bouldin_score(X_scaled, labels_3)
    except Exception:
        sil_3, db_3 = None, None


    # Ringkasan hasil
    st.subheader("📊 Ringkasan Hasil")
    mapping_df = pd.DataFrame({"Cluster": [0, 1, 2], "Label": ["Stunting", "Normal", "Overweight"]})
    st.dataframe(mapping_df, use_container_width=True)

    # Tampilkan metrik k=3
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jumlah data", len(df_work))
    with col2:
        st.metric("Elbow (Inertia)", f"{inertia3:.2f}")
    with col3:
        st.metric("Silhouette", f"{sil_3:.3f}" if sil_3 is not None else "—")
    with col4:
        st.metric("Davies–Bouldin", f"{db_3:.3f}" if db_3 is not None else "—")

    # Statistik per Label
    st.subheader("Statistik per Label")
    agg = df_work.groupby("Label")[feature_cols].mean().round(2)
    agg["Jumlah Anak"] = df_work.groupby("Label").size()
    st.dataframe(agg.reset_index(), use_container_width=True)

    # Visualisasi k=3
    st.subheader("🗺️ Visualisasi Clustering")
    x_axis = "BMI"
    y_axis = "Tinggi" if "Tinggi" in df_work.columns else feature_cols[0]
    fig = px.scatter(df_work, x=x_axis, y=y_axis, color="Label", hover_data=df_work.columns,
                     title=f"Sebaran Data: {x_axis} vs {y_axis}")
    st.plotly_chart(fig, use_container_width=True)

    # Preview hasil
    st.subheader("📝 Data Hasil")
    st.dataframe(df_work, use_container_width=True)

        # -----------------------
    # Unduh hasil
    # -----------------------
    st.subheader("⬇️ Unduh Hasil Analisa K-Means")

    # CSS agar tombol rata tengah
    st.markdown("""
        <style>
        div.download-buttons {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;  /* jarak antar tombol */
            margin-top: 10px;
            margin-bottom: 20px;
        }
        div.download-buttons > div {
            flex: 0 0 auto;  /* agar tombol tidak melar */
        }
        </style>
    """, unsafe_allow_html=True)

    # Buffer CSV dan Excel
    csv_buf = io.StringIO()
    df_work.to_csv(csv_buf, index=False)

    xls_buf = io.BytesIO()
    with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as writer:
        df_work.to_excel(writer, index=False, sheet_name="Hasil")
        mapping_df.to_excel(writer, index=False, sheet_name="Mapping")
        metrics_df = pd.DataFrame({
            "k": ks_scores,
            "silhouette": silhouettes,
            "davies_bouldin": davies
        })
        metrics_df.to_excel(writer, index=False, sheet_name="Metrics_over_k")
        elbow_df = pd.DataFrame({"k": ks_elbow, "inertia": inertias})
        elbow_df.to_excel(writer, index=False, sheet_name="Elbow")

    # Tampilkan tombol di tengah dan sejajar
    st.markdown('<div class="download-buttons">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📄 Unduh Hasil CSV",
            data=csv_buf.getvalue(),
            file_name="hasil_kmeans_fixed.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        st.download_button(
            "📘 Unduh Hasil Excel",
            data=xls_buf.getvalue(),
            file_name="hasil_kmeans_fixed_with_metrics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
