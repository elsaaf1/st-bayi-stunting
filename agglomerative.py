# agglomerative.py
import io
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import numpy as np


# -----------------------
# Helper functions
# -----------------------
@st.cache_data
def read_data(uploaded):
    """Membaca file CSV atau Excel dan membersihkan kolom tanggal lahir agar tidak tampil waktu."""
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    # Format kolom tanggal lahir
    for col in df.columns:
        if "tgl" in col.lower() and "lahir" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col]).dt.date
            except Exception:
                pass
    return df


def fixed_mapping(clusters):
    """Pemetaan fixed cluster → label."""
    mapping = {0: "Stunting", 1: "Normal", 2: "Overweight"}
    return [mapping.get(c, f"Lainnya_{c}") for c in clusters]


def compute_scores_over_k(X_scaled, linkage_type="ward", k_min=2, k_max=10):
    """Hitung silhouette dan davies-bouldin untuk berbagai k."""
    ks, silhouettes, davies = [], [], []
    for k in range(k_min, k_max + 1):
        try:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage_type)
            labels = model.fit_predict(X_scaled)
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
    st.title("Metode Agglomerative — Analisa BMI / Berat / Tinggi")
    st.markdown("""
    Gunakan **Agglomerative Hierarchical Clustering** untuk mengelompokkan data anak berdasarkan kolom:
    **BMI**, **Berat**, dan **Tinggi**.
    """)

    uploaded = st.file_uploader("📂 Unggah file data (Excel/CSV)", type=["xlsx", "xls", "csv"])
    if uploaded is None:
        st.info("Silakan unggah file untuk memulai analisis.")
        return

    df = read_data(uploaded)
    st.subheader("Preview Data")
    st.dataframe(df.head(15), use_container_width=True)

    # Pengaturan Agglomerative di bawah Preview Data
    st.subheader("⚙️ Pengaturan Agglomerative")
    linkage_type = st.selectbox(
        "Pilih jenis linkage",
        ["ward", "average", "complete", "single"],
        help=(
            "- **ward**: minimalkan varian dalam cluster (paling umum, hanya untuk data numerik)\n"
            "- **average**: jarak rata-rata antar semua titik di dua cluster\n"
            "- **complete**: jarak maksimum antar titik di dua cluster\n"
            "- **single**: jarak minimum antar titik di dua cluster (cenderung menghasilkan chaining)"
        )
    )

    # Deteksi otomatis kolom fitur
    expected_cols = ["Berat", "Tinggi", "BMI"]
    feature_cols = [col for col in expected_cols if col in df.columns]
    if len(feature_cols) < 2:
        st.error("❌ Kolom yang dibutuhkan tidak lengkap. Pastikan ada kolom 'Berat', 'Tinggi', dan 'BMI'.")
        return

    # Bersihkan data
    df_work = df.dropna(subset=feature_cols).copy()
    if df_work.empty:
        st.error("Tidak ada data valid setelah pembersihan NA.")
        return

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_work[feature_cols])

    # -----------------------
    # Evaluasi berbagai k
    # -----------------------
    st.subheader(f"🔎 Evaluasi Cluster untuk Berbagai Nilai k — Linkage: **{linkage_type}**")
    k_max = 10 if len(df_work) >= 10 else max(3, len(df_work) - 1)
    ks, silhouettes, davies = compute_scores_over_k(X_scaled, linkage_type=linkage_type, k_min=2, k_max=k_max)

    # Plot Silhouette
    fig_sil = px.line(
        x=ks, y=silhouettes, markers=True,
        title=f"Silhouette Score per k (lebih tinggi lebih baik) — Linkage: {linkage_type}"
    )
    fig_sil.update_xaxes(title="Jumlah Cluster (k)")
    fig_sil.update_yaxes(title="Silhouette Score")
    st.plotly_chart(fig_sil, use_container_width=True)

    # Plot Davies–Bouldin
    fig_db = px.line(
        x=ks, y=davies, markers=True,
        title=f"Davies–Bouldin Index per k (lebih rendah lebih baik) — Linkage: {linkage_type}"
    )
    fig_db.update_xaxes(title="Jumlah Cluster (k)")
    fig_db.update_yaxes(title="Davies–Bouldin Index")
    st.plotly_chart(fig_db, use_container_width=True)

    # -----------------------
    # Jalankan Agglomerative (k=3 fixed)
    # -----------------------
    st.subheader("📊 Hasil Clustering")
    fixed_k = 3
    try:
        model = AgglomerativeClustering(n_clusters=fixed_k, linkage=linkage_type)
        labels = model.fit_predict(X_scaled)
        df_work["Cluster"] = labels
        df_work["Label"] = fixed_mapping(labels)
    except Exception as e:
        st.error(f"Gagal menjalankan AgglomerativeClustering: {e}")
        return

    # Hitung metrik untuk k=3
    sil_3, db_3 = None, None
    try:
        if len(np.unique(labels)) > 1:
            sil_3 = silhouette_score(X_scaled, labels)
            db_3 = davies_bouldin_score(X_scaled, labels)
    except Exception:
        pass

    # Ringkasan hasil
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah data", len(df_work))
    with col2:
        st.metric("Silhouette", f"{sil_3:.3f}" if sil_3 is not None else "—")
    with col3:
        st.metric("Davies–Bouldin", f"{db_3:.3f}" if db_3 is not None else "—")

    # Statistik per Label
    st.subheader("📋 Statistik per Label")
    agg = df_work.groupby("Label")[feature_cols].mean().round(2)
    agg["Jumlah Anak"] = df_work.groupby("Label").size()
    st.dataframe(agg.reset_index(), use_container_width=True)

    # -----------------------
    # Visualisasi hasil
    # -----------------------
    st.subheader("🗺️ Visualisasi Clustering")
    x_axis = "BMI"
    y_axis = "Tinggi" if "Tinggi" in df_work.columns else feature_cols[0]
    fig = px.scatter(
        df_work, x=x_axis, y=y_axis, color="Label", hover_data=df_work.columns,
        title=f"Agglomerative Clustering — Linkage: {linkage_type}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Preview hasil
    st.subheader("📝 Data Hasil")
    st.dataframe(df_work, use_container_width=True)
    
    # -----------------------
    # Unduh hasil
    # -----------------------
    st.subheader("⬇️ Unduh Hasil Analisa Agglomerative")

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
            flex: 0 0 auto;  /* tombol tidak melar */
        }
        </style>
    """, unsafe_allow_html=True)

    # Siapkan buffer CSV & Excel
    csv_buf = io.StringIO()
    df_work.to_csv(csv_buf, index=False)

    xls_buf = io.BytesIO()
    with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as writer:
        df_work.to_excel(writer, index=False, sheet_name="Hasil")
        metrics_df = pd.DataFrame({
            "k": ks,
            "silhouette": silhouettes,
            "davies_bouldin": davies
        })
        metrics_df.to_excel(writer, index=False, sheet_name="Metrics_over_k")

    # Tampilkan tombol rata tengah dan sejajar
    st.markdown('<div class="download-buttons">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📄 Unduh Hasil CSV",
            data=csv_buf.getvalue(),
            file_name=f"hasil_agglomerative_{linkage_type}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        st.download_button(
            "📘 Unduh Hasil Excel",
            data=xls_buf.getvalue(),
            file_name=f"hasil_agglomerative_{linkage_type}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
