# agglomerative.py
import io
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go

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
    return df

def prepare_scaled_matrix(df, feature_cols):
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def run_agglomerative(X_scaled, n_clusters, linkage="ward"):
    """
    Jalankan AgglomerativeClustering dengan linkage='ward' (menggunakan Euclidean).
    """
    ac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = ac.fit_predict(X_scaled)
    return labels

def compute_scores_over_k_agglom(X_scaled, k_min=2, k_max=10, linkage="ward"):
    """
    Hitung silhouette_score dan davies_bouldin_score untuk k dari k_min..k_max.
    Returns: ks, silhouettes, davies
    """
    ks = []
    silhouettes = []
    davies = []
    for k in range(k_min, k_max + 1):
        try:
            labels = run_agglomerative(X_scaled, k, linkage=linkage)
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
    st.title("Metode Agglomerative Clustering — Analisa BMI / Berat / Tinggi")
    st.markdown("""
    Gunakan **Agglomerative Clustering** menggunakan **linkage='ward'** untuk mengelompokkan data anak berdasarkan kolom:
    **BMI**, **Berat**, dan **Tinggi**.
    """)

    uploaded = st.file_uploader("📂 Unggah file data (Excel/CSV)", type=["xlsx", "xls", "csv"])
    if uploaded is None:
        st.info("Silakan unggah file untuk memulai analisis.")
        return

    df = read_data(uploaded)
    st.subheader("Preview Data")
    st.dataframe(df.head(15), use_container_width=True)

    # -----------------------
    # Pilih fitur
    # -----------------------
    st.subheader("⚙️ Pilih Fitur yang Digunakan untuk Clustering")
    available_cols = [c for c in ["Berat", "Tinggi", "BMI"] if c in df.columns]
    feature_cols = st.multiselect("Pilih satu atau lebih fitur:", options=available_cols, default=available_cols)
    if not feature_cols:
        st.error("Pilih minimal satu fitur (Berat / Tinggi / BMI).")
        return

    st.markdown(f"**Fitur yang digunakan:** {', '.join(feature_cols)}")

    # Bersihkan data
    df_work = df.dropna(subset=feature_cols).copy()
    if df_work.empty:
        st.error("Tidak ada data valid setelah pembersihan NA.")
        return

    # scaling
    X_scaled, scaler = prepare_scaled_matrix(df_work, feature_cols)

    # -----------------------
    # Evaluasi over k: silhouette & davies_bouldin
    # -----------------------
    k_max = 10 if X_scaled.shape[0] >= 10 else max(3, X_scaled.shape[0] - 1)
    ks, silhouettes, davies = compute_scores_over_k_agglom(X_scaled, k_min=2, k_max=k_max, linkage="ward")

    # build eval_df so it exists for display & export
    eval_df = pd.DataFrame({"k": ks, "silhouette": silhouettes, "davies_bouldin": davies})

    st.subheader("🔎 Evaluasi Cluster untuk Berbagai Nilai k")
    # Plot Silhouette
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(x=ks, y=silhouettes, mode="lines+markers", name="Silhouette"))
    fig_sil.update_layout(title="Silhouette Score per k (lebih tinggi lebih baik)",
                          xaxis_title="Jumlah Cluster (k)", yaxis_title="Silhouette Score")
    st.plotly_chart(fig_sil, use_container_width=True)

    # Plot Davies-Bouldin
    fig_db = go.Figure()
    fig_db.add_trace(go.Scatter(x=ks, y=davies, mode="lines+markers", name="Davies–Bouldin"))
    fig_db.update_layout(title="Davies–Bouldin Index per k (lebih rendah lebih baik)",
                         xaxis_title="Jumlah Cluster (k)", yaxis_title="Davies–Bouldin Index")
    st.plotly_chart(fig_db, use_container_width=True)

    # -----------------------
    # Jalankan fixed k=3
    # -----------------------
    fixed_k = 3
    if X_scaled.shape[0] < fixed_k:
        st.error(f"Jumlah sampel ({X_scaled.shape[0]}) kurang dari k={fixed_k}.")
        return

    labels_3 = run_agglomerative(X_scaled, fixed_k, linkage="ward")
    df_work["Cluster"] = labels_3

    # Map cluster -> label berdasarkan rata-rata BMI (jika ada), otherwise first feature
    ref_col = "BMI" if "BMI" in df_work.columns else feature_cols[0]
    cluster_order = df_work.groupby("Cluster")[ref_col].mean().sort_values().index.tolist()
    if len(cluster_order) >= 3:
        cluster_to_label = {
            cluster_order[0]: "Stunting",
            cluster_order[1]: "Normal",
            cluster_order[2]: "Overweight"
        }
    else:
        cluster_to_label = {c: f"Cluster_{c}" for c in np.unique(labels_3)}
    df_work["Label"] = df_work["Cluster"].map(cluster_to_label)

    # compute metrics k=3
    sil_3, db_3 = None, None
    try:
        if len(np.unique(labels_3)) > 1:
            sil_3 = silhouette_score(X_scaled, labels_3)
            db_3 = davies_bouldin_score(X_scaled, labels_3)
    except Exception:
        sil_3, db_3 = None, None

    # -----------------------
    # Ringkasan hasil
    # -----------------------
    st.subheader("📊 Ringkasan Hasil (k=3)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah data", len(df_work))
    c2.metric("Silhouette (k=3)", f"{sil_3:.3f}" if sil_3 is not None else "—")
    c3.metric("Davies–Bouldin (k=3)", f"{db_3:.3f}" if db_3 is not None else "—")

    # Statistik per label
    st.subheader("Statistik per Label")
    agg = df_work.groupby("Label")[feature_cols].mean().round(2)
    agg["Jumlah Anak"] = df_work.groupby("Label").size()
    st.dataframe(agg.reset_index(), use_container_width=True)

    # Visualisasi (2D scatter)
    st.subheader("🗺️ Visualisasi Clustering")
    x_axis = feature_cols[0]
    y_axis = feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]
    fig = px.scatter(df_work, x=x_axis, y=y_axis, color="Label", hover_data=df_work.columns,
                     title=f"Sebaran Data: {x_axis} vs {y_axis}")
    st.plotly_chart(fig, use_container_width=True)

    # Data hasil preview
    st.subheader("📝 Data Hasil")
    st.dataframe(df_work, use_container_width=True)

    # -----------------------
    # Unduh hasil
    # -----------------------
    st.subheader("⬇️ Unduh Hasil Analisa Agglomerative Clustering")
    csv_buf = io.StringIO()
    df_work.to_csv(csv_buf, index=False)

    xls_buf = io.BytesIO()
    with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as writer:
        df_work.to_excel(writer, index=False, sheet_name="Hasil")
        eval_df.to_excel(writer, index=False, sheet_name="Metrics_over_k")
        pd.DataFrame({"k": ks, "silhouette": silhouettes, "davies_bouldin": davies}).to_excel(writer, index=False, sheet_name="Metrics_over_k")
        pd.DataFrame({"k": ks}).to_excel(writer, index=False, sheet_name="Ks")
        pd.DataFrame({
            "metric": ["silhouette_k3", "davies_bouldin_k3"],
            "value": [sil_3 if sil_3 is not None else np.nan, db_3 if db_3 is not None else np.nan]
        }).to_excel(writer, index=False, sheet_name="Metrics_k3")
    xls_data = xls_buf.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📄 Unduh CSV", data=csv_buf.getvalue(), file_name="hasil_agglomerative.csv", mime="text/csv", use_container_width=True)
    with col2:
        st.download_button("📘 Unduh Excel", data=xls_data, file_name="hasil_agglomerative.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

if __name__ == "__main__":
    show()
