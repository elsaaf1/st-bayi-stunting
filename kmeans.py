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
    # Format kolom tanggal lahir agar tidak menampilkan waktu (opsional)
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


def fit_kmeans_on_matrix(X_scaled, n_clusters, random_state=42, n_init=10):
    """Menjalankan KMeans pada matrix ter-scaling, dengan n_init eksplisit."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    return labels, km, inertia


def compute_elbow(X_scaled, k_max=10, random_state=42, n_init=10):
    """Hitung inertia untuk k=1..k_max (elbow plot)."""
    inertias = []
    ks = list(range(1, k_max + 1))
    for k in ks:
        try:
            _, km, inertia = fit_kmeans_on_matrix(X_scaled, k, random_state=random_state, n_init=n_init)
            inertias.append(inertia)
        except Exception:
            inertias.append(np.nan)
    return ks, inertias


def compute_scores_over_k(X_scaled, k_min=2, k_max=10, random_state=42, n_init=10):
    """Hitung silhouette & davies untuk k in [k_min..k_max]."""
    ks = []
    silhouettes = []
    davies = []
    for k in range(k_min, k_max + 1):
        try:
            labels, _, _ = fit_kmeans_on_matrix(X_scaled, k, random_state=random_state, n_init=n_init)
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

    # -----------------------
    # Multiselect untuk fitur (default semua terpilih jika tersedia)
    # -----------------------
    st.subheader("⚙️ Pilih Fitur yang Digunakan untuk Clustering")
    available_cols = [col for col in ["Berat", "Tinggi", "BMI"] if col in df.columns]
    feature_cols = st.multiselect(
        "Pilih satu atau lebih fitur:",
        options=available_cols,
        default=available_cols
    )

    if not feature_cols:
        st.error("❌ Pilih minimal satu fitur (Berat / Tinggi / BMI).")
        return

    st.markdown(f"**Fitur yang digunakan:** {', '.join(feature_cols)}")

    # Bersihkan data (hanya baris yang punya semua fitur terpilih)
    df_work = df.dropna(subset=feature_cols).copy()
    if df_work.empty:
        st.error("Tidak ada data valid setelah pembersihan NA.")
        return

    # Scaling data
    X_scaled, scaler = prepare_scaled_matrix(df_work, feature_cols)

    # -----------------------
    # Evaluasi cluster (elbow, silhouette over k, davies)
    # -----------------------
    k_max = 10 if X_scaled.shape[0] >= 10 else max(3, X_scaled.shape[0] - 1)
    ks_elbow, inertias = compute_elbow(X_scaled, k_max=k_max, random_state=42, n_init=10)
    ks_scores, silhouettes, davies = compute_scores_over_k(X_scaled, k_min=2, k_max=k_max, random_state=42, n_init=10)

    st.subheader("🔎 Evaluasi Cluster untuk Berbagai Nilai k")
    # Elbow plot
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

    labels_3, km3, inertia3 = fit_kmeans_on_matrix(X_scaled, fixed_k, random_state=42, n_init=10)
    df_work["Cluster"] = labels_3

    # ----- Mapping cluster -> label secara otomatis (lebih andal daripada fixed mapping) -----
    # gunakan BMI sebagai referensi jika tersedia; jika tidak, gunakan fitur pertama
    if "BMI" in df_work.columns:
        ref_col = "BMI"
    else:
        ref_col = feature_cols[0]

    # urutkan cluster berdasarkan rata-rata ref_col (ascending)
    cluster_order = df_work.groupby("Cluster")[ref_col].mean().sort_values().index.tolist()
    # jika ada 3 cluster, mapping ke Stunting/Normal/Overweight
    if len(cluster_order) >= 3:
        cluster_to_label = {
            cluster_order[0]: "Stunting",
            cluster_order[1]: "Normal",
            cluster_order[2]: "Overweight"
        }
    else:
        # fallback: beri label generik
        cluster_to_label = {c: f"Cluster_{c}" for c in np.unique(labels_3)}

    df_work["Label"] = df_work["Cluster"].map(cluster_to_label)

    # ----- Hitung metrik untuk k=3 (silhouette & davies_bouldin) -----
    sil_3, db_3 = None, None
    try:
        if len(np.unique(labels_3)) > 1:
            sil_3 = silhouette_score(X_scaled, labels_3)
            db_3 = davies_bouldin_score(X_scaled, labels_3)
    except Exception:
        sil_3, db_3 = None, None

    # Ringkasan hasil (tampilkan metrik k=3 juga)
    st.subheader("📊 Ringkasan Hasil (k=3)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jumlah data", len(df_work))
    with col2:
        st.metric("Elbow (Inertia) (k=3)", f"{inertia3:.2f}")
    with col3:
        st.metric("Silhouette (k=3)", f"{sil_3:.3f}" if sil_3 is not None else "—")
    with col4:
        st.metric("Davies–Bouldin (k=3)", f"{db_3:.3f}" if db_3 is not None else "—")

    # Statistik per Label
    st.subheader("Statistik per Label")
    agg = df_work.groupby("Label")[feature_cols].mean().round(2)
    agg["Jumlah Anak"] = df_work.groupby("Label").size()
    st.dataframe(agg.reset_index(), use_container_width=True)

    # Visualisasi k=3
    st.subheader("🗺️ Visualisasi Clustering")
    x_axis = feature_cols[0]
    y_axis = feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]
    fig = px.scatter(df_work, x=x_axis, y=y_axis, color="Label", hover_data=df_work.columns,
                     title=f"Sebaran Data: {x_axis} vs {y_axis}")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # 📝 Data Hasil (preview)
    # -----------------------
    st.subheader("📝 Data Hasil")
    st.dataframe(df_work, use_container_width=True)

    # -----------------------
    # Unduh hasil (CSV + Excel)
    # -----------------------
    st.subheader("⬇️ Unduh Hasil Analisa K-Means")
    # CSS agar tombol rata tengah
    st.markdown("""
        <style>
        div.download-buttons {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-top: 10px;
            margin-bottom: 20px;
        }
        div.download-buttons > div {
            flex: 0 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Buffer CSV
    csv_buf = io.StringIO()
    df_work.to_csv(csv_buf, index=False)

    # Buffer Excel (tambahkan sheet metrics k over k, elbow, dan metrics_k3)
    xls_buf = io.BytesIO()
    with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as writer:
        df_work.to_excel(writer, index=False, sheet_name="Hasil")
        # mapping
        mapping_df = pd.DataFrame({
            "Cluster": list(cluster_to_label.keys()),
            "Label": list(cluster_to_label.values())
        })
        mapping_df.to_excel(writer, index=False, sheet_name="Mapping")
        # metrics over k
        pd.DataFrame({
            "k": ks_scores,
            "silhouette": silhouettes,
            "davies_bouldin": davies
        }).to_excel(writer, index=False, sheet_name="Metrics_over_k")
        # elbow
        pd.DataFrame({"k": ks_elbow, "inertia": inertias}).to_excel(writer, index=False, sheet_name="Elbow")
        # metrics for k=3
        pd.DataFrame({
            "metric": ["silhouette_k3", "davies_bouldin_k3", "inertia_k3"],
            "value": [sil_3 if sil_3 is not None else np.nan,
                      db_3 if db_3 is not None else np.nan,
                      inertia3]
        }).to_excel(writer, index=False, sheet_name="Metrics_k3")
    # make sure to get bytes
    xls_data = xls_buf.getvalue()

    st.markdown('<div class="download-buttons">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📄 Unduh CSV",
            data=csv_buf.getvalue(),
            file_name="hasil_kmeans.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        st.download_button(
            "📘 Unduh Excel",
            data=xls_data,
            file_name="hasil_kmeans.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    show()
