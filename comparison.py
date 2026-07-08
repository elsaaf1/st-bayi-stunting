import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def show():

    st.title("📊 Perbandingan Algoritma Clustering")

    st.markdown("""
    Halaman ini menyajikan hasil evaluasi antara algoritma
    **K-Means** dan **Agglomerative Clustering**
    berdasarkan nilai **Silhouette Score**
    dan **Davies–Bouldin Index (DBI)**.
    """)

    # ==========================
    # Nilai Evaluasi
    # ==========================

    algoritma = ["K-Means", "Agglomerative"]

    silhouette = [0.419, 0.412]

    dbi = [0.863, 0.869]

    x = np.arange(len(algoritma))
    width = 0.35

    # ==========================
    # Grafik
    # ==========================

    fig, ax = plt.subplots(figsize=(8,5))

    bars1 = ax.bar(
        x-width/2,
        silhouette,
        width,
        label="Silhouette Score"
    )

    bars2 = ax.bar(
        x+width/2,
        dbi,
        width,
        label="Davies-Bouldin Index"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(algoritma)

    ax.set_ylabel("Nilai Evaluasi")

    ax.set_title("Perbandingan Algoritma Clustering")

    ax.legend()

    # Menampilkan nilai di atas batang
    for bar in bars1:

        height = bar.get_height()

        ax.text(
            bar.get_x()+bar.get_width()/2,
            height+0.01,
            f"{height:.3f}",
            ha="center",
            fontsize=9
        )

    for bar in bars2:

        height = bar.get_height()

        ax.text(
            bar.get_x()+bar.get_width()/2,
            height+0.01,
            f"{height:.3f}",
            ha="center",
            fontsize=9
        )

    st.pyplot(fig)

    # ==========================
    # Penjelasan
    # ==========================

    st.subheader("Interpretasi Hasil")

    st.write("""
Silhouette Score digunakan untuk mengukur seberapa baik setiap data
berada pada cluster yang terbentuk. Semakin mendekati nilai 1 maka
kualitas cluster semakin baik.

Davies–Bouldin Index (DBI) digunakan untuk mengukur tingkat kemiripan
antar cluster. Semakin kecil nilai DBI maka kualitas cluster semakin baik.

Berdasarkan hasil evaluasi, algoritma K-Means memiliki Silhouette Score
lebih tinggi dan DBI lebih rendah dibandingkan Agglomerative Clustering.
Hal ini menunjukkan bahwa K-Means memberikan kualitas cluster yang lebih baik
untuk data antropometri balita pada penelitian ini.
""")

    # ==========================
    # Tabel
    # ==========================

    hasil = pd.DataFrame({

        "Algoritma": algoritma,

        "Silhouette Score": silhouette,

        "Davies-Bouldin Index": dbi

    })

    st.subheader("Tabel Perbandingan")

    st.dataframe(
        hasil,
        use_container_width=True
    )

    # ==========================
    # Kesimpulan
    # ==========================

    st.success("""
Kesimpulan:

Algoritma K-Means menghasilkan kualitas cluster yang lebih baik
dibandingkan Agglomerative Clustering berdasarkan
Silhouette Score dan Davies–Bouldin Index.
""")
