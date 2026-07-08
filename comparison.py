from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd

# Membaca data
df = pd.read_excel("Rekap Entry April by Name.xlsx")

# Variabel yang digunakan
X = df[["Berat Badan", "Tinggi Badan", "BMI"]]

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Agglomerative
agg = AgglomerativeClustering(n_clusters=3)
labels_agg = agg.fit_predict(X_scaled)

# Hitung metrik
sil = [
    silhouette_score(X_scaled, labels_kmeans),
    silhouette_score(X_scaled, labels_agg)
]

dbi = [
    davies_bouldin_score(X_scaled, labels_kmeans),
    davies_bouldin_score(X_scaled, labels_agg)
]
