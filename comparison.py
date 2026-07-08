from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering

# KMeans
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

dbi_kmeans = davies_bouldin_score(
    X_scaled,
    kmeans_labels
)

# Agglomerative
agg = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

agg_labels = agg.fit_predict(X_scaled)

dbi_agg = davies_bouldin_score(
    X_scaled,
    agg_labels
)

print("DBI KMeans :", dbi_kmeans)
print("DBI Agglomerative :", dbi_agg)

plt.figure(figsize=(6,4))

plt.bar(
    ["K-Means","Agglomerative"],
    [dbi_kmeans, dbi_agg]
)

plt.ylabel("Davies-Bouldin Index")
plt.title("Perbandingan Davies-Bouldin Index")

plt.show()
