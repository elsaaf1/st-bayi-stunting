import pandas as pd

from sklearn.preprocessing import StandardScaler

df = pd.read_excel("Rekap Entry April by Name.xlsx")

X = df[
    [
        "Berat Badan",
        "Tinggi Badan",
        "BMI"
    ]
]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
kmeans.fit_predict(X_scaled)
