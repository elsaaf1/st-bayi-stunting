import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def show():
 st.title("Perbandingan Algoritma Cluster")
 alg=["K-Means","Agglomerative"]
 sil=[0.419,0.412]
 dbi=[0.863,0.869]
 x=np.arange(2);w=0.35
 fig, ax = plt.subplots(figsize=(6,4))
 b1=ax.bar(x-w/2,sil,w,label="Silhouette")
 b2=ax.bar(x+w/2,dbi,w,label="DBI")
 ax.set_xticks(x);ax.set_xticklabels(alg);ax.legend()
 st.pyplot(fig)
