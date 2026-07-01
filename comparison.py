import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def show():
 st.title("Perbandingan Algoritma")
 alg=["K-Means","Agglomerative"]
 sil=[0.398,0.411]
 dbi=[0.905,0.869]
 x=np.arange(2);w=0.35
 fig,ax=plt.subplots()
 b1=ax.bar(x-w/2,sil,w,label="Silhouette")
 b2=ax.bar(x+w/2,dbi,w,label="DBI")
 ax.set_xticks(x);ax.set_xticklabels(alg);ax.legend()
 st.pyplot(fig)
