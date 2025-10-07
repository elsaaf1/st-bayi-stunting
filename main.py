import streamlit as st
from streamlit_option_menu import option_menu
import kmeans
import agglomerative

# -----------------------------
# Konfigurasi halaman Streamlit
# -----------------------------
st.set_page_config(
    page_title="Analisis Cluster BMI Anak",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# CSS Styling — Sidebar & Konten
# -----------------------------
st.markdown("""
    <style>
    /* Hilangkan padding berlebih di halaman utama */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f7f9fb;
        padding-top: 1rem;
        border-right: 1px solid #d3d3d3;
    }

    /* Header sidebar */
    [data-testid="stSidebar"] h2 {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
    }

    /* Option menu text styling */
    div[data-testid="stSidebar"] .nav-link {
        font-size: 1rem !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        margin-bottom: 5px !important;
    }

    /* Hover & active effects */
    div[data-testid="stSidebar"] .nav-link:hover {
        background-color: #e9ecef !important;
        color: #2c3e50 !important;
    }
    div[data-testid="stSidebar"] .nav-link.active {
        background-color: #2c3e50 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar Navigation
# -----------------------------
with st.sidebar:
    st.title("📂 Menu Navigasi")
    selected = option_menu(
        menu_title=None,
        options=["Metode K-Means", "Metode Agglomerative"],
        icons=["bar-chart-line", "diagram-3"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f7f9fb"},
            "icon": {"color": "#2c3e50", "font-size": "1.2rem"},
            "nav-link": {"font-size": "1rem", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#2c3e50", "color": "white"},
        }
    )
    st.markdown("---")
    st.caption("🧠 Analisis data stunting, normal, dan overweight anak menggunakan metode clustering.")


# -----------------------------
# Routing to selected method
# -----------------------------
if selected == "Metode K-Means":
    kmeans.show()
elif selected == "Metode Agglomerative":
    agglomerative.show()
