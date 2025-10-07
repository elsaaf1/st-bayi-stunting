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
# CSS adaptif untuk light & dark mode
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

    /* =====================
       LIGHT MODE STYLING
       ===================== */
    [data-testid="stSidebar"][data-theme="light"], body[data-theme="light"] [data-testid="stSidebar"] {
        background-color: #f7f9fb !important;
        border-right: 1px solid #d3d3d3;
        color: #2c3e50;
    }

    body[data-theme="light"] .nav-link, 
    body[data-theme="light"] [data-testid="stSidebar"] span,
    body[data-theme="light"] [data-testid="stSidebar"] h2 {
        color: #2c3e50 !important;
    }

    body[data-theme="light"] .nav-link.active {
        background-color: #2c3e50 !important;
        color: white !important;
    }

    body[data-theme="light"] .nav-link:hover {
        background-color: #e9ecef !important;
        color: #2c3e50 !important;
    }

    /* =====================
       DARK MODE STYLING
       ===================== */
    [data-testid="stSidebar"][data-theme="dark"], body[data-theme="dark"] [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #333;
        color: #f8f9fa !important;
    }

    body[data-theme="dark"] .nav-link, 
    body[data-theme="dark"] [data-testid="stSidebar"] span,
    body[data-theme="dark"] [data-testid="stSidebar"] h2 {
        color: #e5e7eb !important;
    }

    body[data-theme="dark"] .nav-link.active {
        background-color: #2563eb !important;
        color: white !important;
    }

    body[data-theme="dark"] .nav-link:hover {
        background-color: #1e293b !important;
        color: white !important;
    }

    /* Umum: gaya link */
    .nav-link {
        font-size: 1rem !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        margin-bottom: 6px !important;
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
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"font-size": "1.2rem"},
            "nav-link": {"text-align": "left", "margin": "0px"},
            "nav-link-selected": {"font-weight": "600"},
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
