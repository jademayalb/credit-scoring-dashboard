import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Credit Scoring Dashboard")

st.markdown("""
## Bienvenue dans votre outil d'analyse de crédit

Cet outil vous permet d'analyser les demandes de crédit et d'expliquer les décisions aux clients.

### 📋 Navigation :
- **🏠 Home** : Vue d'ensemble et recherche de clients
- **👤 Profil Client** : Analyse détaillée du profil
- **📊 Comparaison** : Comparaison avec des clients similaires  
- **🔧 Simulation** : Test de modifications sur le dossier

### 🚀 Pour commencer :
Utilisez la sidebar pour naviguer entre les pages ou commencez par la page **Home**.
""")

st.info("💡 Astuce : Toutes vos analyses sont sauvegardées automatiquement pendant votre session.")
