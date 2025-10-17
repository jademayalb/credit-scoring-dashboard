import streamlit as st
import requests
import logging
from config import API_URL_BASE, CLIENT_DETAILS_ENDPOINT, PREDICT_ENDPOINT, SHAP_ENDPOINT

# Configuration basique de la page
st.set_page_config(page_title="Débogage Client", page_icon="🔍")

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Débogage de la page Profil Client")

# Vérification des variables de session
st.subheader("1. État de la session")
st.write("Contenu de session_state:", st.session_state)

if "client_id" in st.session_state:
    client_id = st.session_state.client_id
    st.success(f"ID client trouvé dans la session: {client_id}")
else:
    client_id = st.number_input("Entrez un ID client pour tester:", min_value=100000, max_value=999999, value=100001)
    if st.button("Enregistrer dans la session"):
        st.session_state.client_id = client_id
        st.experimental_rerun()

# Test des appels API
st.subheader("2. Test des appels API")

# Test de connectivité API
st.write("Test de connectivité à l'API de base:")
try:
    response = requests.get(f"{API_URL_BASE}/health", timeout=5)
    st.write(f"- Statut: {response.status_code}")
    st.write(f"- Réponse: {response.text[:200]}")
    if response.status_code == 200:
        st.success("✅ API de base accessible")
    else:
        st.error("❌ Problème d'accès à l'API de base")
except Exception as e:
    st.error(f"❌ Erreur de connexion à l'API: {str(e)}")

# Test des endpoints spécifiques
endpoints = {
    "Prédiction": f"{PREDICT_ENDPOINT}{client_id}",
    "Détails client": f"{CLIENT_DETAILS_ENDPOINT}{client_id}/details",
    "Valeurs SHAP": f"{SHAP_ENDPOINT}{client_id}"
}

for name, url in endpoints.items():
    st.write(f"Test de l'endpoint {name}:")
    st.code(url)  # Affiche l'URL complète pour vérification
    
    try:
        response = requests.get(url, timeout=10)
        st.write(f"- Statut: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ Endpoint {name} accessible")
            with st.expander("Voir la réponse"):
                st.json(data)
        else:
            st.error(f"❌ Problème d'accès à l'endpoint {name}")
            st.write(f"- Erreur: {response.text[:200]}")
    except Exception as e:
        st.error(f"❌ Erreur lors de l'appel à {name}: {str(e)}")

# Affichage des configurations
st.subheader("3. Vérification des configurations")
st.write("URL de base de l'API:", API_URL_BASE)
st.write("Endpoint de prédiction:", PREDICT_ENDPOINT)
st.write("Endpoint des détails client:", CLIENT_DETAILS_ENDPOINT)
st.write("Endpoint des valeurs SHAP:", SHAP_ENDPOINT)

# Instructions
st.info("""
Ce débogage permet d'identifier où se situe le problème:
1. Si les URLs affichées ne sont pas correctes, corrigez le fichier config.py
2. Si l'API n'est pas accessible, vérifiez que le serveur API est bien démarré
3. Si les appels API échouent, vérifiez les logs du serveur API pour plus de détails
""")
