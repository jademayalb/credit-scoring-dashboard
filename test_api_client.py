import streamlit as st
import pandas as pd
import os
from utils.api_client import (
    get_client_prediction,
    get_client_details,
    get_feature_importance,
    get_available_clients
)

# Configuration de la page
st.set_page_config(
    page_title="Test API Client Module",
    layout="wide"
)

st.title("Test du module API Client")
st.write("Cette page permet de tester les fonctions du module `utils/api_client.py`.")

# Vérification préliminaire de l'accès au fichier CSV
st.header("Vérification du fichier CSV")

possible_paths = [
    "data/application_test.csv",
    "application_test.csv",
    "credit-scoring-dashboard/data/application_test.csv",
    "../data/application_test.csv"
]

csv_found = False
csv_path = None

col1, col2 = st.columns(2)

with col1:
    st.write("Chemins testés:")
    for path in possible_paths:
        if os.path.exists(path):
            st.write(f"- **{path}**: Trouvé ✓")
            csv_found = True
            csv_path = path
        else:
            st.write(f"- {path}: Non trouvé")

with col2:
    if csv_found:
        st.success(f"Fichier CSV trouvé à l'emplacement: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            st.write(f"Nombre de clients dans le CSV: {len(df)}")
            st.write(f"Quelques IDs clients: {df['SK_ID_CURR'].head(5).tolist()}")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du CSV: {str(e)}")
    else:
        st.error("Fichier CSV non trouvé! Vérifiez les chemins d'accès.")

# Test 1: Récupération des clients disponibles
st.header("1. Récupération des clients disponibles")

with st.expander("Exécuter le test", expanded=True):
    if st.button("Obtenir la liste des clients", key="btn_clients"):
        with st.spinner("Chargement des clients..."):
            try:
                clients = get_available_clients(limit=20)  # Limiter à 20 pour l'affichage
                st.success(f"{len(clients)} clients récupérés avec succès!")
                st.write("IDs des clients (20 premiers) :")
                st.write(clients[:20])
            except Exception as e:
                st.error(f"Erreur lors de la récupération des clients : {str(e)}")

# Test 2: Récupération des prédictions
st.header("2. Récupération des prédictions")

with st.expander("Exécuter le test", expanded=True):
    # Sélection d'un client pour le test
    client_ids = get_available_clients(10)
    if not client_ids:
        st.error("Aucun client disponible pour le test")
    else:
        client_id = st.selectbox("Sélectionner un client pour le test :", options=client_ids)
        
        if st.button("Obtenir la prédiction", key="btn_prediction"):
            with st.spinner(f"Récupération de la prédiction pour le client {client_id}..."):
                try:
                    prediction = get_client_prediction(client_id)
                    if prediction:
                        st.success(f"Prédiction récupérée avec succès pour le client {client_id} !")
                        
                        # Affichage formaté des informations importantes
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Probabilité de défaut", f"{prediction.get('probability', 0):.2%}")
                        with col2:
                            st.metric("Seuil", f"{prediction.get('threshold', 0):.2%}")
                            
                        # Afficher la décision avec style
                        decision = prediction.get('decision', 'INCONNU')
                        if decision == "ACCEPTÉ":
                            st.markdown(f"<h3 style='color: green;'>Décision : {decision}</h3>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h3 style='color: red;'>Décision : {decision}</h3>", unsafe_allow_html=True)
                        
                        # Affichage des données brutes
                        with st.expander("Données brutes"):
                            st.json(prediction)
                    else:
                        st.error(f"Aucune prédiction retournée pour le client {client_id}")
                except Exception as e:
                    st.error(f"Erreur lors de la récupération de la prédiction : {str(e)}")

# Test 3: Récupération des détails client
st.header("3. Récupération des détails client")

with st.expander("Exécuter le test", expanded=True):
    if not client_ids:
        st.error("Aucun client disponible pour le test")
    else:
        client_id = st.selectbox("Sélectionner un client pour le test :", options=client_ids, key="select_details")
        
        if st.button("Obtenir les détails", key="btn_details"):
            with st.spinner(f"Récupération des détails pour le client {client_id}..."):
                try:
                    details = get_client_details(client_id)
                    if details:
                        st.success(f"Détails client récupérés avec succès pour le client {client_id} !")
                        
                        # Affichage organisé des informations
                        st.subheader("Informations personnelles")
                        personal_df = pd.DataFrame(details["personal_info"].items(), columns=["Attribut", "Valeur"])
                        st.dataframe(personal_df, use_container_width=True)
                        
                        st.subheader("Informations crédit")
                        credit_df = pd.DataFrame(details["credit_info"].items(), columns=["Attribut", "Valeur"])
                        st.dataframe(credit_df, use_container_width=True)
                        
                        # Affichage des features importantes
                        st.subheader("Features")
                        features_df = pd.DataFrame(details["features"].items(), columns=["Feature", "Valeur"])
                        st.dataframe(features_df, use_container_width=True)
                    else:
                        st.error(f"Aucun détail retourné pour le client {client_id}")
                except Exception as e:
                    st.error(f"Erreur lors de la récupération des détails : {str(e)}")

# Test 4: Récupération des valeurs SHAP
st.header("4. Récupération des valeurs SHAP")

with st.expander("Exécuter le test", expanded=True):
    if not client_ids:
        st.error("Aucun client disponible pour le test")
    else:
        client_id = st.selectbox("Sélectionner un client pour le test :", options=client_ids, key="select_shap")
        
        if st.button("Obtenir les valeurs SHAP", key="btn_shap"):
            with st.spinner(f"Récupération des valeurs SHAP pour le client {client_id}..."):
                try:
                    shap_values = get_feature_importance(client_id)
                    if shap_values:
                        st.success(f"Valeurs SHAP récupérées avec succès pour le client {client_id} !")
                        
                        # Conversion en DataFrame pour affichage
                        shap_df = pd.DataFrame(shap_values.items(), columns=["Feature", "Importance"])
                        shap_df = shap_df.sort_values("Importance", ascending=False)
                        
                        # Affichage sous forme de tableau
                        st.dataframe(shap_df, use_container_width=True)
                        
                        # Visualisation avec un graphique à barres horizontales
                        st.bar_chart(shap_df.set_index("Feature"))
                    else:
                        st.error(f"Aucune valeur SHAP retournée pour le client {client_id}")
                except Exception as e:
                    st.error(f"Erreur lors de la récupération des valeurs SHAP : {str(e)}")

# Conclusion
st.header("Résumé des tests")
st.write("""
Cette page vous permet de vérifier que toutes les fonctions principales du module `api_client.py` fonctionnent correctement.
Si tous les tests sont passés avec succès, vous pouvez passer à l'étape suivante du développement.
""")