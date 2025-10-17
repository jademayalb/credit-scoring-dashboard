"""
Module de communication avec l'API de scoring crédit.
Gère les appels API avec mise en cache et gestion des erreurs.
"""

import requests
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from typing import Dict, List, Optional, Any, Union
import logging

# Import de la configuration
from config import (
    API_URL_BASE, PREDICT_ENDPOINT, SHAP_ENDPOINT, DEFAULT_THRESHOLD, 
    FEATURE_DESCRIPTIONS, CSV_PATHS,
    CLIENTS_ENDPOINT, CLIENT_DETAILS_ENDPOINT 
)

# Définir le nouvel endpoint pour les valeurs SHAP mappées
SHAP_MAPPED_ENDPOINT = f"{API_URL_BASE}/shap_values_mapped/"

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fonction principale pour récupérer la prédiction
@st.cache_data(ttl=3600)  # Mise en cache pour 1 heure
def get_client_prediction(client_id: int) -> Optional[Dict[str, Any]]:
    """
    Récupère la prédiction pour un client spécifique depuis l'API Heroku.
    
    Args:
        client_id: Identifiant unique du client
        
    Returns:
        Dictionnaire contenant la prédiction ou None en cas d'erreur
    """
    try:
        logger.info(f"Récupération des prédictions pour le client {client_id}")
        response = requests.get(f"{PREDICT_ENDPOINT}{client_id}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Réponse brute de l'API: {data}")
            
            # Standardisation des noms de clés pour l'interface interne
            # Ajuster pour correspondre aux clés de la nouvelle API SHAP
            result = {
                "client_id": int(client_id),
                # Utiliser les nouvelles clés (probability) ou les anciennes (probabilite_defaut) avec fallback
                "probability": data.get("probability", data.get("probabilite_defaut", 0)),
                "threshold": data.get("threshold", data.get("seuil_optimal", DEFAULT_THRESHOLD)),
                "decision": data.get("decision", "INCONNU"),
                "model_name": data.get("model_name", ""),
                "raw_data": data  # Conservation des données brutes
            }
            
            logger.info(f"Prédiction structurée: {result}")
            return result
        elif response.status_code == 404:
            logger.warning(f"Client {client_id} non trouvé dans l'API")
            return None
        else:
            logger.error(f"Erreur API {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logger.exception(f"Exception lors de l'appel API pour client {client_id}")
        return None

# Nouvelle fonction pour récupérer les détails client depuis l'API
@st.cache_data(ttl=3600)
def get_client_details_from_api(client_id: int) -> Optional[Dict[str, Any]]:
    """
    Récupère les informations détaillées d'un client depuis l'API.
    
    Args:
        client_id: Identifiant unique du client
        
    Returns:
        Dictionnaire contenant les détails du client ou None en cas d'erreur
    """
    try:
        logger.info(f"Récupération des détails pour le client {client_id} depuis l'API")
        response = requests.get(f"{CLIENT_DETAILS_ENDPOINT}{client_id}/details")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Détails du client {client_id} récupérés avec succès depuis l'API")
            return data
        elif response.status_code == 404:
            logger.warning(f"Client {client_id} non trouvé dans l'API")
            return None
        else:
            logger.warning(f"Erreur API {response.status_code}: {response.text}. Utilisation du fallback CSV.")
            return None
            
    except Exception as e:
        logger.exception(f"Exception lors de l'appel API pour client {client_id}.")
        return None

# Fonction existante modifiée pour utiliser l'API en priorité
@st.cache_data(ttl=3600)
def get_client_details(client_id: int) -> Optional[Dict[str, Any]]:
    """
    Récupère les informations détaillées d'un client depuis l'API, avec fallback vers CSV.
    
    Args:
        client_id: Identifiant unique du client
        
    Returns:
        Dictionnaire contenant les détails du client ou None en cas d'erreur
    """
    # Essayer d'abord avec l'API
    api_data = get_client_details_from_api(client_id)
    if api_data:
        return api_data
        
    # Si l'API échoue, utiliser la méthode originale basée sur CSV
    logger.info(f"Utilisation du fallback CSV pour les détails du client {client_id}")
    
    try:
        # Essayer chaque chemin jusqu'à trouver le fichier
        df = None
        for path in CSV_PATHS:
            try:
                if os.path.exists(path):
                    logger.info(f"CSV trouvé à: {path}")
                    df = pd.read_csv(path)
                    logger.info(f"CSV chargé avec succès depuis: {path}")
                    break
            except Exception as e:
                logger.debug(f"Impossible de charger le CSV depuis {path}: {str(e)}")
                continue
        
        if df is None:
            logger.error("Impossible de trouver le fichier CSV")
            return None
            
        # Filtrage pour le client spécifique
        client_data = df[df['SK_ID_CURR'] == client_id]
        
        if client_data.empty:
            logger.warning(f"Client {client_id} non trouvé dans le CSV")
            return None
            
        # Extraction des données en un seul passage
        client = client_data.iloc[0]
        logger.info(f"Données du client {client_id} extraites avec succès")
        
        # Calcul de l'âge à partir de DAYS_BIRTH (valeur négative en jours)
        age = abs(int(client['DAYS_BIRTH'] / 365)) if 'DAYS_BIRTH' in client else 0
        
        # Calcul du temps d'emploi (en années)
        employment_years = abs(int(client['DAYS_EMPLOYED'] / 365)) if 'DAYS_EMPLOYED' in client and client['DAYS_EMPLOYED'] != 365243 else 0
        
        # Constitution du dictionnaire de détails client
        return {
            "client_id": int(client_id),
            "personal_info": {
                "age": age,
                "gender": client.get('CODE_GENDER', ""),
                "family_status": client.get('NAME_FAMILY_STATUS', ""),
                "education": client.get('NAME_EDUCATION_TYPE', ""),
                "income": float(client.get('AMT_INCOME_TOTAL', 0)),
                "employment_years": employment_years
            },
            "credit_info": {
                "amount": float(client.get('AMT_CREDIT', 0)),
                "annuity": float(client.get('AMT_ANNUITY', 0)),
                "goods_price": float(client.get('AMT_GOODS_PRICE', 0)),
                "credit_term": int(float(client.get('AMT_CREDIT', 0) / client.get('AMT_ANNUITY', 1))) if client.get('AMT_ANNUITY', 0) > 0 else 0
            },
            "features": {
                # Inclure les features importantes pour l'explication
                "EXT_SOURCE_3": float(client.get('EXT_SOURCE_3', 0)),
                "EXT_SOURCE_2": float(client.get('EXT_SOURCE_2', 0)),
                "EXT_SOURCE_1": float(client.get('EXT_SOURCE_1', 0)) if 'EXT_SOURCE_1' in client else 0,
                "DAYS_BIRTH": float(client.get('DAYS_BIRTH', 0)),
                "DAYS_EMPLOYED": float(client.get('DAYS_EMPLOYED', 0)),
                "AMT_INCOME_TOTAL": float(client.get('AMT_INCOME_TOTAL', 0)),
                "AMT_CREDIT": float(client.get('AMT_CREDIT', 0)),
                # Calcul des ratios si nécessaire
                "CREDIT_INCOME_RATIO": float(client.get('AMT_CREDIT', 0)) / float(client.get('AMT_INCOME_TOTAL', 1)) 
                    if client.get('AMT_INCOME_TOTAL', 0) > 0 else 0,
                "PAYMENT_RATE": float(client.get('AMT_ANNUITY', 0)) / float(client.get('AMT_CREDIT', 1)) 
                    if client.get('AMT_CREDIT', 0) > 0 else 0
            }
        }
    except Exception as e:
        logger.exception(f"Erreur lors de la récupération des détails du client {client_id}")
        return None

# Fonction pour récupérer les valeurs SHAP (importance des features)
@st.cache_data(ttl=3600)
def get_feature_importance(client_id):
    """
    Récupère les valeurs SHAP pour un client spécifique depuis l'API
    
    Parameters:
        client_id (int): Identifiant unique du client
        
    Returns:
        dict: Dictionnaire des features et leurs valeurs SHAP, ou None en cas d'erreur
    """
    try:
        logger.info(f"Récupération des valeurs SHAP pour le client {client_id}")
        response = requests.get(f"{SHAP_ENDPOINT}{client_id}")
        
        if response.status_code == 200:
            data = response.json()
            shap_values = data.get("shap_values", {})
            logger.info(f"Valeurs SHAP récupérées avec succès pour client {client_id}")
            return shap_values
            
        elif response.status_code == 404:
            logger.warning(f"Client {client_id} non trouvé pour les valeurs SHAP")
            return None
        else:
            logger.error(f"Erreur API SHAP {response.status_code}: {response.text}")
            # En cas d'erreur, utiliser l'ancienne méthode comme fallback
            return _get_feature_importance_fallback(client_id)
        
    except Exception as e:
        logger.exception(f"Exception lors de la récupération des valeurs SHAP pour client {client_id}")
        # En cas d'exception, utiliser l'ancienne méthode comme fallback
        return _get_feature_importance_fallback(client_id)

# Nouvelle fonction pour récupérer les valeurs SHAP mappées
@st.cache_data(ttl=3600)
def get_mapped_feature_importance(client_id):
    """
    Récupère les valeurs SHAP mappées avec les valeurs réelles pour un client spécifique depuis l'API
    
    Parameters:
        client_id (int): Identifiant unique du client
        
    Returns:
        list: Liste des features avec leurs valeurs SHAP et valeurs réelles, ou None en cas d'erreur
    """
    try:
        logger.info(f"Récupération des valeurs SHAP mappées pour le client {client_id}")
        response = requests.get(f"{SHAP_MAPPED_ENDPOINT}{client_id}")
        
        if response.status_code == 200:
            data = response.json()
            mapped_shap_values = data.get("mapped_shap_values", [])
            logger.info(f"Valeurs SHAP mappées récupérées avec succès pour client {client_id}")
            return mapped_shap_values
            
        elif response.status_code == 404:
            logger.warning(f"Client {client_id} non trouvé pour les valeurs SHAP mappées")
            return None
        else:
            logger.error(f"Erreur API SHAP mappées {response.status_code}: {response.text}")
            # En cas d'erreur, générer un mapping local
            return _generate_mapped_feature_importance(client_id)
        
    except Exception as e:
        logger.exception(f"Exception lors de la récupération des valeurs SHAP mappées pour client {client_id}")
        # En cas d'exception, générer un mapping local
        return _generate_mapped_feature_importance(client_id)

def _generate_mapped_feature_importance(client_id):
    """
    Génère localement un mapping entre les valeurs SHAP et les valeurs réelles du client
    Utilisé comme fallback si l'API ne dispose pas de l'endpoint /shap_values_mapped/
    """
    logger.info(f"Génération locale du mapping des valeurs SHAP pour le client {client_id}")
    
    # Récupérer les valeurs SHAP
    shap_values = get_feature_importance(client_id)
    if not shap_values:
        return None
    
    # Récupérer les détails du client pour obtenir les valeurs réelles
    client_details = get_client_details(client_id)
    if not client_details or 'features' not in client_details:
        return None
    
    # Récupérer les features du client
    client_features = client_details['features']
    
    # Créer le mapping entre valeurs SHAP et valeurs réelles
    mapped_values = []
    
    for feature_name, shap_value in shap_values.items():
        # Récupérer la valeur réelle de la feature pour ce client
        real_value = client_features.get(feature_name, "N/A")
        
        # Déterminer la direction de l'impact
        impact_direction = "positif" if shap_value > 0 else "négatif"
        
        # Format de présentation des valeurs réelles selon le type de feature
        display_value = real_value
        if feature_name == "DAYS_BIRTH" and real_value != "N/A":
            try:
                display_value = f"{abs(int(real_value / 365))} ans" 
            except:
                display_value = "N/A"
        elif feature_name == "DAYS_EMPLOYED" and real_value != "N/A":
            try:
                if real_value == 365243:
                    display_value = "Sans emploi"
                else:
                    display_value = f"{abs(int(real_value / 365))} ans"
            except:
                display_value = "N/A"
                
        # Créer l'objet de mapping
        mapped_feature = {
            "feature_name": feature_name,
            "display_name": FEATURE_DESCRIPTIONS.get(feature_name, feature_name),
            "shap_value": shap_value,
            "real_value": real_value,
            "display_value": display_value,
            "impact_direction": impact_direction,
            "impact_value": abs(shap_value)  # Valeur absolue pour trier
        }
        
        mapped_values.append(mapped_feature)
    
    # Trier par importance (valeur absolue de SHAP)
    mapped_values.sort(key=lambda x: x["impact_value"], reverse=True)
    
    return mapped_values

def _get_feature_importance_fallback(client_id):
    """
    Version de fallback qui utilise la méthode précédente basée sur les valeurs globales
    """
    logger.warning(f"Utilisation du fallback pour les valeurs SHAP du client {client_id}")
    
    # Définir l'importance globale des features (pré-calculée)
    feature_importance = {
        "EXT_SOURCE_3": {"importance": 0.364685, "direction": -1},  # Négatif = réduit le risque
        "EXT_SOURCE_2": {"importance": 0.324024, "direction": -1},
        "AMT_GOODS_PRICE": {"importance": 0.215875, "direction": 1},  # Positif = augmente le risque
        "AMT_CREDIT": {"importance": 0.198489, "direction": 1},
        "EXT_SOURCE_1": {"importance": 0.157631, "direction": -1},
        "DAYS_EMPLOYED": {"importance": 0.125956, "direction": -1},
        "CODE_GENDER": {"importance": 0.125809, "direction": 1},  # Pour les hommes
        "NAME_EDUCATION_TYPE": {"importance": 0.090088, "direction": -1},  # Pour Higher education
        "DAYS_BIRTH": {"importance": 0.077843, "direction": -1},
        "AMT_ANNUITY": {"importance": 0.075558, "direction": 1}
    }
    
    # Récupérer les détails du client
    client_details = get_client_details(client_id)
    
    if not client_details or 'features' not in client_details:
        return None
        
    # Récupérer les valeurs réelles des features pour ce client
    client_features = client_details['features']
    
    # Calculer l'impact de chaque feature pour ce client
    client_impacts = {}
    for feature, info in feature_importance.items():
        if feature in client_features:
            # Récupérer la valeur de la feature pour ce client
            value = client_features[feature]
            
            # Calculer l'impact en fonction de l'importance et de la direction
            impact = info["importance"] * info["direction"]
            
            # Ajuster l'impact en fonction de la valeur spécifique du client
            if feature in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
                # Pour les sources externes, plus la valeur est élevée, moins le risque est élevé
                if value > 0.5:  # Supposons que 0.5 est la moyenne
                    impact = -abs(impact)
                else:
                    impact = abs(impact)
            
            elif feature in ["DAYS_BIRTH", "DAYS_EMPLOYED"]:
                # Ces features sont négatives (jours dans le passé)
                years = abs(value) / 365.25
                if feature == "DAYS_BIRTH" and years > 40:  # Plus de 40 ans
                    impact = -abs(impact)
                elif feature == "DAYS_EMPLOYED" and years > 5:  # Plus de 5 ans d'emploi
                    impact = -abs(impact)
                else:
                    impact = abs(impact)
            
            elif feature == "CODE_GENDER":
                # Pour le genre, l'impact est positif pour les hommes
                if value == "M":
                    impact = abs(impact)
                else:
                    impact = -abs(impact)
            
            elif feature == "NAME_EDUCATION_TYPE":
                # Pour l'éducation, l'impact est négatif pour l'éducation supérieure
                if value == "Higher education":
                    impact = -abs(impact)
                else:
                    impact = abs(impact)
            
            # Ajouter l'impact calculé au dictionnaire
            client_impacts[feature] = impact
    
    return client_impacts

# Nouvelle fonction pour récupérer les clients disponibles depuis l'API
@st.cache_data(ttl=3600)
def get_available_clients_from_api(limit: int = 100, offset: int = 0) -> List[int]:
    """
    Récupère la liste des ID clients disponibles depuis l'API.
    
    Args:
        limit: Nombre maximum de clients à récupérer
        offset: Index à partir duquel commencer la récupération
        
    Returns:
        Liste des ID clients disponibles ou liste vide en cas d'erreur
    """
    try:
        logger.info(f"Récupération de la liste des clients depuis l'API (limit={limit}, offset={offset})")
        response = requests.get(f"{CLIENTS_ENDPOINT}?limit={limit}&offset={offset}")
        
        if response.status_code == 200:
            data = response.json()
            client_ids = data.get("client_ids", [])
            total = data.get("total", 0)
            logger.info(f"{len(client_ids)}/{total} IDs clients récupérés depuis l'API")
            return client_ids
        else:
            logger.warning(f"Erreur API {response.status_code}: {response.text}.")
            return []
            
    except Exception as e:
        logger.exception(f"Exception lors de l'appel API pour la liste des clients.")
        return []
        
# Fonction existante modifiée pour utiliser l'API en priorité
@st.cache_data(ttl=86400)  # Mise en cache pour 24 heures
def get_available_clients(limit: int = 100, offset: int = 0) -> List[int]:
    """
    Récupère la liste des ID clients disponibles, prioritairement depuis l'API avec fallback vers CSV.
    
    Args:
        limit: Nombre maximum de clients à récupérer
        offset: Index à partir duquel commencer la récupération
        
    Returns:
        Liste des ID clients disponibles ou liste vide en cas d'erreur
    """
    # Essayer d'abord avec l'API
    api_clients = get_available_clients_from_api(limit, offset)
    if api_clients:
        return api_clients
        
    # Si l'API échoue, utiliser la méthode originale basée sur CSV
    logger.info(f"Utilisation du fallback CSV pour la liste des clients")
    
    try:
        # Essayer chaque chemin jusqu'à trouver le fichier
        df = None
        for path in CSV_PATHS:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    logger.info(f"CSV chargé avec succès pour la liste des clients depuis: {path}")
                    break
            except Exception as e:
                continue
        
        if df is None:
            logger.error("Impossible de trouver le fichier CSV pour la liste des clients")
            # Retourner une liste vide au lieu d'une liste par défaut
            return []
        
        # Récupération des IDs client
        client_ids = df['SK_ID_CURR'].sort_values().tolist()
        logger.info(f"{len(client_ids)} IDs clients récupérés, limités à {min(limit, len(client_ids))}")
        
        return client_ids[:limit]
    except Exception as e:
        logger.exception("Erreur lors de la récupération des clients disponibles")
        # Retourner une liste vide au lieu d'une liste par défaut
        return []

# Nouvelle fonction pour tester l'état de l'API
def test_api_connection() -> Dict[str, Any]:
    """
    Teste la connexion à l'API et renvoie l'état de chaque endpoint.
    
    Returns:
        dict: État de la connexion pour chaque endpoint
    """
    results = {
        "status": "OK",
        "endpoints": {}
    }
    
    try:
        # Test de l'endpoint clients
        start_time = time.time()
        response = requests.get(f"{CLIENTS_ENDPOINT}?limit=1")
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            results["endpoints"]["clients"] = {
                "status": "OK",
                "response_time": response_time,
                "total_clients": response.json().get("total", 0)
            }
        else:
            results["status"] = "PARTIAL_ERROR"
            results["endpoints"]["clients"] = {
                "status": "ERROR",
                "code": response.status_code
            }
            
        # Utiliser un ID client standard pour les tests
        client_id = 100001
        
        # Test de l'endpoint predict
        start_time = time.time()
        response = requests.get(f"{PREDICT_ENDPOINT}{client_id}")
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            results["endpoints"]["predict"] = {
                "status": "OK",
                "response_time": response_time
            }
        else:
            results["status"] = "PARTIAL_ERROR"
            results["endpoints"]["predict"] = {
                "status": "ERROR",
                "code": response.status_code
            }
            
        # Test de l'endpoint shap_values
        start_time = time.time()
        response = requests.get(f"{SHAP_ENDPOINT}{client_id}")
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            results["endpoints"]["shap_values"] = {
                "status": "OK",
                "response_time": response_time,
                "features_count": len(response.json().get("shap_values", {}))
            }
        else:
            results["status"] = "PARTIAL_ERROR"
            results["endpoints"]["shap_values"] = {
                "status": "ERROR",
                "code": response.status_code
            }
            
        # Test de l'endpoint client_details
        start_time = time.time()
        response = requests.get(f"{CLIENT_DETAILS_ENDPOINT}{client_id}/details")
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            results["endpoints"]["client_details"] = {
                "status": "OK",
                "response_time": response_time
            }
        else:
            results["status"] = "PARTIAL_ERROR"
            results["endpoints"]["client_details"] = {
                "status": "ERROR",
                "code": response.status_code
            }
            
    except Exception as e:
        results["status"] = "ERROR"
        results["error"] = str(e)
        logger.exception("Erreur lors du test de connexion à l'API")
        
    return results

def display_api_status():
    """
    Affiche un widget de diagnostic pour tester la connexion à l'API.
    À placer dans la sidebar de votre dashboard.
    """
    with st.sidebar.expander("🌐 Diagnostic API", expanded=False):
        # Texte explicatif
        st.write("Vérifier l'état des services API:")
        
        # Espace pour séparer
        st.write("")
        
        # Utiliser un seul élément au lieu de colonnes pour maximiser la largeur
        # Le bouton occupera toute la largeur disponible
        if st.button("🔄 Tester", key="api_test", use_container_width=True):
            with st.spinner("Test en cours..."):
                # Tester la connexion à l'API
                try:
                    # Récupérer les URL de base depuis config.py
                    from config import API_URL_BASE
                    
                    # Tester la connexion
                    import requests
                    import time
                    start_time = time.time()
                    response = requests.get(f"{API_URL_BASE}/health", timeout=5)
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        st.success("✅ L'API est accessible")
                        st.write(f"Temps de réponse: {response_time:.2f} secondes")
                        
                        # Afficher des infos supplémentaires si disponibles
                        try:
                            data = response.json()
                            st.write(f"Version: {data.get('version', 'Non spécifiée')}")
                            st.write(f"Modèle: {data.get('model', 'Non spécifié')}")
                        except:
                            pass
                    else:
                        st.warning(f"⚠️ L'API a répondu avec le code {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Impossible de se connecter à l'API: {str(e)}")
                    st.info("Vérifiez que l'API est bien démarrée et accessible.")
