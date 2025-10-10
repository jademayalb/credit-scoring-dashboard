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
from config import API_URL_BASE, PREDICT_ENDPOINT, DEFAULT_THRESHOLD, FEATURE_DESCRIPTIONS

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
            
            # Standardisation des noms de clés pour l'interface interne
            result = {
                "client_id": int(client_id),
                "probability": data.get("probabilite_defaut"),
                "threshold": data.get("seuil_optimal", DEFAULT_THRESHOLD),
                "decision": data.get("decision", "INCONNU"),
                "model_name": data.get("model_name", ""),
                "raw_data": data  # Conservation des données brutes
            }
            
            logger.info(f"Prédiction récupérée avec succès pour le client {client_id}")
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

# Fonction pour récupérer les détails d'un client depuis le CSV
@st.cache_data(ttl=3600)
def get_client_details(client_id: int) -> Optional[Dict[str, Any]]:
    """
    Récupère les informations détaillées d'un client depuis le CSV d'application_test.
    
    Args:
        client_id: Identifiant unique du client
        
    Returns:
        Dictionnaire contenant les détails du client ou None en cas d'erreur
    """
    try:
        # Liste des chemins possibles pour trouver le CSV
        possible_paths = [
            "data/application_test.csv",                   # Si exécuté depuis credit-scoring-dashboard/
            "application_test.csv",                        # Si exécuté depuis credit-scoring-dashboard/data/
            "credit-scoring-dashboard/data/application_test.csv",  # Si exécuté depuis le répertoire parent
            "../data/application_test.csv"                 # Si exécuté depuis un sous-répertoire
        ]
        
        # Essayer chaque chemin jusqu'à trouver le fichier
        df = None
        for path in possible_paths:
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
def get_feature_importance(client_id: int) -> Optional[Dict[str, float]]:
    """
    Génère des valeurs d'importance des features basées sur les données du client.
    
    Args:
        client_id: Identifiant unique du client
        
    Returns:
        Dictionnaire avec les noms de features comme clés et les valeurs SHAP comme valeurs
    """
    # Pour l'instant, nous simulons les valeurs SHAP
    # Dans une future itération, vous pourriez étendre l'API pour calculer ces valeurs
    
    # On récupère d'abord la prédiction et les détails client
    prediction = get_client_prediction(client_id)
    details = get_client_details(client_id)
    
    if not prediction or not details:
        logger.warning(f"Impossible de générer les valeurs SHAP pour le client {client_id}: données manquantes")
        return None
    
    # Simulons les valeurs SHAP basées sur les vraies données du client
    features = details["features"]
    
    # Direction du SHAP basée sur la décision (positif = défaut, négatif = non-défaut)
    direction = 1 if prediction.get("decision") == "REFUSÉ" else -1
    
    # Valeurs SHAP générées semi-aléatoirement mais cohérentes avec les données et la décision
    shap_values = {
        # Les sources externes ont généralement un impact important et négatif (plus le score est élevé, moins de risque)
        "EXT_SOURCE_3": -0.35 * (features["EXT_SOURCE_3"] / 0.5) * direction,
        "EXT_SOURCE_2": -0.28 * (features["EXT_SOURCE_2"] / 0.5) * direction,
        "EXT_SOURCE_1": -0.15 * (features["EXT_SOURCE_1"] / 0.5) * direction if "EXT_SOURCE_1" in features else 0,
        
        # L'âge (plus âgé = moins risqué)
        "DAYS_BIRTH": -0.12 * (abs(features["DAYS_BIRTH"]) / 15000) * direction,
        
        # Temps d'emploi (plus long = moins risqué)
        "DAYS_EMPLOYED": -0.10 * (abs(features["DAYS_EMPLOYED"]) / 5000) * direction if features["DAYS_EMPLOYED"] != 365243 else 0,
        
        # Revenu (plus élevé = moins risqué)
        "AMT_INCOME_TOTAL": -0.08 * (features["AMT_INCOME_TOTAL"] / 200000) * direction,
        
        # Montant du crédit (plus élevé = plus risqué)
        "AMT_CREDIT": 0.14 * (features["AMT_CREDIT"] / 1000000) * direction,
        
        # Ratios calculés
        "PAYMENT_RATE": -0.09 * (features["PAYMENT_RATE"] / 0.1) * direction,
        "CREDIT_INCOME_RATIO": 0.18 * (features["CREDIT_INCOME_RATIO"] / 3) * direction
    }
    
    logger.info(f"Valeurs SHAP générées avec succès pour le client {client_id}")
    return shap_values

# Liste des clients disponibles depuis le CSV
@st.cache_data(ttl=86400)  # Mise en cache pour 24 heures
def get_available_clients(limit: int = 100) -> List[int]:
    """
    Récupère la liste des ID clients disponibles dans l'application_test.csv.
    
    Args:
        limit: Nombre maximum de clients à récupérer
        
    Returns:
        Liste des ID clients disponibles
    """
    try:
        # Liste des chemins possibles pour trouver le CSV
        possible_paths = [
            "data/application_test.csv",
            "application_test.csv",
            "credit-scoring-dashboard/data/application_test.csv",
            "../data/application_test.csv"
        ]
        
        # Essayer chaque chemin jusqu'à trouver le fichier
        df = None
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    logger.info(f"CSV chargé avec succès pour la liste des clients depuis: {path}")
                    break
            except Exception as e:
                continue
        
        if df is None:
            logger.error("Impossible de trouver le fichier CSV pour la liste des clients")
            # Retourne une liste par défaut en cas d'erreur
            return [100001, 100005, 100013, 100028, 100038, 100042, 100057, 100069, 100074, 100083]
        
        # Récupération des IDs client
        client_ids = df['SK_ID_CURR'].sort_values().tolist()
        logger.info(f"{len(client_ids)} IDs clients récupérés, limités à {min(limit, len(client_ids))}")
        
        return client_ids[:limit]
    except Exception as e:
        logger.exception("Erreur lors de la récupération des clients disponibles")
        # Retourne une liste par défaut en cas d'erreur
        return [100001, 100005, 100013, 100028, 100038, 100042, 100057, 100069, 100074, 100083]