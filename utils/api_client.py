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
    FEATURE_DESCRIPTIONS, CSV_PATHS
)

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
        
# Liste des clients disponibles depuis le CSV
@st.cache_data(ttl=86400)  # Mise en cache pour 24 heures
def get_available_clients(limit: int = 100) -> List[int]:
    """
    Récupère la liste des ID clients disponibles dans l'application_test.csv.
    
    Args:
        limit: Nombre maximum de clients à récupérer
        
    Returns:
        Liste des ID clients disponibles ou liste vide en cas d'erreur
    """
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