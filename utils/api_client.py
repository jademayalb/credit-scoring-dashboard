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
from datetime import datetime

# Import de la configuration
from config import (
    API_URL_BASE, PREDICT_ENDPOINT, SHAP_ENDPOINT, DEFAULT_THRESHOLD,
    FEATURE_DESCRIPTIONS, CSV_PATHS,
    CLIENTS_ENDPOINT, CLIENT_DETAILS_ENDPOINT
)

# Définir le nouvel endpoint pour les valeurs SHAP mappées (optionnel)
SHAP_MAPPED_ENDPOINT = f"{API_URL_BASE.rstrip('/')}/shap_values_mapped/"

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# get_client_prediction
# ---------------------------
@st.cache_data(ttl=3600)  # Mise en cache pour 1 heure
def get_client_prediction(client_id: Optional[int] = None, features: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Récupère la prédiction d'un client depuis l'API.
    - Si 'features' est fourni, envoie un POST /predict avec le JSON {'client_id': ..., 'features': ...} (ou {'features': ...}).
    - Sinon si client_id est fourni, effectue un GET /predict/<client_id>.
    Retourne None en cas d'erreur.
    """
    try:
        # Prefer POST when features provided (simulate / override)
        if features:
            logger.info(f"[API CLIENT] Envoi POST /predict (client_id={client_id}) payload features keys: {list(features.keys())}")
            post_url = PREDICT_ENDPOINT
            # Ensure trailing slash removed or kept as configured; assume PREDICT_ENDPOINT is base predict URL (e.g. https://.../predict/)
            payload = {"features": features}
            if client_id is not None:
                payload["client_id"] = int(client_id)
            response = requests.post(post_url, json=payload, timeout=15)
        elif client_id is not None:
            logger.info(f"[API CLIENT] GET {PREDICT_ENDPOINT}{client_id}")
            response = requests.get(f"{PREDICT_ENDPOINT}{client_id}", timeout=10)
        else:
            logger.error("[API CLIENT] get_client_prediction appelé sans client_id ni features")
            return None

        if response.status_code == 200:
            data = response.json()
            logger.info(f"[API CLIENT] Réponse API predict: {data}")
            result = {
                "client_id": int(client_id) if client_id is not None else data.get("client_id"),
                "probability": data.get("probability", data.get("probabilite_defaut", 0)),
                "threshold": data.get("threshold", data.get("seuil_optimal", DEFAULT_THRESHOLD)),
                "decision": data.get("decision", "INCONNU"),
                "model_name": data.get("model_name", ""),
                "raw_data": data
            }
            logger.info(f"[API CLIENT] Prédiction structurée: {result}")
            return result
        elif response.status_code == 404:
            logger.warning(f"[API CLIENT] Client {client_id} non trouvé (404)")
            return None
        else:
            logger.error(f"[API CLIENT] Erreur API predict {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logger.exception(f"[API CLIENT] Exception HTTP lors de l'appel predict: {e}")
        return None
    except Exception:
        logger.exception("[API CLIENT] Erreur inattendue dans get_client_prediction")
        return None

# ---------------------------
# get_client_details_from_api (HTTP)
# ---------------------------
@st.cache_data(ttl=3600)
def get_client_details_from_api(client_id: int) -> Optional[Dict[str, Any]]:
    """
    Récupère les informations détaillées d'un client depuis l'API.
    Retourne le JSON décodé si succès, None sinon.
    """
    try:
        logger.info(f"[API CLIENT] Récupération des détails depuis l'API pour client {client_id}")
        url = f"{CLIENT_DETAILS_ENDPOINT}{client_id}/details"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"[API CLIENT] Détails récupérés (client {client_id})")
            return data
        elif response.status_code == 404:
            logger.warning(f"[API CLIENT] Client {client_id} non trouvé (details 404)")
            return None
        else:
            logger.warning(f"[API CLIENT] Erreur API details {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logger.exception(f"[API CLIENT] Exception lors de get_client_details_from_api: {e}")
        return None

# ---------------------------
# get_client_details (API then CSV fallback)
# ---------------------------
@st.cache_data(ttl=3600)
def get_client_details(client_id: int) -> Optional[Dict[str, Any]]:
    """
    Récupère les informations détaillées d'un client depuis l'API, avec fallback vers CSV local.
    Retourne une structure standardisée attendue par le dashboard.
    """
    # Try API first
    api_data = get_client_details_from_api(client_id)
    if api_data:
        return api_data

    logger.info(f"[API CLIENT] Fallback CSV pour client {client_id}")
    try:
        df = None
        for path in CSV_PATHS:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    logger.info(f"[API CLIENT] CSV chargé depuis {path}")
                    break
            except Exception as e:
                logger.debug(f"[API CLIENT] Impossible de charger {path}: {e}")
                continue

        if df is None:
            logger.error("[API CLIENT] Aucun CSV trouvé dans CSV_PATHS")
            return None

        client_data = df[df['SK_ID_CURR'] == client_id]
        if client_data.empty:
            logger.warning(f"[API CLIENT] Client {client_id} introuvable dans CSV")
            return None

        client = client_data.iloc[0]
        age = abs(int(client['DAYS_BIRTH'] / 365)) if 'DAYS_BIRTH' in client else 0
        employment_years = abs(int(client['DAYS_EMPLOYED'] / 365)) if 'DAYS_EMPLOYED' in client and client['DAYS_EMPLOYED'] != 365243 else 0

        details = {
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
                "credit_term": int(float(client.get('AMT_CREDIT', 0)) / float(client.get('AMT_ANNUITY', 1))) if client.get('AMT_ANNUITY', 0) > 0 else 0
            },
            "features": {
                "EXT_SOURCE_3": float(client.get('EXT_SOURCE_3', 0)),
                "EXT_SOURCE_2": float(client.get('EXT_SOURCE_2', 0)),
                "EXT_SOURCE_1": float(client.get('EXT_SOURCE_1', 0)) if 'EXT_SOURCE_1' in client else 0,
                "DAYS_BIRTH": float(client.get('DAYS_BIRTH', 0)),
                "DAYS_EMPLOYED": float(client.get('DAYS_EMPLOYED', 0)),
                "AMT_INCOME_TOTAL": float(client.get('AMT_INCOME_TOTAL', 0)),
                "AMT_CREDIT": float(client.get('AMT_CREDIT', 0)),
                "CREDIT_INCOME_RATIO": float(client.get('AMT_CREDIT', 0)) / float(client.get('AMT_INCOME_TOTAL', 1)) if client.get('AMT_INCOME_TOTAL', 0) > 0 else 0,
                "PAYMENT_RATE": float(client.get('AMT_ANNUITY', 0)) / float(client.get('AMT_CREDIT', 1)) if client.get('AMT_CREDIT', 0) > 0 else 0
            }
        }
        logger.info(f"[API CLIENT] Détails construits depuis CSV pour client {client_id}")
        return details
    except Exception as e:
        logger.exception(f"[API CLIENT] Erreur lors de la lecture CSV pour client {client_id}: {e}")
        return None

# ---------------------------
# get_feature_importance (SHAP)
# ---------------------------
@st.cache_data(ttl=3600)
def get_feature_importance(client_id):
    """
    Récupère les valeurs SHAP pour un client spécifique depuis l'API (endpoint SHAP_ENDPOINT).
    Fallback vers un mapping pré-calculé si l'API échoue.
    """
    try:
        logger.info(f"[API CLIENT] Récupération SHAP pour client {client_id}")
        response = requests.get(f"{SHAP_ENDPOINT}{client_id}", timeout=15)
        if response.status_code == 200:
            data = response.json()
            shap_values = data.get("shap_values", {})
            logger.info(f"[API CLIENT] SHAP récupérés pour client {client_id}")
            return shap_values
        elif response.status_code == 404:
            logger.warning(f"[API CLIENT] SHAP: client {client_id} non trouvé (404)")
            return None
        else:
            logger.error(f"[API CLIENT] Erreur SHAP {response.status_code}: {response.text}")
            return _get_feature_importance_fallback(client_id)
    except Exception as e:
        logger.exception(f"[API CLIENT] Exception lors de la récupération SHAP: {e}")
        return _get_feature_importance_fallback(client_id)

# ---------------------------
# get_mapped_feature_importance
# ---------------------------
@st.cache_data(ttl=3600)
def get_mapped_feature_importance(client_id):
    """
    Récupère les valeurs SHAP mappées depuis l'API (mapped) ou génère localement un mapping si indisponible.
    """
    try:
        logger.info(f"[API CLIENT] Récupération SHAP mappées pour client {client_id}")
        response = requests.get(f"{SHAP_MAPPED_ENDPOINT}{client_id}", timeout=15)
        if response.status_code == 200:
            data = response.json()
            mapped = data.get("mapped_shap_values", [])
            logger.info(f"[API CLIENT] SHAP mappées récupérées pour client {client_id}")
            return mapped
        elif response.status_code == 404:
            logger.warning(f"[API CLIENT] SHAP mappées: client {client_id} non trouvé (404)")
            return None
        else:
            logger.error(f"[API CLIENT] Erreur SHAP mappées {response.status_code}: {response.text}")
            return _generate_mapped_feature_importance(client_id)
    except Exception as e:
        logger.exception(f"[API CLIENT] Exception lors de SHAP mappées: {e}")
        return _generate_mapped_feature_importance(client_id)

def _generate_mapped_feature_importance(client_id):
    """
    Génère localement un mapping entre valeurs SHAP (fallback) et valeurs réelles du client.
    """
    logger.info(f"[API CLIENT] Génération locale du mapping SHAP pour client {client_id}")
    shap_values = get_feature_importance(client_id)
    if not shap_values:
        return None
    client_details = get_client_details(client_id)
    if not client_details or 'features' not in client_details:
        return None
    client_features = client_details['features']
    mapped_values = []
    for feature_name, shap_value in shap_values.items():
        real_value = client_features.get(feature_name, "N/A")
        impact_direction = "Favorable" if shap_value < 0 else "Défavorable"
        display_value = real_value
        if feature_name == "DAYS_BIRTH" and real_value != "N/A":
            try:
                display_value = f"{abs(int(real_value / 365))} ans"
            except:
                display_value = "N/A"
        elif feature_name == "DAYS_EMPLOYED" and real_value != "N/A":
            try:
                display_value = "Sans emploi" if real_value == 365243 else f"{abs(int(real_value / 365))} ans"
            except:
                display_value = "N/A"
        mapped_feature = {
            "feature_name": feature_name,
            "display_name": FEATURE_DESCRIPTIONS.get(feature_name, feature_name),
            "shap_value": shap_value,
            "real_value": real_value,
            "display_value": display_value,
            "impact_direction": impact_direction,
            "impact_value": abs(shap_value)
        }
        mapped_values.append(mapped_feature)
    mapped_values.sort(key=lambda x: x["impact_value"], reverse=True)
    return mapped_values

# ---------------------------
# Fallback feature importance (global heuristic)
# ---------------------------
def _get_feature_importance_fallback(client_id):
    logger.warning(f"[API CLIENT] Utilisation fallback SHAP pour client {client_id}")
    feature_importance = {
        "EXT_SOURCE_3": {"importance": 0.364685, "direction": -1},
        "EXT_SOURCE_2": {"importance": 0.324024, "direction": -1},
        "AMT_GOODS_PRICE": {"importance": 0.215875, "direction": 1},
        "AMT_CREDIT": {"importance": 0.198489, "direction": 1},
        "EXT_SOURCE_1": {"importance": 0.157631, "direction": -1},
        "DAYS_EMPLOYED": {"importance": 0.125956, "direction": -1},
        "CODE_GENDER": {"importance": 0.125809, "direction": 1},
        "NAME_EDUCATION_TYPE": {"importance": 0.090088, "direction": -1},
        "DAYS_BIRTH": {"importance": 0.077843, "direction": -1},
        "AMT_ANNUITY": {"importance": 0.075558, "direction": 1}
    }
    client_details = get_client_details(client_id)
    if not client_details or 'features' not in client_details:
        return None
    client_features = client_details['features']
    client_impacts = {}
    for feature, info in feature_importance.items():
        if feature in client_features:
            value = client_features[feature]
            impact = info["importance"] * info["direction"]
            if feature in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
                impact = -abs(impact) if value > 0.5 else abs(impact)
            elif feature in ["DAYS_BIRTH", "DAYS_EMPLOYED"]:
                years = abs(value) / 365.25 if value is not None else 0
                if feature == "DAYS_BIRTH" and years > 40:
                    impact = -abs(impact)
                elif feature == "DAYS_EMPLOYED" and years > 5:
                    impact = -abs(impact)
                else:
                    impact = abs(impact)
            elif feature == "CODE_GENDER":
                impact = abs(impact) if value == "M" else -abs(impact)
            elif feature == "NAME_EDUCATION_TYPE":
                impact = -abs(impact) if value == "Higher education" else abs(impact)
            client_impacts[feature] = impact
    return client_impacts

# ---------------------------
# Clients list (API then CSV fallback)
# ---------------------------
@st.cache_data(ttl=3600)
def get_available_clients_from_api(limit: int = 100, offset: int = 0) -> List[int]:
    try:
        logger.info(f"[API CLIENT] get_available_clients_from_api(limit={limit}, offset={offset})")
        response = requests.get(f"{CLIENTS_ENDPOINT}?limit={limit}&offset={offset}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            client_ids = data.get("client_ids", [])
            total = data.get("total", 0)
            logger.info(f"[API CLIENT] Récupérés {len(client_ids)}/{total} clients depuis API")
            return client_ids
        else:
            logger.warning(f"[API CLIENT] Erreur clients API {response.status_code}: {response.text}")
            return []
    except Exception as e:
        logger.exception("[API CLIENT] Exception lors de get_available_clients_from_api")
        return []

@st.cache_data(ttl=86400)
def get_available_clients(limit: int = 100, offset: int = 0) -> List[int]:
    api_clients = get_available_clients_from_api(limit, offset)
    if api_clients:
        return api_clients
    logger.info("[API CLIENT] Fallback CSV pour la liste des clients")
    try:
        df = None
        for path in CSV_PATHS:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    break
            except Exception:
                continue
        if df is None:
            logger.error("[API CLIENT] Aucun CSV pour get_available_clients")
            return []
        client_ids = df['SK_ID_CURR'].sort_values().tolist()
        return client_ids[:limit]
    except Exception:
        logger.exception("[API CLIENT] Erreur lors de la récupération des clients depuis CSV")
        return []

# ---------------------------
# Diagnostics API
# ---------------------------
def test_api_connection() -> Dict[str, Any]:
    results = {"status": "OK", "endpoints": {}}
    try:
        start = time.time()
        resp = requests.get(f"{CLIENTS_ENDPOINT}?limit=1", timeout=5)
        elapsed = time.time() - start
        if resp.status_code == 200:
            results["endpoints"]["clients"] = {"status": "OK", "response_time": elapsed, "total_clients": resp.json().get("total", 0)}
        else:
            results["status"] = "PARTIAL_ERROR"
            results["endpoints"]["clients"] = {"status": "ERROR", "code": resp.status_code}
        client_id = 100001
        start = time.time()
        resp = requests.get(f"{PREDICT_ENDPOINT}{client_id}", timeout=5)
        elapsed = time.time() - start
        results["endpoints"]["predict"] = {"status": "OK" if resp.status_code == 200 else "ERROR", "code": resp.status_code, "response_time": elapsed}
        start = time.time()
        resp = requests.get(f"{SHAP_ENDPOINT}{client_id}", timeout=5)
        elapsed = time.time() - start
        results["endpoints"]["shap_values"] = {"status": "OK" if resp.status_code == 200 else "ERROR", "code": resp.status_code}
        start = time.time()
        resp = requests.get(f"{CLIENT_DETAILS_ENDPOINT}{client_id}/details", timeout=5)
        elapsed = time.time() - start
        results["endpoints"]["client_details"] = {"status": "OK" if resp.status_code == 200 else "ERROR", "code": resp.status_code}
    except Exception as e:
        results["status"] = "ERROR"
        results["error"] = str(e)
        logger.exception("[API CLIENT] test_api_connection failed")
    return results

def display_api_status():
    with st.sidebar.expander("🌐 Diagnostic API", expanded=False):
        st.write("Vérifier l'état des services API:")
        if st.button("🔄 Tester", key="api_test", use_container_width=True):
            with st.spinner("Test en cours..."):
                try:
                    start = time.time()
                    response = requests.get(f"{API_URL_BASE.rstrip('/')}/health", timeout=5)
                    elapsed = time.time() - start
                    if response.status_code == 200:
                        st.success("✅ L'API est accessible")
                        st.write(f"Temps de réponse: {elapsed:.2f}s")
                        try:
                            data = response.json()
                            st.write(f"Version: {data.get('version', 'N/A')}")
                            st.write(f"Modèle: {data.get('model', 'N/A')}")
                        except Exception:
                            pass
                    else:
                        st.warning(f"⚠️ L'API a répondu: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Impossible de se connecter à l'API: {e}")
                    st.info("Vérifiez que l'API est démarrée et que les URL dans config.py sont correctes.")
