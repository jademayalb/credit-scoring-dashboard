import os
from flask import Flask, jsonify, request
from joblib import load
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import logging
import shap
import requests
from io import StringIO
from flask_swagger_ui import get_swaggerui_blueprint
import logging.handlers
from datetime import datetime
import json
from werkzeug.exceptions import BadRequest

# Configuration du logger avec rotation des fichiers
BASE_DIR = os.path.dirname(__file__)
log_file = os.path.join(BASE_DIR, "logs", "api.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Chemins
MODEL_PATH = os.path.join(BASE_DIR, "model_complet.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "imputer.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")

# URL vers le CSV remote
GITHUB_CSV_URL = "https://raw.githubusercontent.com/jademayalb/credit-scoring-api/main/data/application_test.csv"

# Cache pour les données
_test_df_cache = None
_last_fetch_time = 0
_cache_ttl = 3600  # 1 heure

FEATURE_DESCRIPTIONS = {
    "EXT_SOURCE_1": "Score normalisé - Source externe 1",
    "EXT_SOURCE_2": "Score normalisé - Source externe 2",
    "EXT_SOURCE_3": "Score normalisé - Source externe 3",
    "DAYS_BIRTH": "Âge (en jours, négatif)",
    "DAYS_EMPLOYED": "Nombre de jours d'emploi",
    "AMT_INCOME_TOTAL": "Revenu total du client",
    "AMT_CREDIT": "Montant du crédit",
    "AMT_ANNUITY": "Montant de l'annuité",
    "AMT_GOODS_PRICE": "Prix des biens financés",
    "CODE_GENDER": "Genre du client",
    "NAME_EDUCATION_TYPE": "Type d'éducation",
    "NAME_FAMILY_STATUS": "Statut familial",
    "CNT_CHILDREN": "Nombre d'enfants",
    "CNT_FAM_MEMBERS": "Nombre de membres dans la famille",
    "NAME_INCOME_TYPE": "Type de revenu",
    "DAYS_ID_PUBLISH": "Nombre de jours depuis la publication de la carte d'identité",
    "REGION_RATING_CLIENT": "Évaluation de la région du client",
    "REGION_RATING_CLIENT_W_CITY": "Évaluation de la région du client avec ville",
    "FLAG_OWN_CAR": "Possession d'une voiture",
    "FLAG_OWN_REALTY": "Possession d'un bien immobilier",
    "NAME_CONTRACT_TYPE": "Type de contrat",
    "ORGANIZATION_TYPE": "Type d'organisation",
    "OCCUPATION_TYPE": "Type d'occupation",
    "CREDIT_INCOME_PERCENT": "Pourcentage de crédit par rapport au revenu",
    "ANNUITY_INCOME_PERCENT": "Pourcentage de l'annuité par rapport au revenu",
    "CREDIT_TERM": "Terme du crédit (années)",
    "DAYS_EMPLOYED_PERCENT": "Pourcentage des jours d'emploi par rapport à l'âge"
}

FEATURE_MAPPING = {
    "EXT_SOURCE_1": {"section": "features", "key": "EXT_SOURCE_1"},
    "EXT_SOURCE_2": {"section": "features", "key": "EXT_SOURCE_2"},
    "EXT_SOURCE_3": {"section": "features", "key": "EXT_SOURCE_3"},
    "DAYS_BIRTH": {"section": "personal_info", "key": "age", "transform": lambda x: abs(x) / 365.25},
    "DAYS_EMPLOYED": {"section": "personal_info", "key": "employment_years", "transform": lambda x: abs(x) / 365.25 if x != 365243 else 0},
    "AMT_INCOME_TOTAL": {"section": "personal_info", "key": "income"},
    "AMT_CREDIT": {"section": "credit_info", "key": "amount"},
    "AMT_ANNUITY": {"section": "credit_info", "key": "annuity"},
    "AMT_GOODS_PRICE": {"section": "credit_info", "key": "goods_price"},
}

def fetch_github_data():
    global _test_df_cache, _last_fetch_time
    current_time = int(pd.Timestamp.now().timestamp())
    if _test_df_cache is not None and (current_time - _last_fetch_time) < _cache_ttl:
        logger.info("Utilisation du cache test_df")
        return _test_df_cache
    try:
        logger.info(f"Téléchargement CSV depuis {GITHUB_CSV_URL}")
        response = requests.get(GITHUB_CSV_URL, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        _test_df_cache = df
        _last_fetch_time = current_time
        logger.info(f"Données téléchargées: {len(df)} lignes")
        return df
    except Exception as e:
        logger.error(f"Erreur téléchargement CSV: {e}")
        if _test_df_cache is not None:
            logger.warning("Utilisation données en cache (obsolètes)")
            return _test_df_cache
        raise

# Charger modèle et artefacts
try:
    model_data = load(MODEL_PATH)
    model = model_data['model']
    scaler = model_data['scaler']
    imputer = model_data['imputer']
    features = model_data['features']
    threshold = float(model_data.get('optimal_threshold', 0.5))
    model_name = model_data.get('model_name', 'model')
    poly_transformer = model_data.get('poly_transformer', None)

    test_df = fetch_github_data()
    logger.info("Modèle et artefacts chargés")
except Exception as e:
    logger.error(f"Erreur chargement model/artifacts: {e}")
    raise

def preprocess(df: pd.DataFrame, features_model: List[str], poly_transformer=None) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            if len(df[col].unique()) <= 2:
                df[col] = le.fit_transform(df[col].astype(str))
    df = pd.get_dummies(df)
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED_ANOM'] = df['DAYS_EMPLOYED'] == 365243
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})
    for col in ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']:
        if col not in df.columns:
            df[col] = np.nan
    # safe ratios
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace({0: np.nan})
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace({0: np.nan})
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'].replace({0: np.nan})
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'].replace({0: np.nan})
    if poly_transformer is not None:
        poly_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']
        for col in poly_cols:
            if col not in df.columns:
                df[col] = np.nan
        try:
            poly_values = poly_transformer.transform(df[poly_cols])
            poly_feature_names = poly_transformer.get_feature_names_out(poly_cols)
            poly_df = pd.DataFrame(poly_values, columns=poly_feature_names, index=df.index)
            df = pd.concat([df, poly_df], axis=1)
        except Exception as e:
            logger.warning(f"Poly transform failed: {e}")
    for col in features_model:
        if col not in df.columns:
            df[col] = 0
    df = df[features_model]
    return df

# SHAP explainer initialization (best-effort)
try:
    if hasattr(model, 'predict_proba'):
        sample_size = min(100, len(test_df))
        sample_data = test_df.sample(sample_size, random_state=42)
        sample_processed = preprocess(sample_data, features, poly_transformer)
        sample_imputed = imputer.transform(sample_processed)
        sample_scaled = scaler.transform(sample_imputed)
        try:
            explainer = shap.TreeExplainer(model, check_additivity=False)
            logger.info("SHAP TreeExplainer initialisé")
        except Exception as e:
            logger.warning(f"TreeExplainer failed: {e}")
            explainer = None
    else:
        logger.warning("Model has no predict_proba -> SHAP unavailable")
        explainer = None
except Exception as e:
    logger.error(f"Erreur initialisation SHAP: {e}")
    explainer = None

app = Flask(__name__)

# Swagger UI
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "Credit Scoring API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/static/swagger.json')
def get_swagger():
    try:
        with open(os.path.join(BASE_DIR, 'static', 'swagger.json'), 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        logger.warning(f"Swagger file missing: {e}")
        return jsonify({})

# GET predict by client id
@app.route('/predict/<int:client_id>', methods=['GET'])
def predict_client(client_id):
    try:
        if client_id <= 0:
            return jsonify({"erreur": "ID client invalide", "status": "INVALID_REQUEST"}), 400
        test_df_local = fetch_github_data()
        client_row = test_df_local[test_df_local['SK_ID_CURR'] == client_id]
        if client_row.empty:
            logger.warning(f"Client {client_id} not found")
            return jsonify({"erreur": f"Client ID {client_id} introuvable", "status": "NOT_FOUND"}), 404
        client_processed = preprocess(client_row, features, poly_transformer)
        X_imputed = imputer.transform(client_processed)
        X_scaled = scaler.transform(X_imputed)
        proba = float(model.predict_proba(X_scaled)[0, 1])
        decision = "REFUSÉ" if proba >= threshold else "ACCEPTÉ"
        logger.info(f"Prediction GET client {client_id}: proba={proba:.4f}")
        return jsonify({
            "client_id": int(client_id),
            "probability": proba,
            "threshold": float(threshold),
            "decision": decision,
            "model_name": model_name,
            "status": "OK"
        })
    except BadRequest as e:
        logger.error(f"BadRequest in predict_client: {e}")
        return jsonify({"erreur": "Requête invalide", "details": str(e), "status": "INVALID_REQUEST"}), 400
    except Exception as e:
        logger.exception(f"Erreur interne predict_client: {e}")
        return jsonify({"erreur": "Erreur interne du serveur", "details": str(e), "status": "ERROR"}), 500

# POST predict for ad-hoc features or override
@app.route('/predict', methods=['POST'])
def predict_from_payload():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"erreur": "Payload JSON manquant", "status": "INVALID_REQUEST"}), 400
        payload_features = payload.get("features") or {}
        client_id = payload.get("client_id", None)
        if client_id is not None:
            test_df_local = fetch_github_data()
            client_row = test_df_local[test_df_local['SK_ID_CURR'] == int(client_id)]
            if client_row.empty:
                logger.warning(f"Client ID {client_id} introuvable for override")
                return jsonify({"erreur": f"Client ID {client_id} introuvable", "status": "NOT_FOUND"}), 404
            client_row_copy = client_row.copy()
            idx = client_row_copy.index[0]
            for k, v in payload_features.items():
                client_row_copy.at[idx, k] = v
            df_to_predict = client_row_copy
        else:
            if not isinstance(payload_features, dict) or len(payload_features) == 0:
                return jsonify({"erreur": "Aucune feature fournie pour la prédiction", "status": "INVALID_REQUEST"}), 400
            df_to_predict = pd.DataFrame([payload_features])
        client_processed = preprocess(df_to_predict, features, poly_transformer)
        X_imputed = imputer.transform(client_processed)
        X_scaled = scaler.transform(X_imputed)
        proba = float(model.predict_proba(X_scaled)[0, 1])
        decision = "REFUSÉ" if proba >= threshold else "ACCEPTÉ"
        logger.info(f"Prediction POST (client_id={client_id}) proba={proba:.4f}")
        result = {
            "client_id": int(client_id) if client_id is not None else None,
            "probability": proba,
            "threshold": float(threshold),
            "decision": decision,
            "model_name": model_name,
            "status": "OK"
        }
        return jsonify(result)
    except BadRequest as e:
        logger.error(f"Payload JSON invalide pour /predict : {e}")
        return jsonify({"erreur": "Payload JSON invalide", "details": str(e), "status": "INVALID_REQUEST"}), 400
    except Exception as e:
        logger.exception(f"Erreur POST /predict: {e}")
        return jsonify({"erreur": "Erreur interne du serveur", "details": str(e), "status": "ERROR"}), 500

# SHAP values endpoint
@app.route('/shap_values/<int:client_id>', methods=['GET'])
def get_shap_values(client_id):
    try:
        if client_id <= 0:
            return jsonify({"erreur": "ID client invalide", "status": "INVALID_REQUEST"}), 400
        try:
            limit = int(request.args.get('limit', 20))
            if limit <= 0:
                return jsonify({"erreur": "Le paramètre 'limit' doit être positif", "status": "INVALID_REQUEST"}), 400
        except ValueError:
            return jsonify({"erreur": "Le paramètre 'limit' doit être un entier", "status": "INVALID_REQUEST"}), 400
        if explainer is None:
            logger.warning("Explainer SHAP non disponible")
            return jsonify({"erreur": "L'explainer SHAP n'est pas disponible", "status": "ERROR"}), 503
        test_df_local = fetch_github_data()
        client_row = test_df_local[test_df_local['SK_ID_CURR'] == client_id]
        if client_row.empty:
            return jsonify({"erreur": f"Client ID {client_id} introuvable", "status": "NOT_FOUND"}), 404
        client_processed = preprocess(client_row, features, poly_transformer)
        X_imputed = imputer.transform(client_processed)
        X_scaled = scaler.transform(X_imputed)
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_dict = {}
        for i, feature_name in enumerate(features):
            shap_dict[feature_name] = float(shap_values[0][i])
        shap_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:limit]
        top_shap_dict = {k: v for k, v in shap_items}
        logger.info(f"SHAP computed for client {client_id}")
        return jsonify({"client_id": int(client_id), "shap_values": top_shap_dict, "status": "OK"})
    except BadRequest as e:
        logger.error(f"BadRequest shap_values: {e}")
        return jsonify({"erreur": "Requête invalide", "details": str(e), "status": "INVALID_REQUEST"}), 400
    except Exception as e:
        logger.exception(f"Erreur SHAP values: {e}")
        return jsonify({"erreur": "Erreur lors du calcul des valeurs SHAP", "details": str(e), "status": "ERROR"}), 500

# SHAP mapped endpoint
@app.route('/shap_values_mapped/<int:client_id>', methods=['GET'])
def get_shap_values_mapped(client_id):
    try:
        if client_id <= 0:
            return jsonify({"erreur": "ID client invalide", "status": "INVALID_REQUEST"}), 400
        try:
            limit = int(request.args.get('limit', 20))
            if limit <= 0:
                return jsonify({"erreur": "Le paramètre 'limit' doit être positif", "status": "INVALID_REQUEST"}), 400
        except ValueError:
            return jsonify({"erreur": "Le paramètre 'limit' doit être un entier", "status": "INVALID_REQUEST"}), 400
        if explainer is None:
            return jsonify({"erreur": "L'explainer SHAP n'est pas disponible", "status": "ERROR"}), 503
        test_df_local = fetch_github_data()
        client_row = test_df_local[test_df_local['SK_ID_CURR'] == client_id]
        if client_row.empty:
            return jsonify({"erreur": f"Client ID {client_id} introuvable", "status": "NOT_FOUND"}), 404
        client_processed = preprocess(client_row, features, poly_transformer)
        X_imputed = imputer.transform(client_processed)
        X_scaled = scaler.transform(X_imputed)
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_dict = {}
        for i, feature_name in enumerate(features):
            shap_dict[feature_name] = float(shap_values[0][i])
        shap_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:limit]
        # Build mapped results using local client details (no internal HTTP)
        client_details = None
        try:
            client_details = get_client_details(client_id)
        except Exception:
            client_details = None
        mapped_results = []
        for feature_name, shap_value in shap_items:
            display_name = FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
            result = {
                "feature": feature_name,
                "display_name": display_name,
                "shap_value": float(shap_value),
                "impact_direction": "Favorable" if shap_value < 0 else "Défavorable",
                "impact_value": abs(round(shap_value, 4)),
                "real_value": "N/A"
            }
            if client_details and feature_name in FEATURE_MAPPING:
                mapping = FEATURE_MAPPING[feature_name]
                if mapping.get("computed"):
                    try:
                        result["real_value"] = mapping["formula"](client_details)
                    except Exception:
                        result["real_value"] = "N/A"
                else:
                    section = mapping.get("section")
                    key = mapping.get("key")
                    if section in client_details and key in client_details[section]:
                        value = client_details[section][key]
                        if "transform" in mapping and callable(mapping["transform"]):
                            try:
                                value = mapping["transform"](value)
                            except Exception:
                                pass
                        result["real_value"] = value
            if isinstance(result["real_value"], (int, float)):
                result["real_value"] = round(result["real_value"], 2)
            mapped_results.append(result)
        logger.info(f"SHAP mapped computed for client {client_id}")
        return jsonify({"client_id": int(client_id), "mapped_shap_values": mapped_results, "status": "OK"})
    except BadRequest as e:
        logger.error(f"BadRequest shap_values_mapped: {e}")
        return jsonify({"erreur": "Requête invalide", "details": str(e), "status": "INVALID_REQUEST"}), 400
    except Exception as e:
        logger.exception(f"Erreur shap_values_mapped: {e}")
        return jsonify({"erreur": "Erreur lors du calcul des valeurs SHAP mappées", "details": str(e), "status": "ERROR"}), 500

# Clients listing
@app.route('/clients', methods=['GET'])
def get_available_clients():
    try:
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        if limit <= 0 or offset < 0:
            return jsonify({"erreur": "Les paramètres 'limit' et 'offset' doivent être positifs", "status": "INVALID_REQUEST"}), 400
    except ValueError:
        return jsonify({"erreur": "Les paramètres 'limit' et 'offset' doivent être des entiers", "status": "INVALID_REQUEST"}), 400
    try:
        test_df_local = fetch_github_data()
        client_ids = test_df_local['SK_ID_CURR'].tolist()
        paginated = client_ids[offset:offset+limit]
        return jsonify({"client_ids": paginated, "total": len(client_ids), "limit": limit, "offset": offset, "status": "OK"})
    except Exception as e:
        logger.exception(f"Erreur clients endpoint: {e}")
        return jsonify({"erreur": "Erreur lors de la récupération des IDs clients", "details": str(e), "status": "ERROR"}), 500

# Client details endpoint
@app.route('/client/<int:client_id>/details', methods=['GET'])
def client_details_endpoint(client_id):
    try:
        if client_id <= 0:
            return jsonify({"erreur": "ID client invalide", "status": "INVALID_REQUEST"}), 400
        test_df_local = fetch_github_data()
        client_row = test_df_local[test_df_local['SK_ID_CURR'] == client_id]
        if client_row.empty:
            return jsonify({"erreur": f"Client ID {client_id} introuvable", "status": "NOT_FOUND"}), 404
        client_data = client_row.iloc[0].to_dict()
        personal_info = {
            "gender": client_data.get('CODE_GENDER', ''),
            "age": int(abs(client_data.get('DAYS_BIRTH', 0)) / 365.25) if 'DAYS_BIRTH' in client_data else None,
            "education": client_data.get('NAME_EDUCATION_TYPE', ''),
            "family_status": client_data.get('NAME_FAMILY_STATUS', ''),
            "children_count": int(client_data.get('CNT_CHILDREN', 0)),
            "family_size": float(client_data.get('CNT_FAM_MEMBERS', 1)),
            "income": float(client_data.get('AMT_INCOME_TOTAL', 0)),
            "employment_type": client_data.get('NAME_INCOME_TYPE', ''),
            "employment_years": int(abs(client_data.get('DAYS_EMPLOYED', 0)) / 365.25) if ('DAYS_EMPLOYED' in client_data and client_data.get('DAYS_EMPLOYED') != 365243) else 0,
            "occupation": client_data.get('OCCUPATION_TYPE', '')
        }
        credit_info = {
            "amount": float(client_data.get('AMT_CREDIT', 0)),
            "credit_term": int(float(client_data.get('AMT_CREDIT', 0)) / float(client_data.get('AMT_ANNUITY', 1))) if ('AMT_ANNUITY' in client_data and float(client_data.get('AMT_ANNUITY', 0)) > 0) else 0,
            "annuity": float(client_data.get('AMT_ANNUITY', 0)),
            "goods_price": float(client_data.get('AMT_GOODS_PRICE', 0)),
            "name_goods_category": client_data.get('NAME_CONTRACT_TYPE', '')
        }
        credit_history = {
            "previous_loans_count": 1,
            "previous_defaults": 0,
            "late_payments": 0,
            "credit_score": int(700 * (1 - float(client_data.get('EXT_SOURCE_3', 0.5)))) if 'EXT_SOURCE_3' in client_data else "N/A",
            "years_with_bank": 3
        }
        features_raw = {}
        for key in client_data.keys():
            if key in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                       'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL',
                       'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH',
                       'AMT_GOODS_PRICE', 'CODE_GENDER', 'FLAG_OWN_CAR',
                       'FLAG_OWN_REALTY', 'REGION_RATING_CLIENT',
                       'REGION_RATING_CLIENT_W_CITY', 'DAYS_REGISTRATION',
                       'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG']:
                try:
                    features_raw[key] = float(client_data.get(key, 0))
                except (ValueError, TypeError):
                    features_raw[key] = client_data.get(key)
        response = {
            "client_id": int(client_id),
            "personal_info": personal_info,
            "credit_info": credit_info,
            "credit_history": credit_history,
            "features": features_raw,
            "status": "OK"
        }
        return jsonify(response)
    except BadRequest as e:
        logger.error(f"BadRequest client details: {e}")
        return jsonify({"erreur": "Requête invalide", "details": str(e), "status": "INVALID_REQUEST"}), 400
    except Exception as e:
        logger.exception(f"Erreur client details: {e}")
        return jsonify({"erreur": "Erreur lors de la récupération des détails du client", "details": str(e), "status": "ERROR"}), 500

# Health endpoint
@app.route('/health', methods=['GET'])
def health_check():
    try:
        if model is None or features is None:
            return jsonify({"status": "ERROR", "message": "Modèle ou features non chargés"}), 500
        return jsonify({"status": "OK", "version": "1.0.0", "model": model_name, "timestamp": datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Erreur health_check: {e}")
        return jsonify({"status": "ERROR", "message": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404: {request.path}")
    return jsonify({"erreur": "Ressource introuvable", "path": request.path, "status": "NOT_FOUND"}), 404

@app.errorhandler(400)
def bad_request(error):
    logger.warning(f"400: {error}")
    return jsonify({"erreur": "Requête invalide", "details": str(error), "status": "INVALID_REQUEST"}), 400

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500: {error}")
    return jsonify({"erreur": "Erreur interne du serveur", "status": "ERROR"}), 500

if __name__ == "__main__":
    # Run development server
    app.run(debug=True, port=5800)
