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
log_file = os.path.join(os.path.dirname(__file__), "logs", "api.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Configuration avancée du logging
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

# Handler pour la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# Handler pour les fichiers avec rotation (5 fichiers de 5 Mo max)
file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Ajout des handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Chemin absolu pour les artefacts
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_complet.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "imputer.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")

# URL vers le fichier CSV sur GitHub (version raw)
GITHUB_CSV_URL = "https://raw.githubusercontent.com/jademayalb/credit-scoring-api/main/data/application_test.csv"

# Cache pour les données
_test_df_cache = None
_last_fetch_time = 0
_cache_ttl = 3600  # 1 heure en secondes

# Dictionnaire des descriptions pour les features
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

# Dictionnaire de mapping entre noms de features SHAP et les propriétés client
FEATURE_MAPPING = {
    # Features sources externes
    "EXT_SOURCE_1": {"section": "features", "key": "EXT_SOURCE_1"},
    "EXT_SOURCE_2": {"section": "features", "key": "EXT_SOURCE_2"},
    "EXT_SOURCE_3": {"section": "features", "key": "EXT_SOURCE_3"},
    
    # Features démographiques
    "DAYS_BIRTH": {"section": "personal_info", "key": "age", "transform": lambda x: abs(x) / 365.25},
    "DAYS_EMPLOYED": {"section": "personal_info", "key": "employment_years", "transform": lambda x: abs(x) / 365.25 if x != 365243 else 0},
    "CODE_GENDER": {"section": "personal_info", "key": "gender"},
    "NAME_EDUCATION_TYPE": {"section": "personal_info", "key": "education"},
    "NAME_FAMILY_STATUS": {"section": "personal_info", "key": "family_status"},
    "CNT_CHILDREN": {"section": "personal_info", "key": "children_count"},
    "CNT_FAM_MEMBERS": {"section": "personal_info", "key": "family_size"},
    "NAME_INCOME_TYPE": {"section": "personal_info", "key": "employment_type"},
    
    # Features financières
    "AMT_INCOME_TOTAL": {"section": "personal_info", "key": "income"},
    "AMT_CREDIT": {"section": "credit_info", "key": "amount"},
    "AMT_ANNUITY": {"section": "credit_info", "key": "annuity"},
    "AMT_GOODS_PRICE": {"section": "credit_info", "key": "goods_price"},
    "NAME_CONTRACT_TYPE": {"section": "credit_info", "key": "name_goods_category"},
    
    # Indicateurs binaires
    "FLAG_OWN_CAR": {"section": "features", "key": "FLAG_OWN_CAR"},
    "FLAG_OWN_REALTY": {"section": "features", "key": "FLAG_OWN_REALTY"},
    
    # Features avec encodage one-hot (exemples)
    "CODE_GENDER_F": {"section": "personal_info", "key": "gender", "transform": lambda x: x == "F"},
    "CODE_GENDER_M": {"section": "personal_info", "key": "gender", "transform": lambda x: x == "M"},
    "NAME_INCOME_TYPE_Working": {"section": "personal_info", "key": "employment_type", "transform": lambda x: x == "Working"},
    
    # Features calculées
    "CREDIT_INCOME_PERCENT": {"computed": True, "formula": lambda client: client["credit_info"]["amount"] / client["personal_info"]["income"] if client["personal_info"]["income"] > 0 else 0},
    "ANNUITY_INCOME_PERCENT": {"computed": True, "formula": lambda client: client["credit_info"]["annuity"] / client["personal_info"]["income"] if client["personal_info"]["income"] > 0 else 0},
    "CREDIT_TERM": {"section": "credit_info", "key": "credit_term"},
    "DAYS_EMPLOYED_PERCENT": {"computed": True, "formula": lambda client: client["personal_info"]["employment_years"] / client["personal_info"]["age"] if client["personal_info"]["age"] > 0 else 0},
    
    # Autres features qui pourraient être importantes
    "REGION_RATING_CLIENT": {"section": "features", "key": "REGION_RATING_CLIENT"},
    "REGION_RATING_CLIENT_W_CITY": {"section": "features", "key": "REGION_RATING_CLIENT_W_CITY"},
    "DAYS_ID_PUBLISH": {"section": "features", "key": "DAYS_ID_PUBLISH"},
    "OCCUPATION_TYPE": {"section": "features", "key": "OCCUPATION_TYPE"},
    "ORGANIZATION_TYPE": {"section": "features", "key": "ORGANIZATION_TYPE"}
}

def fetch_github_data():
    """
    Récupère les données depuis GitHub avec mise en cache
    """
    global _test_df_cache, _last_fetch_time
    
    current_time = int(pd.Timestamp.now().timestamp())
    
    # Utiliser le cache si disponible et pas trop vieux
    if _test_df_cache is not None and (current_time - _last_fetch_time) < _cache_ttl:
        logger.info("Utilisation des données en cache")
        return _test_df_cache
    
    try:
        logger.info(f"Téléchargement des données depuis GitHub: {GITHUB_CSV_URL}")
        response = requests.get(GITHUB_CSV_URL, timeout=10)  # Ajout d'un timeout
        response.raise_for_status()  # Vérifier les erreurs HTTP
        
        # Charger les données dans un DataFrame
        data = StringIO(response.text)
        df = pd.read_csv(data)
        
        # Mettre à jour le cache
        _test_df_cache = df
        _last_fetch_time = current_time
        
        logger.info(f"Données téléchargées avec succès: {len(df)} lignes")
        return df
    
    except requests.exceptions.Timeout:
        logger.error("Timeout lors du téléchargement des données depuis GitHub")
        if _test_df_cache is not None:
            logger.warning("Utilisation des données en cache (obsolètes)")
            return _test_df_cache
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors du téléchargement des données depuis GitHub: {e}")
        if _test_df_cache is not None:
            logger.warning("Utilisation des données en cache (obsolètes)")
            return _test_df_cache
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue lors du téléchargement des données: {e}")
        if _test_df_cache is not None:
            logger.warning("Utilisation des données en cache (obsolètes)")
            return _test_df_cache
        raise

# Charger le modèle et les artefacts
try:
    model_data = load(MODEL_PATH)
    model = model_data['model']
    scaler = model_data['scaler']
    imputer = model_data['imputer']
    features = model_data['features']
    threshold = model_data['optimal_threshold']
    model_name = model_data['model_name']
    poly_transformer = model_data.get('poly_transformer', None)
    
    # Récupérer les données depuis GitHub
    test_df = fetch_github_data()
    
    logger.info("Modèle et artefacts chargés avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle ou des artefacts : {e}")
    raise

def preprocess(df, features_model, poly_transformer=None):
    """
    Prétraite les données pour la prédiction selon le pipeline défini lors de l'entraînement
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données à prétraiter
        features_model (list): Liste des features attendues par le modèle
        poly_transformer (PolynomialFeatures, optional): Transformer pour les features polynomiales
        
    Returns:
        pandas.DataFrame: DataFrame prétraité
    """
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
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    if poly_transformer is not None:
        poly_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']
        for col in poly_cols:
            if col not in df.columns:
                df[col] = np.nan
        poly_values = poly_transformer.transform(df[poly_cols])
        poly_feature_names = poly_transformer.get_feature_names_out(poly_cols)
        poly_df = pd.DataFrame(poly_values, columns=poly_feature_names, index=df.index)
        df = pd.concat([df, poly_df], axis=1)
    for col in features_model:
        if col not in df.columns:
            df[col] = 0
    df = df[features_model]
    return df

# Maintenant que preprocess est défini, initialiser l'explainer
try:
    if hasattr(model, 'predict_proba'):
        sample_size = min(100, len(test_df))  # Réduire la taille pour accélérer
        sample_data = test_df.sample(sample_size, random_state=42)
        sample_processed = preprocess(sample_data, features, poly_transformer)
        sample_imputed = imputer.transform(sample_processed)
        sample_scaled = scaler.transform(sample_imputed)
        
        try:
            # Utiliser TreeExplainer sans vérification d'additivité
            explainer = shap.TreeExplainer(model, check_additivity=False)
            logger.info("Explainer SHAP initialisé avec TreeExplainer.")
        except Exception as e1:
            logger.warning(f"TreeExplainer a échoué: {e1}")
            explainer = None
    else:
        logger.warning("Le modèle ne supporte pas predict_proba, SHAP ne sera pas disponible.")
        explainer = None
except Exception as e:
    logger.error(f"Erreur générale lors de l'initialisation de SHAP: {e}")
    explainer = None

app = Flask(__name__)

# Configuration Swagger
SWAGGER_URL = '/api/docs'  # URL pour accéder à la documentation Swagger
API_URL = '/static/swagger.json'  # URL vers le fichier de spécification de l'API

# Créer le blueprint pour Swagger UI
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Credit Scoring API"
    }
)

# Enregistrer le blueprint
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/static/swagger.json')
def get_swagger():
    """Sert le fichier de spécification swagger"""
    with open('static/swagger.json', 'r') as f:
        return jsonify(json.load(f))

@app.route('/predict/<int:client_id>', methods=['GET'])
def predict_client(client_id):
    """
    Prédit la probabilité de défaut pour un client spécifique.
    ---
    parameters:
      - name: client_id
        in: path
        type: integer
        required: true
        description: Identifiant unique du client
    responses:
      200:
        description: Prédiction réussie
      404:
        description: Client introuvable
      500:
        description: Erreur interne
    """
    try:
        # Validation du client_id
        if client_id <= 0:
            return jsonify({
                "erreur": "ID client invalide",
                "status": "INVALID_REQUEST"
            }), 400
            
        # Récupérer les données depuis GitHub (avec cache)
        test_df = fetch_github_data()
        
        client_row = test_df[test_df['SK_ID_CURR'] == client_id]
        if client_row.empty:
            logger.warning(f"Client ID {client_id} introuvable.")
            return jsonify({
                "erreur": f"Client ID {client_id} introuvable",
                "status": "NOT_FOUND"
            }), 404

        client_processed = preprocess(client_row, features, poly_transformer)
        X_imputed = imputer.transform(client_processed)
        X_scaled = scaler.transform(X_imputed)
        proba = model.predict_proba(X_scaled)[0, 1]
        decision = "REFUSÉ" if proba >= threshold else "ACCEPTÉ"

        logger.info(f"Prédiction réalisée pour client {client_id} : proba={proba:.4f}, décision={decision}")

        return jsonify({
            "client_id": int(client_id),
            "probability": float(proba),
            "threshold": float(threshold),
            "decision": decision,
            "model_name": model_name,
            "status": "OK"
        })
    except BadRequest as e:
        logger.error(f"Erreur de requête pour client {client_id} : {e}")
        return jsonify({
            "erreur": "Requête invalide",
            "details": str(e),
            "status": "INVALID_REQUEST"
        }), 400
    except Exception as e:
        logger.error(f"Erreur interne lors de la prédiction pour client {client_id} : {e}")
        return jsonify({
            "erreur": "Erreur interne du serveur",
            "details": str(e),
            "status": "ERROR"
        }), 500

# --- ENDPOINTS POST CORRIGÉS POUR LA SIMULATION ---
@app.route('/predict', methods=['POST'])
def predict_from_features():
    """
    Prédit la probabilité de défaut depuis un jeu de features transmis en JSON.
    Body attendu : { "features": { "AMT_CREDIT": 400000, ... }, "client_id": 100001 (optionnel) }
    """
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({"erreur": "Payload JSON vide", "status": "INVALID_REQUEST"}), 400

        features_payload = payload.get('features', {})
        client_id = payload.get('client_id', None)

        if not features_payload:
            return jsonify({"erreur": "Aucune feature fournie", "status": "INVALID_REQUEST"}), 400

        # 🔧 CORRECTION : Gestion améliorée des modifications de features
        if client_id is not None:
            try:
                cid = int(client_id)
                df = fetch_github_data()
                client_row = df[df['SK_ID_CURR'] == cid]
                if not client_row.empty:
                    # ✅ PRENDRE TOUTES LES FEATURES DU CLIENT ORIGINAL
                    base = client_row.iloc[0].to_dict()
                    
                    # ✅ APPLIQUER SEULEMENT LES MODIFICATIONS ENVOYÉES
                    for feature_name, new_value in features_payload.items():
                        if feature_name in base:
                            old_value = base[feature_name]
                            base[feature_name] = new_value
                            logger.info(f"Modification appliquée: {feature_name} {old_value} -> {new_value}")
                    
                    df_row = pd.DataFrame([base])
                else:
                    return jsonify({"erreur": f"Client {client_id} non trouvé", "status": "NOT_FOUND"}), 404
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du client {client_id}: {e}")
                return jsonify({"erreur": f"Erreur client: {str(e)}", "status": "ERROR"}), 500
        else:
            # Nouveau client : utiliser uniquement les features fournies
            df_row = pd.DataFrame([features_payload])

        # ✅ PREPROCESSING OBLIGATOIRE
        processed = preprocess(df_row, features, poly_transformer)
        X_imputed = imputer.transform(processed)
        X_scaled = scaler.transform(X_imputed)

        proba = float(model.predict_proba(X_scaled)[0, 1])
        decision = "REFUSÉ" if proba >= threshold else "ACCEPTÉ"

        resp = {
            "client_id": int(client_id) if client_id is not None else None,
            "probability": proba,
            "threshold": float(threshold),
            "decision": decision,
            "model_name": model_name,
            "status": "OK",
            "input_features": features_payload
        }
        logger.info(f"POST /predict -> proba={proba:.4f} client_id={client_id} (features modifiées: {list(features_payload.keys())})")
        return jsonify(resp)
        
    except Exception as e:
        logger.exception(f"Erreur interne POST /predict : {e}")
        return jsonify({"erreur": "Erreur interne du serveur", "details": str(e), "status": "ERROR"}), 500


@app.route('/predict/<int:client_id>', methods=['POST'])
def predict_with_clientid_and_features(client_id):
    """
    Prédit en utilisant les données existantes d'un client en appliquant des overrides.
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}
        features_overrides = payload.get('features', {})

        # ✅ RÉCUPÉRER LE CLIENT OBLIGATOIREMENT
        test_df_local = fetch_github_data()
        client_row = test_df_local[test_df_local['SK_ID_CURR'] == client_id]

        if client_row.empty:
            return jsonify({"erreur": f"Client ID {client_id} introuvable", "status": "NOT_FOUND"}), 404

        # ✅ PARTIR DES DONNÉES COMPLÈTES DU CLIENT
        base = client_row.iloc[0].to_dict()
        
        # ✅ APPLIQUER LES MODIFICATIONS UNE PAR UNE
        modifications_appliquees = {}
        for feature_name, new_value in features_overrides.items():
            if feature_name in base:
                old_value = base[feature_name]
                base[feature_name] = new_value
                modifications_appliquees[feature_name] = {"old": old_value, "new": new_value}
                logger.info(f"Client {client_id}: {feature_name} {old_value} -> {new_value}")
        
        df_row = pd.DataFrame([base])

        # ✅ PREPROCESSING COMPLET
        processed = preprocess(df_row, features, poly_transformer)
        X_imputed = imputer.transform(processed)
        X_scaled = scaler.transform(X_imputed)

        proba = float(model.predict_proba(X_scaled)[0, 1])
        decision = "REFUSÉ" if proba >= threshold else "ACCEPTÉ"

        resp = {
            "client_id": int(client_id),
            "probability": proba,
            "threshold": float(threshold),
            "decision": decision,
            "model_name": model_name,
            "status": "OK",
            "input_features": features_overrides,
            "modifications_applied": modifications_appliquees  # Debug info
        }
        logger.info(f"POST /predict/{client_id} -> proba={proba:.4f} (modif: {list(features_overrides.keys())})")
        return jsonify(resp)
        
    except Exception as e:
        logger.exception(f"Erreur interne POST /predict/{client_id} : {e}")
        return jsonify({"erreur": "Erreur interne du serveur", "details": str(e), "status": "ERROR"}), 500

# --- FIN DES CORRECTIONS POUR LA SIMULATION ---

# NOUVEL ENDPOINT POUR LES VALEURS SHAP
@app.route('/shap_values/<int:client_id>', methods=['GET'])
def get_shap_values(client_id):
    """
    Calcule les valeurs SHAP locales pour un client spécifique.
    Ces valeurs expliquent la contribution de chaque feature à la prédiction.
    ---
    parameters:
      - name: client_id
        in: path
        type: integer
        required: true
        description: Identifiant unique du client
      - name: limit
        in: query
        type: integer
        required: false
        default: 20
        description: Nombre maximum de features à retourner
    responses:
      200:
        description: Valeurs SHAP calculées avec succès
      400:
        description: Requête invalide
      404:
        description: Client introuvable
      503:
        description: Explainer SHAP indisponible
    """
    try:
        # Validation du client_id
        if client_id <= 0:
            return jsonify({
                "erreur": "ID client invalide",
                "status": "INVALID_REQUEST"
            }), 400
            
        # Récupérer le paramètre limit
        try:
            limit = int(request.args.get('limit', 20))
            if limit <= 0:
                return jsonify({
                    "erreur": "Le paramètre 'limit' doit être positif",
                    "status": "INVALID_REQUEST"
                }), 400
        except ValueError:
            return jsonify({
                "erreur": "Le paramètre 'limit' doit être un entier",
                "status": "INVALID_REQUEST"
            }), 400
        
        # Vérifier si l'explainer est disponible
        if explainer is None:
            logger.warning(f"L'explainer SHAP n'est pas disponible pour le client {client_id}")
            return jsonify({
                "erreur": "L'explainer SHAP n'est pas disponible",
                "message": "Impossible de calculer les explications SHAP pour ce modèle",
                "status": "ERROR"
            }), 503  # Service temporairement indisponible
        
        # Récupérer les données du client
        test_df = fetch_github_data()
        client_row = test_df[test_df['SK_ID_CURR'] == client_id]
        if client_row.empty:
            logger.warning(f"Client ID {client_id} introuvable pour SHAP.")
            return jsonify({
                "erreur": f"Client ID {client_id} introuvable",
                "status": "NOT_FOUND"
            }), 404
        
        # Prétraiter les données comme pour la prédiction
        client_processed = preprocess(client_row, features, poly_transformer)
        X_imputed = imputer.transform(client_processed)
        X_scaled = scaler.transform(X_imputed)
        
        # Calculer les valeurs SHAP
        shap_values = explainer.shap_values(X_scaled)
        
        # Pour les modèles avec plusieurs classes, prendre la classe positive (défaut)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Classe positive (défaut de paiement)
        
        # Créer un dictionnaire des valeurs SHAP par feature
        shap_dict = {}
        
        # Mapper chaque valeur SHAP à sa feature
        for i, feature_name in enumerate(features):
            shap_dict[feature_name] = float(shap_values[0][i])
        
        # Récupérer les features avec les plus fortes valeurs SHAP (en valeur absolue)
        shap_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:limit]
        top_shap_dict = {k: v for k, v in shap_items}
        
        logger.info(f"Valeurs SHAP calculées pour client {client_id}")
        
        return jsonify({
            "client_id": int(client_id),
            "shap_values": top_shap_dict,
            "status": "OK"
        })
        
    except BadRequest as e:
        logger.error(f"Erreur de requête pour les valeurs SHAP du client {client_id}: {e}")
        return jsonify({
            "erreur": "Requête invalide",
            "details": str(e),
            "status": "INVALID_REQUEST"
        }), 400
    except Exception as e:
        logger.error(f"Erreur lors du calcul des valeurs SHAP pour client {client_id}: {e}")
        return jsonify({
            "erreur": "Erreur lors du calcul des valeurs SHAP",
            "details": str(e),
            "status": "ERROR"
        }), 500

# NOUVEL ENDPOINT POUR LES VALEURS SHAP AVEC MAPPING
@app.route('/shap_values_mapped/<int:client_id>', methods=['GET'])
def get_shap_values_mapped(client_id):
    """
    Calcule les valeurs SHAP locales pour un client spécifique et les mappe aux données client.
    Cette version ajoute des informations supplémentaires pour faciliter l'utilisation par le frontend.
    ---
    parameters:
      - name: client_id
        in: path
        type: integer
        required: true
        description: Identifiant unique du client
      - name: limit
        in: query
        type: integer
        required: false
        default: 20
        description: Nombre maximum de features à retourner
    responses:
      200:
        description: Valeurs SHAP calculées et mappées avec succès
      400:
        description: Requête invalide
      404:
        description: Client introuvable
      503:
        description: Explainer SHAP indisponible
    """
    try:
        # Validation du client_id
        if client_id <= 0:
            return jsonify({
                "erreur": "ID client invalide",
                "status": "INVALID_REQUEST"
            }), 400
            
        # Récupérer le paramètre limit
        try:
            limit = int(request.args.get('limit', 20))
            if limit <= 0:
                return jsonify({
                    "erreur": "Le paramètre 'limit' doit être positif",
                    "status": "INVALID_REQUEST"
                }), 400
        except ValueError:
            return jsonify({
                "erreur": "Le paramètre 'limit' doit être un entier",
                "status": "INVALID_REQUEST"
            }), 400
        
        # Vérifier si l'explainer est disponible
        if explainer is None:
            logger.warning(f"L'explainer SHAP n'est pas disponible pour le client {client_id}")
            return jsonify({
                "erreur": "L'explainer SHAP n'est pas disponible",
                "message": "Impossible de calculer les explications SHAP pour ce modèle",
                "status": "ERROR"
            }), 503  # Service temporairement indisponible
        
        # Récupérer les données du client
        test_df = fetch_github_data()
        client_row = test_df[test_df['SK_ID_CURR'] == client_id]
        if client_row.empty:
            logger.warning(f"Client ID {client_id} introuvable pour SHAP.")
            return jsonify({
                "erreur": f"Client ID {client_id} introuvable",
                "status": "NOT_FOUND"
            }), 404
        
        # Prétraiter les données comme pour la prédiction
        client_processed = preprocess(client_row, features, poly_transformer)
        X_imputed = imputer.transform(client_processed)
        X_scaled = scaler.transform(X_imputed)
        
        # Calculer les valeurs SHAP
        shap_values = explainer.shap_values(X_scaled)
        
        # Pour les modèles avec plusieurs classes, prendre la classe positive (défaut)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Classe positive (défaut de paiement)
        
        # Créer un dictionnaire des valeurs SHAP par feature
        shap_dict = {}
        
        # Mapper chaque valeur SHAP à sa feature
        for i, feature_name in enumerate(features):
            shap_dict[feature_name] = float(shap_values[0][i])
        
        # Récupérer les features avec les plus fortes valeurs SHAP (en valeur absolue)
        shap_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:limit]
        
        # Récupérer les détails du client pour mapper les valeurs SHAP
        response_client = get_client_details(client_id)
        if response_client.status_code != 200:
            return response_client
        
        client_details = json.loads(response_client.data)
        
        # Préparer les résultats mappés
        mapped_results = []
        
        for feature_name, shap_value in shap_items:
            # Déterminer le nom d'affichage pour la feature
            display_name = FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
            
            # Préparer un dictionnaire pour ce résultat
            result = {
                "feature": feature_name,
                "display_name": display_name,
                "shap_value": float(shap_value),
                "impact_direction": "Favorable" if shap_value < 0 else "Défavorable",
                "impact_value": abs(round(shap_value, 4)),
                "real_value": "N/A"
            }
            
            # Essayer de trouver la valeur réelle
            if feature_name in FEATURE_MAPPING:
                mapping = FEATURE_MAPPING[feature_name]
                
                if "computed" in mapping and mapping["computed"]:
                    # Pour les valeurs calculées à la volée
                    try:
                        result["real_value"] = mapping["formula"](client_details)
                    except Exception as e:
                        logger.warning(f"Erreur lors du calcul de la valeur réelle pour {feature_name}: {e}")
                        result["real_value"] = "N/A"
                else:
                    # Pour les valeurs directement accessibles
                    section = mapping.get("section")
                    key = mapping.get("key")
                    
                    if section in client_details and key in client_details[section]:
                        value = client_details[section][key]
                        
                        # Appliquer une transformation si nécessaire
                        if "transform" in mapping and callable(mapping["transform"]):
                            try:
                                value = mapping["transform"](value)
                            except Exception as e:
                                logger.warning(f"Erreur lors de la transformation pour {feature_name}: {e}")
                        
                        result["real_value"] = value
            
            # Formater la valeur si c'est un nombre
            if isinstance(result["real_value"], (int, float)):
                result["real_value"] = round(result["real_value"], 2)
            
            mapped_results.append(result)
        
        logger.info(f"Valeurs SHAP mappées calculées pour client {client_id}")
        
        return jsonify({
            "client_id": int(client_id),
            "mapped_shap_values": mapped_results,
            "status": "OK"
        })
        
    except BadRequest as e:
        logger.error(f"Erreur de requête pour les valeurs SHAP mappées du client {client_id}: {e}")
        return jsonify({
            "erreur": "Requête invalide",
            "details": str(e),
            "status": "INVALID_REQUEST"
        }), 400
    except Exception as e:
        logger.error(f"Erreur lors du calcul des valeurs SHAP mappées pour client {client_id}: {e}")
        return jsonify({
            "erreur": "Erreur lors du calcul des valeurs SHAP mappées",
            "details": str(e),
            "status": "ERROR"
        }), 500

# Ajouter un endpoint pour les clients disponibles
@app.route('/clients', methods=['GET'])
def get_available_clients():
    """
    Retourne la liste des IDs clients disponibles.
    ---
    parameters:
      - name: limit
        in: query
        type: integer
        required: false
        default: 100
        description: Nombre maximum de clients à retourner
      - name: offset
        in: query
        type: integer
        required: false
        default: 0
        description: Index de départ pour la pagination
    responses:
      200:
        description: Liste des clients récupérée avec succès
      400:
        description: Paramètres de requête invalides
      500:
        description: Erreur interne du serveur
    """
    try:
        # Validation des paramètres
        try:
            limit = int(request.args.get('limit', 100))
            offset = int(request.args.get('offset', 0))
            if limit <= 0 or offset < 0:
                return jsonify({
                    "erreur": "Les paramètres 'limit' et 'offset' doivent être positifs",
                    "status": "INVALID_REQUEST"
                }), 400
        except ValueError:
            return jsonify({
                "erreur": "Les paramètres 'limit' et 'offset' doivent être des entiers",
                "status": "INVALID_REQUEST"
            }), 400
        
        # Récupérer les données depuis GitHub (avec cache)
        test_df = fetch_github_data()
        
        client_ids = test_df['SK_ID_CURR'].tolist()
        paginated_ids = client_ids[offset:offset+limit]
        
        return jsonify({
            "client_ids": paginated_ids,
            "total": len(client_ids),
            "limit": limit,
            "offset": offset,
            "status": "OK"
        })
    except BadRequest as e:
        logger.error(f"Erreur de requête pour la liste des clients: {e}")
        return jsonify({
            "erreur": "Requête invalide",
            "details": str(e),
            "status": "INVALID_REQUEST"
        }), 400
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des IDs clients: {e}")
        return jsonify({
            "erreur": "Erreur lors de la récupération des IDs clients",
            "details": str(e),
            "status": "ERROR"
        }), 500

# Nouvel endpoint pour récupérer les détails d'un client
@app.route('/client/<int:client_id>/details', methods=['GET'])
def get_client_details(client_id):
    """
    Renvoie les détails d'un client spécifique pour l'affichage dans le dashboard.
    """
    try:
        # Validation du client_id
        if client_id <= 0:
            return jsonify({
                "erreur": "ID client invalide",
                "status": "INVALID_REQUEST"
            }), 400
            
        # Récupérer les données depuis GitHub (avec cache)
        test_df = fetch_github_data()
        
        client_row = test_df[test_df['SK_ID_CURR'] == client_id]
        if client_row.empty:
            return jsonify({
                "erreur": f"Client ID {client_id} introuvable",
                "status": "NOT_FOUND"
            }), 404
        
        # Extraire les informations pertinentes
        client_data = client_row.iloc[0].to_dict()
        
        # Organiser les données en catégories pour l'interface
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
            "credit_term": int(float(client_data.get('AMT_CREDIT', 0)) / float(client_data.get('AMT_ANNUITY', 1))) if 'AMT_ANNUITY' in client_data and float(client_data.get('AMT_ANNUITY', 0)) > 0 else 0,
            "annuity": float(client_data.get('AMT_ANNUITY', 0)),
            "goods_price": float(client_data.get('AMT_GOODS_PRICE', 0)),
            "name_goods_category": client_data.get('NAME_CONTRACT_TYPE', ''),
        }
        
        # Ajouter quelques données d'historique simulées pour l'interface
        credit_history = {
            "previous_loans_count": 1,
            "previous_defaults": 0,
            "late_payments": 0,
            "credit_score": int(700 * (1 - float(client_data.get('EXT_SOURCE_3', 0.5)))) if 'EXT_SOURCE_3' in client_data else "N/A",
            "years_with_bank": 3
        }
        
        # Extraire les features brutes pour les visualisations
        features_raw = {}
        for key in client_data.keys():
            # Inclure toutes les features importantes pour SHAP
            if key in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
                      'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL',
                      'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH',
                      'AMT_GOODS_PRICE', 'CODE_GENDER', 'FLAG_OWN_CAR',
                      'FLAG_OWN_REALTY', 'REGION_RATING_CLIENT',
                      'REGION_RATING_CLIENT_W_CITY', 'DAYS_REGISTRATION',
                      'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG']:
                # Convertir en float si possible, sinon garder la valeur telle quelle
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
        logger.error(f"Erreur de requête pour les détails du client {client_id}: {e}")
        return jsonify({
            "erreur": "Requête invalide",
            "details": str(e),
            "status": "INVALID_REQUEST"
        }), 400
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des détails du client {client_id}: {e}")
        return jsonify({
            "erreur": "Erreur lors de la récupération des détails du client",
            "details": str(e),
            "status": "ERROR"
        }), 500

# Endpoint pour vérifier l'état de l'API
@app.route('/health', methods=['GET'])
def health_check():
    """
    Vérifie si l'API est fonctionnelle.
    ---
    responses:
      200:
        description: API fonctionnelle
      500:
        description: Problème avec l'API
    """
    try:
        # Vérifier si les composants critiques sont chargés
        if model is None or features is None:
            return jsonify({
                "status": "ERROR",
                "message": "Modèle ou features non chargés"
            }), 500
        
        return jsonify({
            "status": "OK",
            "version": "1.0.0",
            "model": model_name,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Erreur lors du health check: {e}")
        return jsonify({
            "status": "ERROR",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Gère les erreurs 404 (page non trouvée)"""
    logger.warning(f"Erreur 404: {request.path}")
    return jsonify({
        "erreur": "Ressource introuvable",
        "path": request.path,
        "status": "NOT_FOUND"
    }), 404

@app.errorhandler(400)
def bad_request(error):
    """Gère les erreurs 400 (mauvaise requête)"""
    logger.warning(f"Erreur 400: {error}")
    return jsonify({
        "erreur": "Requête invalide",
        "details": str(error),
        "status": "INVALID_REQUEST"
    }), 400

@app.errorhandler(500)
def internal_error(error):
    """Gère les erreurs 500 (erreur interne du serveur)"""
    logger.error(f"Erreur 500 : {error}")
    return jsonify({
        "erreur": "Erreur interne du serveur",
        "status": "ERROR"
    }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5800)
