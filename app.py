import os
from flask import Flask, jsonify, request
from joblib import load
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import logging
import shap  # Ajout de la bibliothèque SHAP

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Chemin absolu pour les artefacts
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model_complet.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "imputer.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")
TEST_CSV_PATH = os.path.join(BASE_DIR, "application_test.csv")

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
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    logging.info("Modèle et artefacts chargés avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle ou des artefacts : {e}")
    raise

def preprocess(df, features_model, poly_transformer=None):
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
            logging.info("Explainer SHAP initialisé avec TreeExplainer.")
        except Exception as e1:
            logging.warning(f"TreeExplainer a échoué: {e1}")
            explainer = None
    else:
        logging.warning("Le modèle ne supporte pas predict_proba, SHAP ne sera pas disponible.")
        explainer = None
except Exception as e:
    logging.error(f"Erreur générale lors de l'initialisation de SHAP: {e}")
    explainer = None

app = Flask(__name__)

@app.route('/predict/<int:client_id>', methods=['GET'])
def predict_client(client_id):
    try:
        client_row = test_df[test_df['SK_ID_CURR'] == client_id]
        if client_row.empty:
            logging.warning(f"Client ID {client_id} introuvable.")
            return jsonify({
                "erreur": f"Client ID {client_id} introuvable",
                "status": "NOT_FOUND"
            }), 404

        client_processed = preprocess(client_row, features, poly_transformer)
        X_imputed = imputer.transform(client_processed)
        X_scaled = scaler.transform(X_imputed)
        proba = model.predict_proba(X_scaled)[0, 1]
        decision = "REFUSÉ" if proba >= threshold else "ACCEPTÉ"

        logging.info(f"Prédiction réalisée pour client {client_id} : proba={proba:.4f}, décision={decision}")

        return jsonify({
            "client_id": int(client_id),
            "probability": float(proba),
            "threshold": float(threshold),
            "decision": decision,
            "model_name": model_name,
            "status": "OK"
        })
    except Exception as e:
        logging.error(f"Erreur interne lors de la prédiction pour client {client_id} : {e}")
        return jsonify({
            "erreur": "Erreur interne du serveur",
            "details": str(e),
            "status": "ERROR"
        }), 500

# NOUVEL ENDPOINT POUR LES VALEURS SHAP
@app.route('/shap_values/<int:client_id>', methods=['GET'])
def get_shap_values(client_id):
    """
    Calcule les valeurs SHAP locales pour un client spécifique.
    Ces valeurs expliquent la contribution de chaque feature à la prédiction.
    """
    try:
        # Vérifier si l'explainer est disponible
        if explainer is None:
            logging.warning(f"L'explainer SHAP n'est pas disponible pour le client {client_id}")
            return jsonify({
                "erreur": "L'explainer SHAP n'est pas disponible",
                "message": "Impossible de calculer les explications SHAP pour ce modèle",
                "status": "ERROR"
            }), 503  # Service temporairement indisponible
        
        # Récupérer les données du client
        client_row = test_df[test_df['SK_ID_CURR'] == client_id]
        if client_row.empty:
            logging.warning(f"Client ID {client_id} introuvable pour SHAP.")
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
        
        # Récupérer les 20 features avec les plus fortes valeurs SHAP (en valeur absolue)
        shap_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        top_shap_dict = {k: v for k, v in shap_items}
        
        logging.info(f"Valeurs SHAP calculées pour client {client_id}")
        
        return jsonify({
            "client_id": int(client_id),
            "shap_values": top_shap_dict,
            "status": "OK"
        })
        
    except Exception as e:
        logging.error(f"Erreur lors du calcul des valeurs SHAP pour client {client_id}: {e}")
        return jsonify({
            "erreur": "Erreur lors du calcul des valeurs SHAP",
            "details": str(e),
            "status": "ERROR"
        }), 500

# Ajouter un endpoint pour les clients disponibles
@app.route('/clients', methods=['GET'])
def get_available_clients():
    try:
        limit = request.args.get('limit', default=100, type=int)
        offset = request.args.get('offset', default=0, type=int)
        
        client_ids = test_df['SK_ID_CURR'].tolist()
        paginated_ids = client_ids[offset:offset+limit]
        
        return jsonify({
            "client_ids": paginated_ids,
            "total": len(client_ids),
            "limit": limit,
            "offset": offset,
            "status": "OK"
        })
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des IDs clients: {e}")
        return jsonify({
            "erreur": "Erreur lors de la récupération des IDs clients",
            "details": str(e),
            "status": "ERROR"
        }), 500

# Nouvel endpoint pour récupérer les détails d'un client
@app.route('/client/<int:client_id>/details', methods=['GET'])
def get_client_details(client_id):
    """
    Renvoie les détails d'un client spécifique pour l'affichage dans le dashboard
    """
    try:
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
            "gender": "F" if client_data.get('CODE_GENDER') == "F" else "M",
            "age": int(abs(client_data.get('DAYS_BIRTH', 0)) / 365.25) if 'DAYS_BIRTH' in client_data else None,
            "education": client_data.get('NAME_EDUCATION_TYPE'),
            "family_status": client_data.get('NAME_FAMILY_STATUS'),
            "children_count": int(client_data.get('CNT_CHILDREN', 0)),
            "family_size": int(client_data.get('CNT_FAM_MEMBERS', 1)),
            "income": float(client_data.get('AMT_INCOME_TOTAL', 0)),
            "employment_type": client_data.get('NAME_INCOME_TYPE'),
            "employment_years": int(abs(client_data.get('DAYS_EMPLOYED', 0)) / 365.25) if ('DAYS_EMPLOYED' in client_data and client_data.get('DAYS_EMPLOYED') != 365243) else 0,
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
            if key in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
                      'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL',
                      'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH',
                      'AMT_GOODS_PRICE', 'CODE_GENDER']:
                features_raw[key] = float(client_data.get(key, 0))
        
        response = {
            "client_id": int(client_id),
            "personal_info": personal_info,
            "credit_info": credit_info,
            "credit_history": credit_history,
            "features": features_raw,
            "status": "OK"
        }
        
        return jsonify(response)
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des détails du client {client_id}: {e}")
        return jsonify({
            "erreur": "Erreur lors de la récupération des détails du client",
            "details": str(e),
            "status": "ERROR"
        }), 500

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Erreur 500 : {error}")
    return jsonify({"erreur": "Erreur interne du serveur"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5800)