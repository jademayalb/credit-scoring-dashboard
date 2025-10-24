# api/app.py
"""
# Full Flask application for prediction endpoints.
# Replaces previous implementation: POST /predict and POST /predict/<client_id> now rebuild a profile
# from base_row + overrides, run preprocess -> imputer -> scaler -> model.predict_proba and return
# updated probability. Includes a debug mode returning processed / imputed / scaled vectors.

import os
import glob
import logging
import traceback
import json
from pathlib import Path

from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# Create Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals that will be loaded at startup
model = None
imputer = None
scaler = None
features = None
poly_transformer = None
preprocess = None
test_df = None
threshold = None
model_name = "unknown_model"

ARTIFACT_DIRS = ["artifacts", "models", "api/artifacts", "api/models", "."]

def find_file(patterns):
    """Search common artifact directories for files matching any of the glob patterns.
    Return first match or None."""
    for d in ARTIFACT_DIRS:
        for pat in patterns:
            path = os.path.join(d, pat)
            matches = glob.glob(path)
            if matches:
                return matches[0]
    return None

def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception:
        logger.exception(f"Failed to joblib.load {path}")
        return None

def load_artifacts():
    global model, imputer, scaler, features, poly_transformer, preprocess, test_df, threshold, model_name

    # Model
    model_path = find_file(["model*.joblib", "model*.pkl", "best_model*.joblib", "clf*.joblib"])
    if model_path:
        model = safe_load_joblib(model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning("No model file found with common patterns. Ensure model file exists.")

    # Imputer
    imputer_path = find_file(["imputer*.joblib", "imputer*.pkl"]) or find_file(["preprocessor*.joblib"]) 
    if imputer_path:
        imputer = safe_load_joblib(imputer_path)
        logger.info(f"Loaded imputer from {imputer_path}")
    else:
        logger.warning("No imputer file found with common patterns. Imputer may be embedded in pipeline.")

    # Scaler
    scaler_path = find_file(["scaler*.joblib", "scaler*.pkl"]) 
    if scaler_path:
        scaler = safe_load_joblib(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
    else:
        logger.warning("No scaler file found with common patterns. Scaler may be embedded in pipeline.")

    # Poly transformer
    poly_path = find_file(["poly_transformer*.joblib", "poly_transformer*.pkl", "poly*.joblib", "poly*.pkl"]) 
    if poly_path:
        poly_transformer = safe_load_joblib(poly_path)
        logger.info(f"Loaded poly_transformer from {poly_path}")
    else:
        logger.info("No poly_transformer found; polynomial features may not be used.")

    # Features list
    features_path = find_file(["features*.joblib", "features*.pkl", "features*.json"]) 
    if features_path:
        try:
            if features_path.endswith('.json'):
                with open(features_path) as f:
                    features = json.load(f)
            else:
                features = joblib.load(features_path)
            logger.info(f"Loaded features list from {features_path} (len={len(features)})")
        except Exception:
            logger.exception(f"Failed to load features from {features_path}")
            features = None
    else:
        logger.warning("No features file found; will attempt to infer from preprocess or model.")

    # Preprocess callable (try to import or load)
    # Try common module locations for a preprocess function
    try:
        # If there is a module named preprocess.py in repo
        import preprocess as pp
        if hasattr(pp, 'preprocess'):
            preprocess = pp.preprocess
            logger.info("Imported preprocess.preprocess from preprocess.py module")
    except Exception:
        # ignore
        pass

    # Try loading preprocessor callable pickled
    if preprocess is None:
        preproc_path = find_file(["preprocess*.joblib", "preprocess*.pkl", "preprocessor*.joblib", "preprocessor*.pkl"]) 
        if preproc_path:
            loaded = safe_load_joblib(preproc_path)
            # If it's a sklearn Pipeline, try to use its transform method
            if callable(loaded):
                preprocess = loaded
                logger.info(f"Loaded callable preprocess from {preproc_path}")
            elif hasattr(loaded, 'transform'):
                def _callable_preprocess(df, target_features=None, poly=None):
                    return pd.DataFrame(loaded.transform(df), columns=(target_features or list(df.columns)))
                preprocess = _callable_preprocess
                logger.info(f"Loaded transformer preprocess from {preproc_path}")

    # Test dataframe (used for GET of existing clients)
    test_df_path = find_file(["test_df*.joblib", "test_df*.pkl", "test_df*.csv"]) 
    if test_df_path:
        try:
            if test_df_path.endswith('.csv'):
                test_df = pd.read_csv(test_df_path)
            else:
                test_df = joblib.load(test_df_path)
            logger.info(f"Loaded test_df from {test_df_path} (rows={len(test_df)})")
        except Exception:
            logger.exception(f"Failed to load test_df from {test_df_path}")
            test_df = None
    else:
        logger.warning("No test_df file found; GET /predict/<id> will not work without test data.")

    # Threshold
    threshold_path = find_file(["threshold*.json", "threshold*.joblib", "threshold*.pkl"]) 
    if threshold_path:
        try:
            if threshold_path.endswith('.json'):
                with open(threshold_path) as f:
                    threshold = json.load(f).get('threshold')
            else:
                threshold = joblib.load(threshold_path)
            logger.info(f"Loaded threshold from {threshold_path}: {threshold}")
        except Exception:
            logger.exception(f"Failed to load threshold from {threshold_path}")
            threshold = None
    else:
        logger.info("No threshold file found; defaulting to 0.5")
        threshold = 0.5

    # Model name
    name_path = find_file(["model_name*.txt", "model_name*.json"]) 
    if name_path:
        try:
            if name_path.endswith('.json'):
                with open(name_path) as f:
                    model_name = json.load(f).get('model_name', model_name)
            else:
                with open(name_path) as f:
                    model_name = f.read().strip()
            logger.info(f"Model name set to {model_name}")
        except Exception:
            logger.exception("Failed to read model_name file")

    # If model is a pipeline containing preprocessing, attempt to extract pieces if missing
    try:
        from sklearn.pipeline import Pipeline
        if model is not None and isinstance(model, Pipeline):
            # If imputer/scaler not set, try to find in pipeline
            for step_name, step in model.named_steps.items():
                if imputer is None and hasattr(step, 'fill_value') or (step_name and 'imput' in step_name):
                    imputer = step
                if scaler is None and ('scaler' in step_name or 'standard' in step_name or hasattr(step, 'scale_')):
                    scaler = step
            logger.info("Extracted imputer/scaler from model pipeline where possible.")
    except Exception:
        pass


# Initialize artifacts at startup
load_artifacts()


def _apply_pipeline(X_df):
    """Apply preprocess -> imputer -> scaler -> model.predict_proba and return proba and intermediate arrays.
    X_df is a single-row DataFrame already constructed from merged inputs."""
    if preprocess is None:
        # If no preprocess callable, try to align columns to features
        if features is not None:
            # Ensure all features present
            missing = [c for c in features if c not in X_df.columns]
            for m in missing:
                X_df[m] = np.nan
            X_proc = X_df[features]
        else:
            # fallback: use X_df as-is
            X_proc = X_df
    else:
        # Prefer calling preprocess with (df, features, poly_transformer) signature if possible
        try:
            X_proc = preprocess(X_df, features, poly_transformer)
        except TypeError:
            try:
                X_proc = preprocess(X_df)
            except Exception:
                logger.exception("preprocess callable failed with both signatures")
                raise

    # Convert to numpy arrays via imputer/scaler
    X_imputed = None
    X_scaled = None

    # Imputer
    if imputer is not None:
        try:
            X_imputed = imputer.transform(X_proc)
        except Exception:
            logger.exception("Imputer transformation failed")
            raise
    else:
        # If no imputer, convert DataFrame to numpy (may contain NaNs)
        X_imputed = X_proc.values

    # Scaler
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X_imputed)
        except Exception:
            logger.exception("Scaler transformation failed")
            raise
    else:
        X_scaled = X_imputed

    # Predict
    if model is None:
        raise RuntimeError("No model loaded - cannot predict")

    try:
        proba = float(model.predict_proba(X_scaled)[0, 1])
    except Exception:
        logger.exception("model.predict_proba failed")
        raise

    return X_proc, X_imputed, X_scaled, proba


@app.route('/')
def index():
    return jsonify({
        "status": "ok",
        "message": "Prediction API",
        "model_loaded": model is not None,
        "features_count": len(features) if features is not None else None,
        "threshold": float(threshold) if threshold is not None else None,
        "model_name": model_name
    })


@app.route('/predict/<int:client_id>', methods=['GET'])
def predict_client_get(client_id):
    """GET existing client prediction: fetch row from test_df and run pipeline.
    Keeps behavior compatible with previous implementation."""
    try:
        if test_df is None:
            return jsonify({"error": "test_df not available on server", "status": "NOT_AVAILABLE"}), 500

        df_row = test_df[test_df['SK_ID_CURR'] == client_id]
        if df_row.empty:
            return jsonify({"error": f"Client ID {client_id} not found", "status": "NOT_FOUND"}), 404

        X_proc, X_imputed, X_scaled, proba = _apply_pipeline(df_row)
        decision = "REFUSÉ" if proba >= threshold else "ACCEPTÉ"

        return jsonify({
            "client_id": int(client_id),
            "probability": proba,
            "threshold": float(threshold),
            "decision": decision,
            "model_name": model_name,
            "status": "OK"
        })
    except Exception as e:
        logger.exception("Error in GET /predict/<client_id>")
        return jsonify({"error": "internal server error", "details": str(e), "traceback": traceback.format_exc(), "status": "ERROR"}), 500


@app.route('/predict', methods=['POST'])
@app.route('/predict/<int:client_id>', methods=['POST'])
def predict_with_overrides(client_id=None):
    """POST endpoint that accepts JSON with {"features": {...}, "debug": true}.

    If client_id is provided, load the base row from test_df and apply overrides.
    Otherwise build a single-row DataFrame from overrides alone.

    Returns recalculated probability after running full pipeline.
    """
    try:
        payload = request.get_json(silent=True) or {}
        overrides = payload.get('features', {}) or {}
        debug_flag = bool(payload.get('debug', False)) or (request.args.get('debug') in ('1', 'true', 'True'))

        # Base row
        base_row = {}
        if client_id is not None:
            if test_df is None:
                return jsonify({"error": "test_df not available on server", "status": "NOT_AVAILABLE"}), 500
            df_row = test_df[test_df['SK_ID_CURR'] == client_id]
            if df_row.empty:
                return jsonify({"error": f"Client ID {client_id} not found", "status": "NOT_FOUND"}), 404
            base_row = df_row.iloc[0].to_dict()

        # Merge overrides
        merged = dict(base_row)
        for k, v in (overrides.items() if isinstance(overrides, dict) else []):
            merged[k] = v

        # Harmonize common UI-friendly fields
        try:
            if "DAYS_BIRTH" in merged:
                v = merged["DAYS_BIRTH"]
                if isinstance(v, (int, float)) and 0 < float(v) < 150:
                    merged["DAYS_BIRTH"] = int(round(-abs(float(v)) * 365))
        except Exception:
            logger.exception("Failed to harmonize DAYS_BIRTH")

        input_df = pd.DataFrame([merged])

        # Run pipeline
        try:
            X_proc, X_imputed, X_scaled, proba = _apply_pipeline(input_df)
        except Exception as e:
            logger.exception("Pipeline application failed for merged input")
            return jsonify({
                "status": "error",
                "message": "Pipeline failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "merged_input": merged
            }), 500

        decision = "REFUSÉ" if proba >= threshold else "ACCEPTÉ"

        resp = {
            "client_id": int(client_id) if client_id is not None else merged.get('SK_ID_CURR'),
            "probability": proba,
            "threshold": float(threshold),
            "decision": decision,
            "model_name": model_name,
            "raw_data": {"input_features": merged},
            "status": "OK"
        }

        if debug_flag:
            # processed
            try:
                if isinstance(X_proc, pd.DataFrame):
                    resp['raw_data']['processed_vector'] = X_proc.iloc[0].to_dict()
                else:
                    resp['raw_data']['processed_vector'] = list(X_proc[0])
            except Exception:
                resp['raw_data']['processed_vector'] = None

            # imputed
            try:
                resp['raw_data']['imputed_vector'] = X_imputed[0].tolist() if hasattr(X_imputed, 'shape') else None
            except Exception:
                resp['raw_data']['imputed_vector'] = None

            # scaled
            try:
                resp['raw_data']['scaled_vector'] = X_scaled[0].tolist() if hasattr(X_scaled, 'shape') else None
            except Exception:
                resp['raw_data']['scaled_vector'] = None

        return jsonify(resp), 200

    except Exception as e:
        logger.exception("Unexpected error in POST /predict")
        return jsonify({"status": "error", "message": "Unexpected server error", "error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == '__main__':
    # For local testing only; production should use gunicorn/uwsgi
    port = int(os.environ.get('PORT', 5800))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', '0') == '1')
