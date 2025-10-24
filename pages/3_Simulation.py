"""
Page de simulation : modifier/entrer un profil client et obtenir un score rafraîchi.

Modifications importantes (par rapport à la version précédente) :
- Affiche la prédiction "avant" (si client existant) et la prédiction "après" (résultat de la simulation)
  côte à côte pour faciliter la comparaison.
- Affiche les erreurs/retours détaillés de l'appel API pour diagnostiquer pourquoi rien ne se passe.
- Affiche le payload envoyé quand l'option debug est cochée.
- Comportement défensif et messages explicites si l'API ne renvoie rien.
- Conserve accessibilité et champs éditables restreints.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

from utils.api_client import get_client_prediction, get_client_details, get_available_clients
from config import FEATURE_DESCRIPTIONS, COLORBLIND_FRIENDLY_PALETTE, UI_CONFIG

st.set_page_config(page_title="Simulateur de scoring", page_icon="🧪", layout="wide")

st.title("Simulateur de scoring — Client existant ou nouveau profil")
st.markdown(
    "Choisissez un client existant ou saisissez un nouveau profil. Modifiez les valeurs des "
    "caractéristiques sélectionnées, puis cliquez sur « Obtenir score rafraîchi » pour appeler "
    "l'API et afficher la probabilité et la décision. Un comparatif Avant / Après est présenté."
)

# --- Features retenues pour la simulation (modifiable si besoin) ---
SIM_FEATURES = [
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "AMT_ANNUITY",
    "DAYS_BIRTH",       # shown as years (positive), converted to negative days for the model
    "EXT_SOURCE_3"
]

# Ensure descriptions exist
FEATURE_DESCRIPTIONS.setdefault("AMT_CREDIT", "Montant du crédit demandé.")
FEATURE_DESCRIPTIONS.setdefault("AMT_GOODS_PRICE", "Valeur du bien/service financé.")
FEATURE_DESCRIPTIONS.setdefault("AMT_ANNUITY", "Montant de l'annuité (mensualité).")
FEATURE_DESCRIPTIONS.setdefault("DAYS_BIRTH", "Âge exprimé en jours (négatif dans les données sources). L'interface affiche des années positives.")
FEATURE_DESCRIPTIONS.setdefault("EXT_SOURCE_3", "Score externe (valeur continue entre 0 et 1).")

st.markdown("<small>Champs accessibles : utilisez Tab pour naviguer. Les libellés contiennent des descriptions pour les lecteurs d'écran.</small>", unsafe_allow_html=True)

# ---------- Helpers ----------
def normalize_id_list(lst):
    out = []
    for v in lst or []:
        try:
            out.append(int(v))
        except Exception:
            try:
                out.append(int(str(v)))
            except Exception:
                out.append(v)
    return out

def display_prediction_block(pred: Dict[str, Any], title: str):
    """
    Affiche jauge + décision à partir du dictionnaire retourné par l'API.
    Attendu: pred contient 'probability' (float) et éventuellement 'decision' et 'threshold'.
    """
    if pred is None:
        st.info(f"{title} : pas de prédiction disponible.")
        return

    prob = pred.get("probability", None)
    threshold = pred.get("threshold", 0.5)
    decision = pred.get("decision", None)
    try:
        prob_val = float(prob)
    except Exception:
        prob_val = None

    if prob_val is None:
        st.warning(f"{title} : la réponse de l'API est invalide ou ne contient pas de probabilité.")
        st.write(pred)
        return

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_val,
        number={"valueformat": ".1%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 1], "tickformat": ".0%"},
            "bar": {"color": COLORBLIND_FRIENDLY_PALETTE.get("refused", "#d62728")},
            "steps": [
                {"range": [0, threshold * 0.5], "color": "rgba(1,133,113,0.3)"},
                {"range": [threshold * 0.5, threshold], "color": "rgba(1,133,113,0.6)"},
                {"range": [threshold, 1], "color": "rgba(166,97,26,0.6)"}
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": threshold}
        },
        title={"text": title, "font": {"size": 12}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Decision banner
    if decision is None:
        decision = "ACCEPTÉ" if prob_val < threshold else "REFUSÉ"
    color = COLORBLIND_FRIENDLY_PALETTE.get("accepted", "#2ca02c") if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get("refused", "#d62728")
    st.markdown(
        f"""
        <div role="status" aria-live="polite" style="padding:0.6rem; border-radius:6px; background:{color}20; border:1px solid {color};">
            <strong>{title} — Décision : {decision}</strong><br>
            Probabilité: <strong>{prob_val:.1%}</strong> (Seuil: {threshold:.2f})
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Load clients ----------
with st.spinner("Chargement de la liste des clients..."):
    try:
        avail = get_available_clients(limit=UI_CONFIG.get("default_limit", 200))
        available_clients = normalize_id_list(avail or [])
    except Exception:
        available_clients = []

# UI: mode and client selector
col_mode, col_select = st.columns([2, 1])
with col_mode:
    mode = st.radio("Mode", options=["Client existant", "Nouveau client"], index=0,
                    help="Choisissez 'Client existant' pour préremplir depuis un client ; sinon saisissez un nouveau profil.")
with col_select:
    if available_clients:
        selected_client = st.selectbox("Choisir un client existant", options=available_clients,
                                       format_func=lambda x: f"Client #{x}",
                                       disabled=(mode != "Client existant"))
    else:
        selected_client = None
        st.info("Liste clients indisponible.")

# If existing client, fetch its current prediction (before)
original_prediction = None
if mode == "Client existant" and selected_client is not None:
    with st.spinner(f"Chargement données et score du client #{selected_client}..."):
        try:
            details = get_client_details(int(selected_client))
            # Attempt to fetch current prediction (API may provide it)
            original_prediction = get_client_prediction(int(selected_client))
        except Exception as e:
            st.warning("Impossible de récupérer la prédiction initiale du client (API).")
            original_prediction = None

# Prepare defaults and prefill
defaults: Dict[str, Any] = {
    "AMT_CREDIT": 500000.0,
    "AMT_GOODS_PRICE": 600000.0,
    "AMT_ANNUITY": 15000.0,
    "DAYS_BIRTH": 35.0,   # years positive for UX
    "EXT_SOURCE_3": 0.5
}
prefill = {}
if mode == "Client existant" and selected_client is not None:
    try:
        details = details or get_client_details(int(selected_client))
        feats = (details or {}).get("features", {})
        for f in SIM_FEATURES:
            raw = feats.get(f, None)
            if raw is None:
                continue
            if f == "DAYS_BIRTH":
                try:
                    val = float(raw)
                    prefill[f] = round((-val) / 365.0, 2) if val not in (None, 365243) else defaults[f]
                except Exception:
                    prefill[f] = defaults[f]
            else:
                prefill[f] = raw
    except Exception:
        prefill = {}

values = {f: prefill.get(f, defaults.get(f)) for f in SIM_FEATURES}

# ---------- Form to edit features ----------
st.markdown("### Valeurs des caractéristiques (édition)")
with st.form(key="sim_form", clear_on_submit=False):
    input_cols = st.columns(2)
    widgets = {}
    for i, feat in enumerate(SIM_FEATURES):
        col = input_cols[i % 2]
        label = f"{feat} — {FEATURE_DESCRIPTIONS.get(feat, '')}"
        if feat in {"AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY"}:
            widgets[feat] = col.number_input(label, value=float(values.get(feat, 0.0) or 0.0),
                                             min_value=0.0, format="%.2f", step=100.0,
                                             help=FEATURE_DESCRIPTIONS.get(feat, ""))
        elif feat == "DAYS_BIRTH":
            widgets[feat] = col.number_input(label, value=float(values.get(feat, 35.0) or 35.0),
                                             min_value=16.0, max_value=120.0, format="%.1f",
                                             help="Saisir l'âge en années (valeur positive).")
        elif feat.startswith("EXT_SOURCE"):
            widgets[feat] = col.number_input(label, value=float(values.get(feat, 0.5) or 0.5),
                                             min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
                                             help=FEATURE_DESCRIPTIONS.get(feat, ""))
        else:
            widgets[feat] = col.text_input(label, value=str(values.get(feat, "")),
                                           help=FEATURE_DESCRIPTIONS.get(feat, ""))

    apply_button = st.form_submit_button("Appliquer valeurs saisies")

if apply_button:
    for f in SIM_FEATURES:
        values[f] = widgets.get(f)

# ---------- Action : obtenir score rafraîchi ----------
st.markdown("---")
st.markdown("### Simulation : obtenir un score avec ces valeurs")

col_btn, col_debug = st.columns([2, 1])
with col_btn:
    compute = st.button("Obtenir score rafraîchi", help="Envoie les valeurs à l'API et récupère la prédiction (probabilité + décision).")
with col_debug:
    show_raw = st.checkbox("Afficher payload envoyé (debug)", value=False)

prediction_result = None
api_exception = None
payload_features = {}

if compute:
    # Build payload: convert DAYS_BIRTH years -> negative days
    for f in SIM_FEATURES:
        v = values.get(f)
        if f == "DAYS_BIRTH":
            try:
                years = float(v)
                payload_features[f] = int(round(-abs(years) * 365))
            except Exception:
                payload_features[f] = None
        else:
            try:
                payload_features[f] = float(v)
            except Exception:
                payload_features[f] = v

    if show_raw:
        st.json(payload_features)

    with st.spinner("Appel de l'API de scoring..."):
        try:
            # Try preferred signature: get_client_prediction(client_id, features=...)
            if mode == "Client existant" and selected_client is not None:
                try:
                    prediction_result = get_client_prediction(int(selected_client), features=payload_features)
                except TypeError:
                    # fallbacks: try other calling styles
                    try:
                        prediction_result = get_client_prediction({"client_id": int(selected_client), "features": payload_features})
                    except Exception:
                        prediction_result = get_client_prediction(payload_features)
            else:
                try:
                    prediction_result = get_client_prediction(features=payload_features)
                except TypeError:
                    prediction_result = get_client_prediction(payload_features)
        except Exception as e:
            api_exception = e
            prediction_result = None

# ---------- Display Before / After ----------
st.markdown("### Résultat — Avant / Après")
left, right = st.columns(2)

with left:
    st.markdown("**Avant (client actuel)**")
    if original_prediction:
        display_prediction_block(original_prediction, "Avant — prédiction actuelle")
    else:
        st.info("Aucune prédiction initiale disponible pour ce client.")

with right:
    st.markdown("**Après (simulation)**")
    if compute:
        if api_exception:
            st.error("L'appel à l'API a échoué.")
            st.exception(api_exception)
            if prediction_result is None:
                st.info("Aucune prédiction renvoyée par l'API.")
        elif prediction_result:
            display_prediction_block(prediction_result, "Après — simulation")
        else:
            st.info("L'API n'a pas renvoyé de résultat (prediction_result est vide). Active 'Afficher payload envoyé' et vérifie l'API.")
    else:
        st.info("Cliquez sur 'Obtenir score rafraîchi' pour exécuter la simulation et afficher le résultat ici.")

# If debug and API returned unexpected object, show it
if compute and show_raw and prediction_result is not None:
    st.markdown("**Réponse brute de l'API :**")
    st.write(prediction_result)

# Footer help
st.markdown("---")
st.markdown(
    "- Si rien ne se passe après avoir cliqué sur « Obtenir score rafraîchi », activez « Afficher payload envoyé (debug) » puis vérifiez les logs côté API / utils.api_client.\n"
    "- Si l'appel lève une exception, le détail s'affichera dans la colonne 'Après'. Copiez‑le pour diagnostiquer la signature attendue par get_client_prediction(...).\n"
    "- Pour ajuster les features simulées, modifiez la liste SIM_FEATURES au début du fichier."
)
