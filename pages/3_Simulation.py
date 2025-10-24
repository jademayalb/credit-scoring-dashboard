"""
Page de simulation : modifier/entrer un profil client et obtenir un score rafraîchi.

Fonctionnalités :
- Choisir un client existant ou "Nouveau client".
- Afficher et éditer un ensemble restreint de features (5 retenues).
- Bouton "Obtenir score rafraîchi" : envoie les valeurs à l'API via utils.api_client.get_client_prediction(...)
  - si édition d'un client existant : on tente d'appeler l'API en fournissant client_id + features (si supporté) ;
  - sinon on envoie les features seules.
- Affichage du résultat : jauge / métrique + décision textuelle + détails (probabilité, seuil).
- Accessibilité : labels explicites, aide textuelle (descriptions), contrastes simples.
- Défensive : messages d'erreur clairs si l'API ne répond pas.

Remarque : adapte l'appel get_client_prediction(...) si l'API attend un schéma particulier.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

from utils.api_client import get_client_prediction, get_client_details, get_available_clients
from config import FEATURE_DESCRIPTIONS, COLORBLIND_FRIENDLY_PALETTE, UI_CONFIG

st.set_page_config(page_title="Simulation - Scoring Crédit", page_icon="🧪", layout="wide")

st.title("Simulation de scoring — Client existant ou nouveau profil")
st.markdown(
    "Choisissez un client existant ou saisissez un nouveau profil. "
    "Modifiez les valeurs des caractéristiques sélectionnées, puis cliquez sur "
    "« Obtenir score rafraîchi » pour appeler l'API et afficher la probabilité et la décision."
)

# --- Features retenues pour la simulation (modifiable si besoin) ---
SIM_FEATURES = [
    "AMT_CREDIT",       # montant du crédit demandé
    "AMT_GOODS_PRICE",  # valeur du bien
    "AMT_ANNUITY",      # mensualité
    "DAYS_BIRTH",       # âge en jours négatifs dans les données sources ; on présentera en années
    "EXT_SOURCE_3"      # score externe (float 0-1)
]

# Ensure basic descriptions exist
FEATURE_DESCRIPTIONS.setdefault("AMT_CREDIT", "Montant du crédit demandé.")
FEATURE_DESCRIPTIONS.setdefault("AMT_GOODS_PRICE", "Valeur du bien/service financé.")
FEATURE_DESCRIPTIONS.setdefault("AMT_ANNUITY", "Montant de l'annuité (mensualité).")
FEATURE_DESCRIPTIONS.setdefault("DAYS_BIRTH", "Âge exprimé en jours (négatif dans les données sources). L'interface affiche des années positives.")
FEATURE_DESCRIPTIONS.setdefault("EXT_SOURCE_3", "Score externe (valeur continue entre 0 et 1).")

# --- Accessibility helper text ---
st.markdown(
    "<small>Champs accessibles : utilisez Tab pour naviguer. Les libellés contiennent des descriptions pour les lecteurs d'écran.</small>",
    unsafe_allow_html=True
)

# ---------- Chargement liste clients ----------
with st.spinner("Chargement de la liste des clients..."):
    try:
        avail = get_available_clients(limit=UI_CONFIG.get("default_limit", 200))
        available_clients = [int(x) for x in (avail or [])]
    except Exception:
        available_clients = []

col1, col2 = st.columns([2, 1])

with col1:
    mode = st.radio(
        "Mode",
        options=["Client existant", "Nouveau client"],
        index=0,
        help="Sélectionnez 'Client existant' pour préremplir les champs à partir d'un client ; "
             "'Nouveau client' permet de saisir manuellement un profil."
    )

with col2:
    if available_clients:
        selected_client = st.selectbox(
            "Choisir un client existant",
            options=available_clients,
            format_func=lambda x: f"Client #{x}",
            disabled=(mode != "Client existant")
        )
    else:
        selected_client = None
        st.info("Aucune liste de clients disponible (API get_available_clients).")

# ---------- Prepare initial values ----------
# Defaults if new
defaults: Dict[str, Any] = {
    "AMT_CREDIT": 500000.0,
    "AMT_GOODS_PRICE": 600000.0,
    "AMT_ANNUITY": 15000.0,
    "DAYS_BIRTH": 35.0,   # we'll present years positive; convert later to negative days
    "EXT_SOURCE_3": 0.5
}

# If client selected and mode is existing, try to fetch details to prefill
prefill = {}
if mode == "Client existant" and selected_client is not None:
    with st.spinner(f"Chargement des données du client #{selected_client}..."):
        try:
            details = get_client_details(int(selected_client))
            feats = (details or {}).get("features", {})
            # map features into prefill if present; convert DAYS_BIRTH to positive years for input
            for f in SIM_FEATURES:
                raw = feats.get(f, None)
                if raw is None:
                    continue
                if f == "DAYS_BIRTH":
                    try:
                        # raw might be negative days; convert to years positive
                        val = float(raw)
                        prefill[f] = round((-val) / 365.0, 2) if val != 365243 else defaults[f]
                    except Exception:
                        prefill[f] = defaults[f]
                else:
                    prefill[f] = raw
        except Exception:
            st.warning("Impossible de récupérer les détails du client. Saisir manuellement les valeurs.")
            prefill = {}

# Merge defaults and prefill
values = {f: prefill.get(f, defaults.get(f)) for f in SIM_FEATURES}

# ---------- Form to edit features ----------
st.markdown("### Valeurs des caractéristiques (édition)")
form = st.form(key="sim_form", clear_on_submit=False)

with form:
    input_cols = st.columns(2)
    widgets = {}
    for i, feat in enumerate(SIM_FEATURES):
        col = input_cols[i % 2]
        label = f"{feat} — {FEATURE_DESCRIPTIONS.get(feat, '')}"
        # numeric features
        if feat in {"AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY", "AMT_INCOME_TOTAL"}:
            widgets[feat] = col.number_input(
                label,
                value=float(values.get(feat, 0.0) or 0.0),
                min_value=0.0,
                format="%.2f",
                help=FEATURE_DESCRIPTIONS.get(feat, ""),
                step=100.0
            )
        elif feat == "DAYS_BIRTH":
            # present to user as years positive
            widgets[feat] = col.number_input(
                label,
                value=float(values.get(feat, 35.0) or 35.0),
                min_value=16.0,
                max_value=120.0,
                format="%.1f",
                help="Saisir l'âge en années (valeur positive, ex: 35)."
            )
        elif feat.startswith("EXT_SOURCE"):
            widgets[feat] = col.number_input(
                label,
                value=float(values.get(feat, 0.5) or 0.5),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.3f",
                help=FEATURE_DESCRIPTIONS.get(feat, "")
            )
        else:
            # generic numeric fallback
            widgets[feat] = col.text_input(label, value=str(values.get(feat, "")), help=FEATURE_DESCRIPTIONS.get(feat, ""))

    submitted = st.form_submit_button("Appliquer les valeurs saisies")

# update values from form if submitted
if submitted:
    for f in SIM_FEATURES:
        values[f] = widgets.get(f)

# ---------- Action : obtenir score rafraîchi ----------
st.markdown("---")
st.markdown("### Simulation : obtenir un score avec ces valeurs")

col_a, col_b = st.columns([2, 1])
with col_a:
    explain = "Cliquez pour envoyer les valeurs à l'API et obtenir une prédiction (probabilité + décision)."
    compute = st.button("Obtenir score rafraîchi", help=explain)
with col_b:
    show_raw = st.checkbox("Afficher payload envoyé (debug)", value=False, help="Montre les valeurs envoyées à l'API.")

prediction_result = None
if compute:
    # Build payload: convert DAYS_BIRTH years -> negative days for model input
    payload_features: Dict[str, Any] = {}
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

    # Call API defensively. Try multiple calling styles depending on utils.api_client implementation.
    with st.spinner("Appel de l'API de scoring..."):
        try:
            # Preferred: if editing an existing client, try to pass client id + features (some APIs accept that)
            if mode == "Client existant" and selected_client is not None:
                try:
                    prediction_result = get_client_prediction(int(selected_client), features=payload_features)
                except TypeError:
                    # fallback variant: pass a dict directly
                    try:
                        prediction_result = get_client_prediction({"client_id": int(selected_client), "features": payload_features})
                    except Exception:
                        # final fallback: send features only
                        prediction_result = get_client_prediction(payload_features)
            else:
                # New client: send features only
                try:
                    prediction_result = get_client_prediction(features=payload_features)
                except TypeError:
                    prediction_result = get_client_prediction(payload_features)
        except Exception as e:
            st.error(f"Erreur lors de l'appel API : {e}")
            prediction_result = None

# ---------- Display result (gauge + decision) ----------
if prediction_result:
    # Expect prediction_result to contain at least 'probability' and optionally 'decision' and 'threshold'
    prob = prediction_result.get("probability", None)
    decision = prediction_result.get("decision", None)
    threshold = prediction_result.get("threshold", 0.5)

    # Normalize probability
    try:
        prob_val = float(prob)
    except Exception:
        prob_val = None

    if prob_val is None:
        st.warning("L'API a renvoyé un résultat non interprétable.")
    else:
        # Gauge using Plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_val,
            number={"valueformat": ".1%", "font": {"size": 32}},
            delta={"reference": threshold, "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
            gauge={
                "axis": {"range": [0, 1], "tickformat": ".0%"},
                "bar": {"color": COLORBLIND_FRIENDLY_PALETTE.get("refused", "#d62728")},
                "steps": [
                    {"range": [0, threshold*0.5], "color": "rgba(1,133,113,0.3)"},
                    {"range": [threshold*0.5, threshold], "color": "rgba(1,133,113,0.6)"},
                    {"range": [threshold, 1], "color": "rgba(166,97,26,0.6)"}
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": threshold}
            },
            title={"text": f"Probabilité de défaut (client simulé)", "font": {"size": 14}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Decision text (accessible)
        if decision is None:
            # derive decision from threshold if missing
            decision = "ACCEPTÉ" if prob_val < threshold else "REFUSÉ"
        # Colored banner
        color = COLORBLIND_FRIENDLY_PALETTE.get("accepted", "#2ca02c") if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get("refused", "#d62728")
        st.markdown(
            f"""
            <div role="status" aria-live="polite" style="padding:0.8rem; border-radius:8px; background:{color}20; border:1.5px solid {color};">
                <strong>Décision : {decision}</strong><br>
                Probabilité: <strong>{prob_val:.1%}</strong> (Seuil: {threshold:.2f})
            </div>
            """,
            unsafe_allow_html=True
        )

        # Show returned explanation/details if present
        if "explanation" in prediction_result:
            st.markdown("**Explication fournie par l'API :**")
            st.write(prediction_result["explanation"])

# ---------- Footer / help ----------
st.markdown("---")
st.markdown(
    "Notes :\n"
    "- Si l'appel API échoue, vérifiez la signature de get_client_prediction(...) dans utils.api_client.\n"
    "- Les âges sont saisis en années (positives) et convertis en jours négatifs avant envoi, comme dans les données sources.\n"
    "- Cette page n'envoie que les 5 features listées ; pour utiliser d'autres features, modifiez SIM_FEATURES au début du fichier."
)

