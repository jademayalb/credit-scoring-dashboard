"""
Page de comparaison entre clients — version simplifiée pour conseillers métier.

Principes appliqués ici :
- La transformation des variables (jours→années, compression des montants, scaling)
  est entièrement gérée en backend et invisible pour l'utilisateur.
- L'utilisateur choisit uniquement une paire "métier" pré‑sélectionnée parmi les plus
  parlantes (ex : montant du bien vs montant du crédit, score externe vs montant, âge vs score).
- Pas d'options techniques ni de statistiques affichées (Pearson/Spearman/R² supprimés).
- Tooltips conservent toujours les valeurs brutes pour transparence métier.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from utils.api_client import get_client_prediction, get_client_details, get_available_clients
from config import COLORBLIND_FRIENDLY_PALETTE, UI_CONFIG, FEATURE_DESCRIPTIONS

st.set_page_config(page_title="Comparaison de Clients", page_icon="📊", layout="wide")

st.title("Comparaison de profils clients (vue simplifiée)")
st.markdown(
    "Choisissez une paire de caractéristiques pertinentes pour comparer des clients. "
    "Les transformations nécessaires sont appliquées automatiquement en arrière-plan."
)

# Helpers
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

def convert_days_to_years(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace({365243: np.nan})  # placeholder cleaning if present
    return (-s / 365.0)

def backend_prepare_plot(df, x_feat, y_feat):
    """
    Effectue, en backend et sans exposition à l'utilisateur :
    - conversions DAYS_* -> années
    - compaction log1p pour montants (AMT_*)
    - scaling (StandardScaler) sur l'espace de tracé pour rendre les paires comparables
    Retourne df avec colonnes x_plot, y_plot et labels d'axes.
    """
    dfp = df.copy()
    dfp["x_num"] = pd.to_numeric(dfp["x_raw"], errors="coerce")
    dfp["y_num"] = pd.to_numeric(dfp["y_raw"], errors="coerce")

    # Convert DAYS fields to positive years if needed
    if x_feat in ("DAYS_BIRTH", "DAYS_EMPLOYED"):
        dfp["x_num_conv"] = convert_days_to_years(dfp["x_num"])
    else:
        dfp["x_num_conv"] = dfp["x_num"]

    if y_feat in ("DAYS_BIRTH", "DAYS_EMPLOYED"):
        dfp["y_num_conv"] = convert_days_to_years(dfp["y_num"])
    else:
        dfp["y_num_conv"] = dfp["y_num"]

    # Money compression heuristic
    money_feats = {"AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY"}
    if (x_feat in money_feats) or (y_feat in money_feats):
        # apply log1p for positive values then standardize
        x_pre = np.where(dfp["x_num_conv"] > 0, np.log1p(dfp["x_num_conv"]), dfp["x_num_conv"])
        y_pre = np.where(dfp["y_num_conv"] > 0, np.log1p(dfp["y_num_conv"]), dfp["y_num_conv"])
    else:
        x_pre = dfp["x_num_conv"].values
        y_pre = dfp["y_num_conv"].values

    # Standard scaling to place variables on comparable plotting scale
    try:
        sx = StandardScaler()
        sy = StandardScaler()
        dfp["x_plot"] = sx.fit_transform(np.array(x_pre).reshape(-1, 1)).flatten()
        dfp["y_plot"] = sy.fit_transform(np.array(y_pre).reshape(-1, 1)).flatten()
    except Exception:
        # fallback to raw converted values if scaling fails
        dfp["x_plot"] = dfp["x_num_conv"]
        dfp["y_plot"] = dfp["y_num_conv"]

    x_label = FEATURE_DESCRIPTIONS.get(x_feat, x_feat)
    y_label = FEATURE_DESCRIPTIONS.get(y_feat, y_feat)

    return dfp, x_label, y_label

# Load available clients
with st.spinner("Chargement de la liste des clients..."):
    try:
        available_clients = normalize_id_list(get_available_clients(limit=UI_CONFIG.get("default_limit", 100)) or [])
    except Exception:
        available_clients = []

if not available_clients:
    st.error("Impossible de charger la liste des clients.")
    st.stop()

# Simple selection of clients to compare and a reference client
selected_clients = st.multiselect(
    "Sélectionnez 2 à 4 clients à comparer",
    options=available_clients,
    default=available_clients[:2] if len(available_clients) >= 2 else available_clients,
    max_selections=4
)
selected_clients = normalize_id_list(selected_clients)
if len(selected_clients) < 2:
    st.warning("Veuillez sélectionner au moins deux clients pour la comparaison.")
    st.stop()

reference_client = st.selectbox("Choisir un client de référence", options=selected_clients, format_func=lambda x: f"Client #{x}")
reference_client = int(reference_client)

# Load selected clients data (for the top cards and small tables)
client_data = {}
with st.spinner("Chargement des données des clients sélectionnés..."):
    for cid in selected_clients:
        try:
            pred = get_client_prediction(cid)
            det = get_client_details(cid)
            if pred and det:
                client_data[int(cid)] = {"prediction": pred, "details": det}
        except Exception:
            pass

if not client_data:
    st.error("Aucune donnée pour les clients sélectionnés.")
    st.stop()

# Status cards
st.subheader("Statut des demandes")
cols = st.columns(len(client_data))
for i, (cid, data) in enumerate(client_data.items()):
    pred = data["prediction"]
    prob = pred.get("probability", 0)
    decision = pred.get("decision", "INCONNU")
    color = COLORBLIND_FRIENDLY_PALETTE.get("accepted", "#2ca02c") if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get("refused", "#d62728")
    with cols[i]:
        st.markdown(
            f"""
            <div style="padding:0.6rem;border-radius:8px;border:1px solid {color};background:{color}20">
              <strong>Client #{cid}</strong><br>
              <span style="color:{color}">{decision}</span><br>
              Probabilité: <strong>{prob:.1%}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

# Predefined, business-meaningful pairs (chosen among the 9 SHAP features)
PAIRS = [
    {"key": "price_vs_credit", "x": "AMT_GOODS_PRICE", "y": "AMT_CREDIT",
     "label": "Prix du bien vs Montant du crédit", "type": "money_vs_money"},
    {"key": "ext3_vs_credit", "x": "EXT_SOURCE_3", "y": "AMT_CREDIT",
     "label": "Score externe (EXT_SOURCE_3) vs Montant du crédit", "type": "score_vs_money"},
    {"key": "ext3_vs_annuity", "x": "EXT_SOURCE_3", "y": "AMT_ANNUITY",
     "label": "Score externe (EXT_SOURCE_3) vs Mensualité (annuité)", "type": "score_vs_money"},
    {"key": "age_vs_ext2", "x": "DAYS_BIRTH", "y": "EXT_SOURCE_2",
     "label": "Âge vs Score externe (EXT_SOURCE_2)", "type": "age_vs_score"},
    {"key": "education_vs_ext3", "x": "NAME_EDUCATION_TYPE", "y": "EXT_SOURCE_3",
     "label": "Niveau d'éducation vs Score externe (EXT_SOURCE_3)", "type": "cat_vs_score"}
]

pair_labels = {p["key"]: p["label"] for p in PAIRS}
pair_map = {p["key"]: p for p in PAIRS}

# Single select for meaningful pair
choice_key = st.selectbox("Choisir une paire d'analyse (sélection métier simple)", options=list(pair_labels.keys()), format_func=lambda k: pair_labels[k])

pair = pair_map[choice_key]
x_feature = pair["x"]
y_feature = pair["y"]
pair_type = pair["type"]

st.markdown(f"Vous comparez : **{pair_labels[choice_key]}**. (Les transformations sont gérées automatiquement.)")

# Build sample of up to 100 clients to plot
limit_bi = 100
with st.spinner("Chargement d'un échantillon de clients pour le graphique..."):
    sample_ids = normalize_id_list(get_available_clients(limit=limit_bi) or [])[:limit_bi]
    if not sample_ids:
        sample_ids = available_clients[:limit_bi]
    sample_rows = []
    for cid in sample_ids:
        try:
            d = get_client_details(cid)
            p = get_client_prediction(cid)
            if not d or "features" not in d or not p:
                continue
            feats = d["features"]
            x_raw = feats.get(x_feature, None)
            y_raw = feats.get(y_feature, None)
            if x_raw is None or y_raw is None:
                continue
            prob = p.get("probability", 0)
            thr = p.get("threshold", 0.5)
            decision = "ACCEPTÉ" if prob < thr else "REFUSÉ"
            sample_rows.append({"client_id": int(cid), "x_raw": x_raw, "y_raw": y_raw, "probability": prob, "decision": decision})
        except Exception:
            continue

if not sample_rows:
    st.info("Pas assez de données pour cette paire. Essayez une autre paire.")
else:
    df = pd.DataFrame(sample_rows)

    # If categorical vs numeric pair (e.g., education vs score), show a boxplot
    if pair_type == "cat_vs_score":
        # ensure category is string and numeric is numeric
        df["category"] = df["x_raw"].astype(str)
        df["value"] = pd.to_numeric(df["y_raw"], errors="coerce")
        df = df.dropna(subset=["value"])
        if df.empty:
            st.info("Pas assez de données numériques pour le boxplot.")
        else:
            fig_box = px.box(df, x="category", y="value", points="all",
                             labels={"category": FEATURE_DESCRIPTIONS.get(x_feature, x_feature),
                                     "value": FEATURE_DESCRIPTIONS.get(y_feature, y_feature)},
                             title=pair_labels[choice_key],
                             color_discrete_sequence=[COLORBLIND_FRIENDLY_PALETTE.get("primary", "#636EFA")])
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        # prepare plotting values using backend rules
        df_prepared, x_label, y_label = backend_prepare_plot(df, x_feature, y_feature)

        # identify selected & reference
        sel_set = set(selected_clients)
        df_prepared["is_selected"] = df_prepared["client_id"].isin(sel_set)
        df_prepared["is_reference"] = df_prepared["client_id"] == reference_client

        df_other = df_prepared[~df_prepared["is_selected"] & ~df_prepared["is_reference"]]
        df_sel = df_prepared[df_prepared["is_selected"] & ~df_prepared["is_reference"]]
        df_ref = df_prepared[df_prepared["is_reference"]]

        fig = go.Figure()

        if not df_other.empty:
            fig.add_trace(go.Scatter(
                x=df_other["x_plot"], y=df_other["y_plot"], mode="markers",
                marker=dict(size=8, symbol="circle",
                            color=[COLORBLIND_FRIENDLY_PALETTE.get('accepted','#2ca02c') if d == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused','#d62728') for d in df_other["decision"]],
                            opacity=0.8),
                customdata=np.stack([df_other["client_id"], df_other["x_raw"], df_other["y_raw"], df_other["probability"]], axis=-1),
                hovertemplate=("Client #%{customdata[0]}<br>"
                               f"{x_label}: " + "%{customdata[1]}<br>"
                               f"{y_label}: " + "%{customdata[2]}<br>"
                               "Probabilité: %{customdata[3]:.1%}<extra></extra>"),
                name="Autres clients"
            ))

        if not df_sel.empty:
            fig.add_trace(go.Scatter(
                x=df_sel["x_plot"], y=df_sel["y_plot"], mode="markers+text",
                marker=dict(size=13, symbol="diamond", line=dict(width=1, color="black"),
                            color=[COLORBLIND_FRIENDLY_PALETTE.get('accepted','#2ca02c') if d == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused','#d62728') for d in df_sel["decision"]]),
                text=[f"#{int(c)}" for c in df_sel["client_id"]],
                textposition="top center",
                customdata=np.stack([df_sel["client_id"], df_sel["x_raw"], df_sel["y_raw"], df_sel["probability"]], axis=-1),
                hovertemplate=("Client #%{customdata[0]}<br>"
                               f"{x_label}: " + "%{customdata[1]}<br>"
                               f"{y_label}: " + "%{customdata[2]}<br>"
                               "Probabilité: %{customdata[3]:.1%}<extra></extra>"),
                name="Clients sélectionnés"
            ))

        if not df_ref.empty:
            r = df_ref.iloc[0]
            fig.add_trace(go.Scatter(
                x=[r["x_plot"]], y=[r["y_plot"]], mode="markers+text",
                marker=dict(size=20, symbol="star", color="black", line=dict(width=2, color="white")),
                text=[f"Réf #{int(r['client_id'])}"], textposition="bottom center",
                hovertemplate=(f"Client #{int(r['client_id'])}<br>{x_label}: {r['x_raw']}<br>{y_label}: {r['y_raw']}<br>Probabilité: {r['probability']:.1%}<extra></extra>"),
                name="Client référence"
            ))

        fig.update_layout(title=pair_labels[choice_key], xaxis_title=x_label, yaxis_title=y_label, template="simple_white", height=600)

        # Simple visual option: trendline (visual only, no technical stats shown)
        if st.checkbox("Afficher droite de tendance (optionnel, juste pour aide visuelle)", value=False):
            try:
                lr = LinearRegression().fit(df_prepared[["x_plot"]], df_prepared["y_plot"])
                x_line = np.linspace(df_prepared["x_plot"].min(), df_prepared["x_plot"].max(), 200)
                y_line = lr.predict(x_line.reshape(-1, 1)).flatten()
                fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="black", dash="dash"), name="Tendance"))
            except Exception:
                st.info("Impossible de tracer la droite de tendance sur ces données.")

        st.plotly_chart(fig, use_container_width=True)

# Footer: rappeler que les transformations sont appliquées en backend
st.markdown(
    "<small>Remarque : les conversions/compactions nécessaires (jours→années, compression des montants, mise à l'échelle pour affichage) "
    "sont faites automatiquement pour faciliter l'interprétation métier. Les tooltips montrent toujours les valeurs brutes.</small>",
    unsafe_allow_html=True
)
