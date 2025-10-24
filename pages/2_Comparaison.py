```python name=pages/2_Comparaison.py
"""
Page de comparaison entre clients

Cette version conserve l'ensemble des visuels de la page :
- cartes statut client
- tableaux de comparaison (infos perso, crédits)
- analyse univariée (distribution des probabilités)
- analyse bivariée : section intégrée (choix limité à paires métier significatives,
  transformations appliquées en backend et invisibles pour les conseillers)
- comparaison des risques (bar chart)
- explications et footer

Modifications :
- Option technique supprimée : transformations gérées en backend, pas d'options visibles.
- UI métier : l'utilisateur choisit une paire pré-définie compréhensible.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from utils.api_client import get_client_prediction, get_client_details, get_available_clients
from config import COLORBLIND_FRIENDLY_PALETTE, UI_CONFIG, FEATURE_DESCRIPTIONS

st.set_page_config(page_title="Comparaison de Clients - Dashboard de Scoring Crédit", page_icon="📊", layout="wide")

# --- Styles légers pour accessibilité ---
st.markdown("""
<style>
    .dataframe th { background-color: #f0f0f0 !important; color: #000000 !important; font-weight: bold !important; }
    .dataframe td { background-color: #ffffff !important; color: #000000 !important; }
    body, .stMarkdown, .stText { font-size: 1rem !important; line-height: 1.6 !important; }
</style>
""", unsafe_allow_html=True)

# --- Titre ---
st.title("Comparaison de profils clients")
st.markdown("Comparez plusieurs profils clients côte à côte et explorez des relations entre caractéristiques importantes. Les transformations nécessaires sont appliquées automatiquement en arrière-plan.")

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

def convert_days_to_years(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace({365243: np.nan})  # placeholder cleaning if present
    return (-s / 365.0)

def backend_prepare_plot(df, x_feat, y_feat):
    """
    Transformations en backend (non visibles) :
    - DAYS_* -> années positives
    - compression log1p pour montants (si présent)
    - standard scaling pour comparabilité visuelle
    Retourne df avec x_plot/y_plot et labels.
    """
    dfp = df.copy()
    dfp["x_num"] = pd.to_numeric(dfp["x_raw"], errors="coerce")
    dfp["y_num"] = pd.to_numeric(dfp["y_raw"], errors="coerce")

    # Convert DAYS fields
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
        x_pre = np.where(dfp["x_num_conv"] > 0, np.log1p(dfp["x_num_conv"]), dfp["x_num_conv"])
        y_pre = np.where(dfp["y_num_conv"] > 0, np.log1p(dfp["y_num_conv"]), dfp["y_num_conv"])
    else:
        x_pre = dfp["x_num_conv"].values
        y_pre = dfp["y_num_conv"].values

    # Standard scaling to make comparable plotting scales
    try:
        sx = StandardScaler(); sy = StandardScaler()
        dfp["x_plot"] = sx.fit_transform(np.array(x_pre).reshape(-1,1)).flatten()
        dfp["y_plot"] = sy.fit_transform(np.array(y_pre).reshape(-1,1)).flatten()
    except Exception:
        dfp["x_plot"] = dfp["x_num_conv"]
        dfp["y_plot"] = dfp["y_num_conv"]

    x_label = FEATURE_DESCRIPTIONS.get(x_feat, x_feat)
    y_label = FEATURE_DESCRIPTIONS.get(y_feat, y_feat)

    return dfp, x_label, y_label

# ---------- Chargement liste clients ----------
with st.spinner("Chargement de la liste des clients..."):
    try:
        available_clients = normalize_id_list(get_available_clients(limit=UI_CONFIG.get("default_limit", 100)) or [])
    except Exception:
        available_clients = []

if not available_clients:
    st.error("Impossible de charger la liste des clients.")
    st.stop()

# ---------- Sélection clients ----------
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

reference_client = st.selectbox("Choisir un client de référence (mise en évidence)", options=selected_clients, format_func=lambda x: f"Client #{x}")
try:
    reference_client = int(reference_client)
except Exception:
    pass

# ---------- Charger données clients sélectionnés ----------
with st.spinner("Chargement des données des clients sélectionnés..."):
    client_data = {}
    for cid in selected_clients:
        try:
            pred = get_client_prediction(cid)
            det = get_client_details(cid)
            if pred and det:
                client_data[int(cid)] = {"prediction": pred, "details": det}
        except Exception:
            # silencieux ici ; affichage d'erreur en cas d'absence totale plus bas
            pass

if not client_data:
    st.error("Aucune donnée client n'a pu être chargée.")
    st.stop()

# ---------- Cartes statut ----------
st.subheader("Statut des demandes de crédit")
cols = st.columns(len(client_data))
for i, (cid, data) in enumerate(client_data.items()):
    prediction = data["prediction"]
    probability = prediction.get("probability", 0)
    decision = prediction.get("decision", "INCONNU")
    with cols[i]:
        status_color = COLORBLIND_FRIENDLY_PALETTE.get('accepted', '#2ca02c') if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused', '#d62728')
        status_icon = "✅" if decision == "ACCEPTÉ" else "❌"
        st.markdown(
            f"""
            <div style="padding: 0.6rem; border-radius: 8px; background: {status_color}20; border: 1.5px solid {status_color};">
                <strong>Client #{cid}</strong><br>
                <span style="color:{status_color}">{status_icon} <strong>{decision}</strong></span><br>
                Probabilité: <strong>{probability:.1%}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------- Tableaux de comparaison ----------
st.subheader("Comparaison des informations personnelles")
comparison_data = []
for client_id, data in client_data.items():
    details = data["details"]
    pi = details.get("personal_info", {})
    comparison_data.append({
        "ID Client": client_id,
        "Âge": pi.get("age", "N/A"),
        "Genre": pi.get("gender", "N/A"),
        "Éducation": pi.get("education", "N/A"),
        "Statut familial": pi.get("family_status", "N/A"),
        "Revenu annuel": pi.get("income", "N/A"),
        "Ancienneté d'emploi": pi.get("employment_years", "N/A"),
    })
comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)

st.subheader("Comparaison des crédits demandés")
credit_data = []
for client_id, data in client_data.items():
    details = data["details"]
    credit_info = details.get("credit_info", {})
    income = details.get("personal_info", {}).get("income", 0) or 0
    annuity = credit_info.get("annuity", 0) or 0
    payment_ratio = annuity * 12 / max(income, 1) if income > 0 else np.nan
    credit_data.append({
        "ID Client": client_id,
        "Montant demandé": credit_info.get("amount", "N/A"),
        "Durée (mois)": credit_info.get("credit_term", "N/A"),
        "Mensualité": annuity,
        "Valeur du bien": credit_info.get("goods_price", "N/A"),
        "Ratio mensualité/revenu": payment_ratio
    })
credit_df = pd.DataFrame(credit_data)
st.dataframe(credit_df, use_container_width=True)

# ---------- Analyse univariée ----------
st.subheader("Analyse univariée : distribution des probabilités de défaut")
all_client_ids = normalize_id_list(get_available_clients(limit=UI_CONFIG.get("default_limit", 100)) or [])
probs = []
ids_with_prob = []
with st.spinner("Récupération des probabilités pour la distribution..."):
    for cid in all_client_ids:
        try:
            pred = get_client_prediction(cid)
            if pred:
                probs.append(pred.get("probability", 0))
                ids_with_prob.append((cid, pred.get("probability", 0)))
        except Exception:
            continue

if not probs:
    st.info("Aucune probabilité disponible pour l'instant.")
else:
    dist_df = pd.DataFrame({"client_id": [c for c, p in ids_with_prob], "probability": [p for c, p in ids_with_prob]})
    fig_hist = px.histogram(dist_df, x="probability", nbins=30, title="Distribution des probabilités de défaut (ensemble des clients)",
                            color_discrete_sequence=[COLORBLIND_FRIENDLY_PALETTE.get("primary", "#636EFA")])
    fig_hist.update_layout(xaxis=dict(tickformat=".0%", title="Probabilité de défaut"), yaxis_title="Nombre de clients", height=320)
    try:
        ref_pred = client_data[int(reference_client)]["prediction"]
        ref_prob = ref_pred.get("probability", 0)
        fig_hist.add_vline(x=ref_prob, line_dash="dash", line_color="black", annotation_text=f"Réf #{reference_client}: {ref_prob:.1%}", annotation_position="top right")
    except Exception:
        ref_prob = None
    st.plotly_chart(fig_hist, use_container_width=True)

    if ref_prob is not None:
        percentile = (np.sum(np.array(probs) <= ref_prob) / len(probs)) * 100
        st.markdown(f"**Interprétation rapide :** Le client #{reference_client} est au {percentile:.1f}e percentile (0% = le plus sûr).")
        if percentile <= 25:
            st.success("Ce client est dans les 25% les plus sûrs.")
        elif percentile <= 50:
            st.info("Ce client est dans le 25-50% (risque relativement faible).")
        elif percentile <= 75:
            st.warning("Ce client est dans le 50-75% (risque modéré).")
        else:
            st.error("Ce client est dans les 25% les moins sûrs.")

# ---------- Analyse bivariée (section intégrée) ----------
st.subheader("Analyse bivariée : comparer deux caractéristiques pertinentes")

# Paires métiers pré-définies (choix restreint et compréhensible)
PAIRS = [
    {"key": "price_vs_credit", "x": "AMT_GOODS_PRICE", "y": "AMT_CREDIT", "label": "Prix du bien vs Montant du crédit", "type": "money_vs_money"},
    {"key": "ext3_vs_credit", "x": "EXT_SOURCE_3", "y": "AMT_CREDIT", "label": "Score externe (EXT_SOURCE_3) vs Montant du crédit", "type": "score_vs_money"},
    {"key": "ext3_vs_annuity", "x": "EXT_SOURCE_3", "y": "AMT_ANNUITY", "label": "Score externe (EXT_SOURCE_3) vs Mensualité (annuité)", "type": "score_vs_money"},
    {"key": "age_vs_ext2", "x": "DAYS_BIRTH", "y": "EXT_SOURCE_2", "label": "Âge vs Score externe (EXT_SOURCE_2)", "type": "age_vs_score"},
    {"key": "education_vs_ext3", "x": "NAME_EDUCATION_TYPE", "y": "EXT_SOURCE_3", "label": "Niveau d'éducation vs Score externe (EXT_SOURCE_3)", "type": "cat_vs_score"}
]
pair_map = {p["key"]: p for p in PAIRS}
pair_labels = {p["key"]: p["label"] for p in PAIRS}

choice_key = st.selectbox("Choisir une paire métier à explorer", options=list(pair_labels.keys()), format_func=lambda k: pair_labels[k])
pair = pair_map[choice_key]
x_feature = pair["x"]; y_feature = pair["y"]; pair_type = pair["type"]
st.markdown(f"Comparaison sélectionnée : **{pair_labels[choice_key]}** (transformations automatiques appliquées).")

# construire échantillon pour graphe (jusqu'à 100 clients)
limit_for_bi = 100
with st.spinner("Chargement d'un échantillon de clients pour la bivariée..."):
    sample_ids = normalize_id_list(get_available_clients(limit=limit_for_bi) or [])[:limit_for_bi]
    if not sample_ids:
        sample_ids = available_clients[:limit_for_bi]
    rows = []
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
            rows.append({"client_id": int(cid), "x_raw": x_raw, "y_raw": y_raw, "probability": prob, "decision": decision})
        except Exception:
            continue

if not rows:
    st.info("Pas assez de données pour cette paire — essayez une autre paire.")
else:
    df = pd.DataFrame(rows)

    # si catégorie vs score => boxplot
    if pair_type == "cat_vs_score":
        df["category"] = df["x_raw"].astype(str)
        df["value"] = pd.to_numeric(df["y_raw"], errors="coerce")
        df = df.dropna(subset=["value"])
        if df.empty:
            st.info("Pas assez de données pour le boxplot.")
        else:
            fig_box = px.box(df, x="category", y="value", points="all",
                             labels={"category": FEATURE_DESCRIPTIONS.get(x_feature, x_feature),
                                     "value": FEATURE_DESCRIPTIONS.get(y_feature, y_feature)},
                             title=pair_labels[choice_key],
                             color_discrete_sequence=[COLORBLIND_FRIENDLY_PALETTE.get("primary", "#636EFA")])
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        df_prep, x_label, y_label = backend_prepare_plot(df, x_feature, y_feature)

        # marquer sélection/référence
        sel_set = set(selected_clients)
        df_prep["is_selected"] = df_prep["client_id"].isin(sel_set)
        df_prep["is_reference"] = df_prep["client_id"] == int(reference_client)

        df_other = df_prep[~df_prep["is_selected"] & ~df_prep["is_reference"]]
        df_sel = df_prep[df_prep["is_selected"] & ~df_prep["is_reference"]]
        df_ref = df_prep[df_prep["is_reference"]]

        fig = go.Figure()
        if not df_other.empty:
            fig.add_trace(go.Scatter(
                x=df_other["x_plot"], y=df_other["y_plot"], mode="markers",
                marker=dict(size=8, symbol="circle",
                            color=[COLORBLIND_FRIENDLY_PALETTE.get('accepted','#2ca02c') if d == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused','#d62728') for d in df_other["decision"]],
                            opacity=0.8),
                customdata=np.stack([df_other["client_id"], df_other["x_raw"], df_other["y_raw"], df_other["probability"]], axis=-1),
                hovertemplate=("Client #%{customdata[0]}<br>" + f"{x_label}: " + "%{customdata[1]}<br>" + f"{y_label}: " + "%{customdata[2]}<br>Probabilité: %{customdata[3]:.1%}<extra></extra>"),
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
                hovertemplate=("Client #%{customdata[0]}<br>" + f"{x_label}: " + "%{customdata[1]}<br>" + f"{y_label}: " + "%{customdata[2]}<br>Probabilité: %{customdata[3]:.1%}<extra></extra>"),
                name="Clients sélectionnés"
            ))
        if not df_ref.empty:
            r = df_ref.iloc[0]
            fig.add_trace(go.Scatter(
                x=[r["x_plot"]], y=[r["y_plot"]], mode="markers+text",
                marker=dict(size=20, symbol="star", color="black", line=dict(width=2, color='white')),
                text=[f"Réf #{int(r['client_id'])}"], textposition="bottom center",
                hovertemplate=(f"Client #{int(r['client_id'])}<br>{x_label}: {r['x_raw']}<br>{y_label}: {r['y_raw']}<br>Probabilité: {r['probability']:.1%}<extra></extra>"),
                name="Client référence"
            ))

        fig.update_layout(title=pair_labels[choice_key], xaxis_title=x_label, yaxis_title=y_label, template="simple_white", height=600)

        # Option simple : montrer une droite de tendance purement visuelle
        if st.checkbox("Afficher droite de tendance (aide visuelle)", value=False):
            try:
                lr = LinearRegression().fit(df_prep[["x_plot"]], df_prep[["y_plot"]])
                x_line = np.linspace(df_prep["x_plot"].min(), df_prep["x_plot"].max(), 200)
                y_line = lr.predict(x_line.reshape(-1,1)).flatten()
                fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="black", dash="dash"), name="Tendance"))
            except Exception:
                st.info("Impossible de tracer la droite de tendance sur ces données.")

        st.plotly_chart(fig, use_container_width=True)

# ---------- Explications simples ----------
if st.button("Explication des caractéristiques sélectionnées", key="exp_feat"):
    for f in [locals().get("x_feature"), locals().get("y_feature")]:
        if f:
            desc = FEATURE_DESCRIPTIONS.get(f, None)
            if desc:
                st.markdown(f"**{f}** — {desc}")
            else:
                st.markdown(f"**{f}** — Pas de description disponible.")

# ---------- Comparaison des risques ----------
st.subheader("Comparaison des risques de défaut")
threshold = client_data[list(client_data.keys())[0]]["prediction"].get("threshold", 0.5)
sorted_clients = sorted([(cid, dd["prediction"].get("probability", 0)) for cid, dd in client_data.items()], key=lambda x: x[1])

fig = go.Figure()
risk_zones = [
    {"name": "RISQUE TRÈS FAIBLE", "min": 0, "max": 0.2, "color": "rgba(1,133,113,0.4)"},
    {"name": "RISQUE FAIBLE", "min": 0.2, "max": 0.4, "color": "rgba(1,133,113,0.6)"},
    {"name": "RISQUE MODÉRÉ", "min": 0.4, "max": threshold, "color": "rgba(1,133,113,0.8)"},
    {"name": "RISQUE ÉLEVÉ", "min": threshold, "max": 0.7, "color": "rgba(166,97,26,0.6)"},
    {"name": "RISQUE TRÈS ÉLEVÉ", "min": 0.7, "max": 1, "color": "rgba(166,97,26,0.8)"}
]
for zone in risk_zones:
    fig.add_shape(type="rect", x0=zone["min"], x1=zone["max"], y0=-1, y1=len(sorted_clients), fillcolor=zone["color"], line=dict(width=0), layer="below")
fig.add_shape(type="line", x0=threshold, x1=threshold, y0=-2, y1=len(sorted_clients), line=dict(color="black", width=2, dash="dash"))
fig.add_annotation(x=threshold, y=-2.5, text=f"SEUIL: {threshold:.2f}", showarrow=False)

for i, (cid, prob) in enumerate(sorted_clients):
    decision = "ACCEPTÉ" if prob < threshold else "REFUSÉ"
    color = COLORBLIND_FRIENDLY_PALETTE.get('accepted','#2ca02c') if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused','#d62728')
    fig.add_trace(go.Bar(y=[i], x=[prob], orientation='h', marker=dict(color=color, line=dict(color='rgba(0,0,0,0.5)', width=1)), hovertemplate=f"Client #{cid}<br>Probabilité: {prob:.1%}<br>Décision: {decision}<extra></extra>", showlegend=False))
    fig.add_annotation(x=-0.05, y=i, text=f"#{cid}", showarrow=False, xanchor="right")
    pos_x = prob + 0.03 if abs(prob - threshold) > 0.05 else prob + 0.06
    fig.add_annotation(x=pos_x, y=i, text=f"{prob:.1%}", showarrow=False, xanchor="left")

fig.update_layout(title="Comparaison des risques de défaut par client", height=max(300, 150 + 40 * len(sorted_clients)), xaxis=dict(title="Probabilité de défaut", range=[-0.1, 1.05], tickformat=".0%"), yaxis=dict(showticklabels=False))
st.plotly_chart(fig, use_container_width=True)

# ---------- Footer ----------
st.markdown("""
<hr>
<div style="text-align:center; padding:8px; border-radius:6px; background:#f8f9fa;">
    <strong>Comparaison de clients</strong> — Transformations appliquées automatiquement pour faciliter l'interprétation ; tooltips conservent les valeurs brutes.
</div>
""", unsafe_allow_html=True)
```
