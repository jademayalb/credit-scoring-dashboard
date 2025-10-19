"""
Page de comparaison entre clients
Permet de comparer les profils de différents clients et leurs scores
Ajouts:
- Analyse univariée (distribution des probabilités) avec position d'un client de référence et quantile (ex. dans les 25% les plus sûrs)
- Analyse bivariée (scatter) entre deux features sélectionnables, coloration par décision et mise en évidence du client de référence
- Sélection d'un client de référence parmi les clients sélectionnés
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from utils.api_client import get_client_prediction, get_client_details, get_available_clients

# Import de la configuration
from config import (
    COLORBLIND_FRIENDLY_PALETTE, UI_CONFIG,
    FEATURE_DESCRIPTIONS, CSV_PATHS
)

# Configuration de la page
st.set_page_config(
    page_title="Comparaison de Clients - Dashboard de Scoring Crédit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS pour l'accessibilité
st.markdown("""
<style>
    .visually-hidden {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        padding: 0 !important;
        margin: -1px !important;
        overflow: hidden !important;
        clip: rect(0, 0, 0, 0) !important;
        white-space: nowrap !important;
        border: 0 !important;
    }
    .dataframe th { background-color: #f0f0f0 !important; color: #000000 !important; font-weight: bold !important; }
    .dataframe td { background-color: #ffffff !important; color: #000000 !important; }
    body, .stMarkdown, .stText { font-size: 1rem !important; line-height: 1.6 !important; }
    h1 { font-size: 2rem !important; } h2 { font-size: 1.75rem !important; } h3 { font-size: 1.5rem !important; }
    .score-gauge { padding: 0.5rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .score-marker { font-size: 1.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Titre et présentation
st.title("Dashboard Credit Scoring")
st.markdown('<span class="visually-hidden" aria-hidden="false">Icône représentant une carte de crédit pour le dashboard de scoring</span>', unsafe_allow_html=True)

# Barre de navigation principale
tabs = ["Accueil", "Profil Client", "Comparaison", "Simulation"]
selected_tab = st.tabs(tabs)

# Navigation
if selected_tab[0].button("Accueil", key="nav_home", use_container_width=True):
    st.switch_page("Home.py")
elif selected_tab[1].button("Profil Client", key="nav_profile", use_container_width=True):
    st.switch_page("pages/1_Profil_Client.py")
elif selected_tab[3].button("Simulation", key="nav_simulation", use_container_width=True):
    st.switch_page("pages/3_Simulation.py")

# Titre de la page
st.title("Comparaison de profils clients")
st.markdown("""
Cette page vous permet de comparer plusieurs profils clients côte à côte, 
pour analyser les différences entre leurs caractéristiques et comprendre les variations dans les décisions de crédit.
""")

# Chargement de la liste des clients disponibles (application_test.csv contient les 100 premiers IDs)
with st.spinner("Chargement de la liste des clients..."):
    available_clients = get_available_clients(limit=UI_CONFIG["default_limit"])

if not available_clients:
    st.error("Impossible de charger la liste des clients.")
    st.stop()

# Sélection des clients à comparer (multiselect)
selected_clients = st.multiselect(
    "Sélectionnez 2 à 4 clients à comparer:",
    options=available_clients,
    default=[available_clients[0], available_clients[1]] if len(available_clients) > 1 else [available_clients[0]],
    max_selections=4,
    key="client_selection"
)

if len(selected_clients) < 2:
    st.warning("Veuillez sélectionner au moins deux clients pour la comparaison.")
    st.stop()

# Option: choisir un client de référence parmi les sélectionnés (pour l'univariée et bivariée)
reference_client = st.selectbox(
    "Choisir un client de référence (pour mise en évidence):",
    options=selected_clients,
    format_func=lambda x: f"Client #{x}"
)

# Chargement des données des clients sélectionnés
with st.spinner("Chargement des données des clients sélectionnés..."):
    client_data = {}
    for client_id in selected_clients:
        prediction = get_client_prediction(client_id)
        details = get_client_details(client_id)
        if prediction and details:
            client_data[client_id] = {"prediction": prediction, "details": details}
        else:
            st.error(f"Impossible de charger les données pour le client {client_id}.")

if not client_data:
    st.error("Aucune donnée client n'a pu être chargée.")
    st.stop()

# Affichage des cartes de statut des clients
st.subheader("Statut des demandes de crédit")
cols = st.columns(len(client_data))
for i, (client_id, data) in enumerate(client_data.items()):
    prediction = data["prediction"]
    probability = prediction.get("probability", 0)
    decision = prediction.get("decision", "INCONNU")
    with cols[i]:
        status_color = COLORBLIND_FRIENDLY_PALETTE['accepted'] if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE['refused']
        status_icon = "✅" if decision == "ACCEPTÉ" else "❌"
        st.markdown(
            f"""
            <div style="padding: 0.75rem; border-radius: 0.5rem; background-color: {status_color}22; border: 2px solid {status_color}; margin-bottom: 1rem;">
                <h3 style="margin-top: 0; font-size: 1.2rem;">Client #{client_id}</h3>
                <div style="color: {status_color}; margin: 0; display: flex; align-items: center; font-size: 1.1rem;">
                    <span aria-hidden="true">{status_icon}</span> 
                    <span><strong>{decision}</strong></span>
                </div>
                <div>Probabilité: <strong>{probability:.1%}</strong></div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Comparaison des informations personnelles (tableau)
st.subheader("Comparaison des informations personnelles")
comparison_data = []
for client_id, data in client_data.items():
    details = data["details"]
    client_row = {
        "ID Client": client_id,
        "Âge": details.get("personal_info", {}).get("age", "N/A"),
        "Genre": details.get("personal_info", {}).get("gender", "N/A"),
        "Éducation": details.get("personal_info", {}).get("education", "N/A"),
        "Statut familial": details.get("personal_info", {}).get("family_status", "N/A"),
        "Revenu annuel": details.get("personal_info", {}).get("income", "N/A"),
        "Ancienneté d'emploi": details.get("personal_info", {}).get("employment_years", "N/A"),
    }
    comparison_data.append(client_row)

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(
    comparison_df,
    column_config={
        "ID Client": st.column_config.TextColumn("ID Client"),
        "Âge": st.column_config.NumberColumn("Âge", format="%d ans"),
        "Genre": st.column_config.TextColumn("Genre"),
        "Éducation": st.column_config.TextColumn("Éducation"),
        "Statut familial": st.column_config.TextColumn("Statut familial"),
        "Revenu annuel": st.column_config.NumberColumn("Revenu annuel", format=f"%d {UI_CONFIG['currency_symbol']}"),
        "Ancienneté d'emploi": st.column_config.NumberColumn("Ancienneté d'emploi", format="%d ans")
    },
    hide_index=True,
    use_container_width=True
)

# Comparaison des informations de crédit (tableau)
st.subheader("Comparaison des crédits demandés")
credit_data = []
for client_id, data in client_data.items():
    details = data["details"]
    credit_info = details.get("credit_info", {})
    income = details.get("personal_info", {}).get("income", 0)
    payment_ratio = credit_info.get("annuity", 0) * 12 / max(income, 1) if income > 0 else 0
    credit_row = {
        "ID Client": client_id,
        "Montant demandé": credit_info.get("amount", "N/A"),
        "Durée (mois)": credit_info.get("credit_term", "N/A"),
        "Mensualité": credit_info.get("annuity", "N/A"),
        "Valeur du bien": credit_info.get("goods_price", "N/A"),
        "Ratio mensualité/revenu": payment_ratio
    }
    credit_data.append(credit_row)

credit_df = pd.DataFrame(credit_data)
st.dataframe(
    credit_df,
    column_config={
        "ID Client": st.column_config.TextColumn("ID Client"),
        "Montant demandé": st.column_config.NumberColumn("Montant demandé", format=f"%d {UI_CONFIG['currency_symbol']}"),
        "Durée (mois)": st.column_config.NumberColumn("Durée (mois)", format="%d"),
        "Mensualité": st.column_config.NumberColumn("Mensualité", format=f"%d {UI_CONFIG['currency_symbol']}"),
        "Valeur du bien": st.column_config.NumberColumn("Valeur du bien", format=f"%d {UI_CONFIG['currency_symbol']}"),
        "Ratio mensualité/revenu": st.column_config.NumberColumn("Ratio mensualité/revenu", format="%.2f")
    },
    hide_index=True,
    use_container_width=True
)

# -----------------------------
# Analyse univariée (distribution des probabilités)
# -----------------------------
st.subheader("Analyse univariée : distribution des probabilités de défaut")

# Récupérer la distribution des probabilités pour l'ensemble des clients disponibles
all_client_ids = get_available_clients(limit=UI_CONFIG.get("default_limit", 100))
probs = []
ids_with_prob = []
with st.spinner("Récupération des probabilités pour la distribution..."):
    for cid in all_client_ids:
        pred = get_client_prediction(cid)
        if pred:
            probs.append(pred.get("probability", 0))
            ids_with_prob.append((cid, pred.get("probability", 0)))

if not probs:
    st.info("Aucune probabilité disponible pour l'instant.")
else:
    dist_df = pd.DataFrame({"client_id": [c for c, p in ids_with_prob], "probability": [p for c, p in ids_with_prob]})
    # Histogramme + Kernel density
    fig = px.histogram(
        dist_df, x="probability", nbins=30,
        title="Distribution des probabilités de défaut (ensemble des clients)",
        color_discrete_sequence=[COLORBLIND_FRIENDLY_PALETTE["primary"]]
    )
    fig.update_layout(xaxis=dict(tickformat=".0%", title="Probabilité de défaut"), yaxis_title="Nombre de clients", height=350)
    # Position du client de référence
    ref_pred = client_data[reference_client]["prediction"]
    ref_prob = ref_pred.get("probability", 0)
    # Ajouter une ligne verticale pour le client de référence
    fig.add_vline(x=ref_prob, line_dash="dash", line_color="black", annotation_text=f"Réf #{reference_client}: {ref_prob:.1%}", annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)

    # Calcul du quantile / percentile (plus bas = plus "safe")
    percentile = (np.sum(np.array(probs) <= ref_prob) / len(probs)) * 100
    percentile_text = f"Le client #{reference_client} se situe au {percentile:.1f}e percentile (où 0% = le plus sûr, 100% = le moins sûr)."
    st.markdown(f"**Interprétation rapide :** {percentile_text}")
    if percentile <= 25:
        st.success(f"Le client #{reference_client} se situe dans les 25% les plus sûrs.")
    elif percentile <= 50:
        st.info(f"Le client #{reference_client} se situe dans le 25-50% (risque relativement faible).")
    elif percentile <= 75:
        st.warning(f"Le client #{reference_client} se situe dans le 50-75% (risque modéré).")
    else:
        st.error(f"Le client #{reference_client} se situe dans les 25% les moins sûrs.")

# -----------------------------
# Analyse bivariée (scatter plot)
# -----------------------------
st.subheader("Analyse bivariée : comparer deux caractéristiques")

# Features disponibles (prendre depuis le premier client chargé)
available_features = list(client_data[list(client_data.keys())[0]]["details"]["features"].keys())

# Filtrer features numériques pour scatter par défaut
numeric_features = []
for f in available_features:
    try:
        # simple heuristique: tenter de convertir à float
        float(client_data[list(client_data.keys())[0]]["details"]["features"].get(f, 0))
        numeric_features.append(f)
    except Exception:
        continue

# Garder la liste visible avec noms lisibles
display_names = {f: FEATURE_DESCRIPTIONS.get(f, f) for f in available_features}

# Sélecteurs pour X et Y
col_x, col_y = st.columns(2)
with col_x:
    x_feature = st.selectbox("Axe X", options=numeric_features, format_func=lambda v: display_names.get(v, v))
with col_y:
    y_feature = st.selectbox("Axe Y", options=[f for f in numeric_features if f != x_feature], format_func=lambda v: display_names.get(v, v))

if x_feature and y_feature:
    # Construire dataframe pour scatter
    scatter_rows = []
    for cid, data in client_data.items():
        feats = data["details"]["features"]
        x_val = feats.get(x_feature, np.nan)
        y_val = feats.get(y_feature, np.nan)
        prob = data["prediction"].get("probability", 0)
        decision = "ACCEPTÉ" if prob < data["prediction"].get("threshold", 0.5) else "REFUSÉ"
        scatter_rows.append({"client_id": cid, "x": x_val, "y": y_val, "decision": decision})

    scatter_df = pd.DataFrame(scatter_rows)

    # Plot
    fig2 = px.scatter(
        scatter_df,
        x="x", y="y",
        color="decision",
        hover_data=["client_id"],
        color_discrete_map={"ACCEPTÉ": COLORBLIND_FRIENDLY_PALETTE["accepted"], "REFUSÉ": COLORBLIND_FRIENDLY_PALETTE["refused"]},
        labels={"x": display_names.get(x_feature, x_feature), "y": display_names.get(y_feature, y_feature)},
        title=f"Scatter: {display_names.get(x_feature, x_feature)} vs {display_names.get(y_feature, y_feature)}"
    )

    # Mettre en évidence le client de référence
    ref_feats = client_data[reference_client]["details"]["features"]
    ref_x = ref_feats.get(x_feature, None)
    ref_y = ref_feats.get(y_feature, None)
    if ref_x is not None and ref_y is not None:
        fig2.add_trace(go.Scatter(
            x=[ref_x],
            y=[ref_y],
            mode="markers+text",
            marker=dict(size=16, symbol='diamond', color='black', line=dict(width=2, color='white')),
            text=[f"Réf #{reference_client}"],
            textposition="top center",
            showlegend=False,
            hovertemplate=f"Client #{reference_client}<br>{display_names.get(x_feature,x_feature)}: {ref_x}<br>{display_names.get(y_feature,y_feature)}: {ref_y}<extra></extra>"
        ))

    fig2.update_layout(height=500, legend_title="Décision")
    st.plotly_chart(fig2, use_container_width=True)

    # Texte d'aide reliant l'univariée (expliquer si le client est déjà dans 25% safe)
    st.markdown("Note: la position du client dans le scatter s'interprète en complément de l'analyse univariée ci‑dessus. Si le client est dans les 25% les plus sûrs, cela renforce l'interprétation favorable.")

# -----------------------------
# Explications des caractéristiques sélectionnées
# -----------------------------
if st.button("Explication des caractéristiques sélectionnées", key="exp_feat"):
    for f in [x_feature, y_feature]:
        if f in FEATURE_DESCRIPTIONS:
            st.markdown(f"**{display_names.get(f,f)}**: {FEATURE_DESCRIPTIONS[f]}")
        else:
            st.markdown(f"**{display_names.get(f,f)}**: Pas de description disponible")

# -----------------------------
# Comparaison probabilités (déjà présent)
# -----------------------------
st.subheader("Comparaison des risques de défaut")
threshold = client_data[list(client_data.keys())[0]]["prediction"].get("threshold", 0.5)

sorted_clients = sorted(
    [(client_id, data["prediction"].get("probability", 0))
     for client_id, data in client_data.items()],
    key=lambda x: x[1]
)

fig = go.Figure()
risk_zones = [
    {"name": "RISQUE TRÈS FAIBLE", "min": 0, "max": 0.2, "color": "rgba(1, 133, 113, 0.4)"},
    {"name": "RISQUE FAIBLE", "min": 0.2, "max": 0.4, "color": "rgba(1, 133, 113, 0.6)"},
    {"name": "RISQUE MODÉRÉ", "min": 0.4, "max": threshold, "color": "rgba(1, 133, 113, 0.8)"},
    {"name": "RISQUE ÉLEVÉ", "min": threshold, "max": 0.7, "color": "rgba(166, 97, 26, 0.6)"},
    {"name": "RISQUE TRÈS ÉLEVÉ", "min": 0.7, "max": 1, "color": "rgba(166, 97, 26, 0.8)"}
]

for zone in risk_zones:
    fig.add_shape(type="rect", x0=zone["min"], x1=zone["max"], y0=-1, y1=len(sorted_clients), fillcolor=zone["color"], line=dict(width=0), layer="below")
    fig.add_annotation(x=(zone["min"] + zone["max"]) / 2, y=-1.5, text=zone["name"], showarrow=False,
                       font=dict(size=16, color='black', family='Arial', weight='bold'), bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1, borderpad=3)

fig.add_shape(type="line", x0=threshold, x1=threshold, y0=-2, y1=len(sorted_clients), line=dict(color="black", width=3, dash="dash"))
fig.add_annotation(x=threshold, y=-2.5, text=f"SEUIL: {threshold:.2f}", showarrow=False, font=dict(size=18, color="black", family="Arial", weight="bold"), bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1, borderpad=4, align='center')

for i, (client_id, probability) in enumerate(sorted_clients):
    decision = "ACCEPTÉ" if probability < threshold else "REFUSÉ"
    color = COLORBLIND_FRIENDLY_PALETTE['accepted'] if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE['refused']
    fig.add_trace(go.Bar(y=[i], x=[probability], orientation='h', marker=dict(color=color, line=dict(color='rgba(0,0,0,0.5)', width=1)), hovertemplate=f"Client #{client_id}<br>Probabilité: {probability:.1%}<br>Décision: {decision}<extra></extra>", name=f"Client #{client_id}", showlegend=False))
    fig.add_annotation(x=-0.05, y=i, text=f"#{client_id}", showarrow=False, xanchor="right", font=dict(size=14, color='black', family='Arial', weight='bold'), bgcolor='white', borderpad=2)
    position_x = probability + 0.03 if abs(probability - threshold) > 0.05 else probability + 0.06
    fig.add_annotation(x=position_x, y=i, text=f"{probability:.1%}", showarrow=False, xanchor="left", font=dict(size=14, color='black', family='Arial'), bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.3)', borderwidth=1, borderpad=2)

fig.update_layout(title={'text': "Comparaison des risques de défaut par client", 'font': {'size': 18, 'family': 'Arial', 'color': 'black'}, 'x': 0.5, 'xanchor': 'center'}, height=max(400, 100 + 50 * len(sorted_clients)), bargap=0.3, plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=60, r=100, t=80, b=100), xaxis=dict(title="Probabilité de défaut", titlefont=dict(size=14), range=[-0.1, 1.1], tickformat='.0%', tickvals=[0, 0.2, 0.4, threshold, 0.7, 1], ticktext=['0%', '20%', '40%', f'{threshold:.0%}', '70%', '100%'], tickfont=dict(size=14), showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'), yaxis=dict(title="", showticklabels=False, showgrid=False, range=[-3, len(sorted_clients)]), showlegend=False)

st.plotly_chart(fig, use_container_width=True)

# Navigation vers les pages
st.markdown('<h3 class="section-header">Outils d\'analyse pour le chargé de relation</h3>', unsafe_allow_html=True)
col_nav1, col_nav2, col_nav3 = st.columns(3)
with col_nav1:
    if st.button("📋 Profil détaillé et facteurs décisifs", key="btn_profile", use_container_width=True):
        st.switch_page("pages/1_Profil_Client.py")
with col_nav2:
    if st.button("📊 Comparaison avec clients similaires", key="btn_compare", use_container_width=True):
        st.switch_page("pages/2_Comparaison.py")
with col_nav3:
    if st.button("🔄 Simulation de modifications", key="btn_simulate", use_container_width=True):
        st.switch_page("pages/3_Simulation.py")

# Footer
st.markdown("""
<hr>
<div style="text-align: center; color: #333333; background-color: #f8f9fa; padding: 0.75rem; border-radius: 0.5rem; margin-top: 1rem;">
    <div>
        <strong>Comparaison de clients</strong> | Prêt à dépenser
    </div>
    <div>
        <span>Montants exprimés en roubles (₽)</span> | 
        <span>Contact support: poste 4242</span>
    </div>
</div>
""", unsafe_allow_html=True)
