"""
Page de comparaison entre clients
Permet de comparer les profils de différents clients et leurs scores

Ajouts et modifications :
- Analyse univariée (distribution des probabilités) avec position d'un client de référence et quantile
- Analyse bivariée amélioree :
    - Base : 100 premiers clients (application_test.csv via get_available_clients(limit=100))
    - Restriction aux 9 features prioritaires
    - Mode Simple (métiers) et Avancé (technique) pour la mise à l'échelle
    - Transformations applicables en interne (Log1p+Standard, Standard, MinMax, Aucun)
    - Mise en évidence :
        - autres clients : cercles
        - clients sélectionnés : diamant + label
        - client référence : étoile
    - Option d'affichage d'une droite de tendance (régression linéaire) et/ou de la ligne d'identité y=x
    - Option boxplot alternatif si comparaison catégorie vs numérique
- Conservation des valeurs brutes dans les tooltips
"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
    try:
        available_clients = get_available_clients(limit=UI_CONFIG.get("default_limit", 100)) or []
    except Exception:
        available_clients = []

if not available_clients:
    st.error("Impossible de charger la liste des clients.")
    st.stop()

# Normaliser les IDs (utilisation interne)
def normalize_id_list(lst):
    out = []
    for v in lst:
        try:
            out.append(int(v))
        except Exception:
            try:
                out.append(int(str(v)))
            except Exception:
                out.append(v)
    return out

available_clients = normalize_id_list(available_clients)

# Sélection des clients à comparer (multiselect)
selected_clients = st.multiselect(
    "Sélectionnez 2 à 4 clients à comparer:",
    options=available_clients,
    default=[available_clients[0], available_clients[1]] if len(available_clients) > 1 else [available_clients[0]],
    max_selections=4,
    key="client_selection"
)
selected_clients = normalize_id_list(selected_clients)

if len(selected_clients) < 2:
    st.warning("Veuillez sélectionner au moins deux clients pour la comparaison.")
    st.stop()

# Option: choisir un client de référence parmi les sélectionnés (pour l'univariée et bivariée)
reference_client = st.selectbox(
    "Choisir un client de référence (pour mise en évidence):",
    options=selected_clients,
    format_func=lambda x: f"Client #{x}"
)
try:
    reference_client = int(reference_client)
except Exception:
    pass

# Chargement des données des clients sélectionnés
with st.spinner("Chargement des données des clients sélectionnés..."):
    client_data = {}
    for client_id in selected_clients:
        try:
            pred = get_client_prediction(client_id)
            det = get_client_details(client_id)
            if pred and det:
                client_data[int(client_id)] = {"prediction": pred, "details": det}
            else:
                st.error(f"Impossible de charger les données pour le client {client_id}.")
        except Exception:
            st.error(f"Erreur lors du chargement du client {client_id}.")

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
        status_color = COLORBLIND_FRIENDLY_PALETTE.get('accepted', '#2ca02c') if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused', '#d62728')
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
        "Revenu annuel": st.column_config.NumberColumn("Revenu annuel", format=f"%d {UI_CONFIG.get('currency_symbol','')}"),
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
        "Montant demandé": st.column_config.NumberColumn("Montant demandé", format=f"%d {UI_CONFIG.get('currency_symbol','')}"),
        "Durée (mois)": st.column_config.NumberColumn("Durée (mois)", format="%d"),
        "Mensualité": st.column_config.NumberColumn("Mensualité", format=f"%d {UI_CONFIG.get('currency_symbol','')}"),
        "Valeur du bien": st.column_config.NumberColumn("Valeur du bien", format=f"%d {UI_CONFIG.get('currency_symbol','')}"),
        "Ratio mensualité/revenu": st.column_config.NumberColumn("Ratio mensualité/revenu", format="%.2f")
    },
    hide_index=True,
    use_container_width=True
)

# -----------------------------
# Analyse univariée (distribution des probabilités)
# -----------------------------
st.subheader("Analyse univariée : distribution des probabilités de défaut")

all_client_ids = normalize_id_list(get_available_clients(limit=UI_CONFIG.get("default_limit", 100)) or [])
probs = []
ids_with_prob = []
with st.spinner("Récupération des probabilités pour la distribution..."):
    for cid in all_client_ids:
        try:
            pred = get_client_prediction(cid)
            if pred:
                prob = pred.get("probability", 0)
                probs.append(prob)
                ids_with_prob.append((cid, prob))
        except Exception:
            continue

if not probs:
    st.info("Aucune probabilité disponible pour l'instant.")
else:
    dist_df = pd.DataFrame({"client_id": [c for c, p in ids_with_prob], "probability": [p for c, p in ids_with_prob]})
    fig = px.histogram(
        dist_df, x="probability", nbins=30,
        title="Distribution des probabilités de défaut (ensemble des clients)",
        color_discrete_sequence=[COLORBLIND_FRIENDLY_PALETTE.get("primary", "#636EFA")]
    )
    fig.update_layout(xaxis=dict(tickformat=".0%", title="Probabilité de défaut"), yaxis_title="Nombre de clients", height=350)
    try:
        ref_pred = client_data[int(reference_client)]["prediction"]
        ref_prob = ref_pred.get("probability", 0)
        fig.add_vline(x=ref_prob, line_dash="dash", line_color="black", annotation_text=f"Réf #{reference_client}: {ref_prob:.1%}", annotation_position="top right")
    except Exception:
        ref_prob = None
    st.plotly_chart(fig, use_container_width=True)

    if ref_prob is not None:
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
# Analyse bivariée (scatter plot) — nouvelle version complète
# -----------------------------
st.subheader("Analyse bivariée : comparer deux caractéristiques (100 clients, 9 features prioritaires)")

PRIOR_FEATURES = [
    "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1",
    "AMT_GOODS_PRICE", "AMT_CREDIT", "DAYS_EMPLOYED",
    "NAME_EDUCATION_TYPE", "AMT_ANNUITY", "DAYS_BIRTH"
]

# Récupérer 100 premiers clients pour l'analyse globale
limit_for_bi = 100
all_client_ids_100 = normalize_id_list(get_available_clients(limit=limit_for_bi) or [])[:limit_for_bi]
if not all_client_ids_100:
    # fallback rapide
    all_client_ids_100 = available_clients[:limit_for_bi]

if not all_client_ids_100:
    st.info("Impossible de récupérer la liste des 100 clients pour l'analyse bivariée.")
else:
    # construire un échantillon: récupère details et predictions une fois
    sample_rows = {}
    with st.spinner("Chargement des détails/predictions pour l'échantillon (100 clients)..."):
        for cid in all_client_ids_100:
            try:
                d = get_client_details(cid)
                p = get_client_prediction(cid)
                if d and "features" in d and p:
                    sample_rows[int(cid)] = {"features": d["features"], "prediction": p}
            except Exception:
                continue

    # Déterminer les features prior disponibles
    present_prior = [f for f in PRIOR_FEATURES if any(f in v["features"] for v in sample_rows.values())]

    if not present_prior:
        st.info("Aucune des 9 features prioritaires n'est disponible dans l'échantillon.")
    else:
        st.markdown("Sélection limitée aux 9 features importantes pour garantir une comparaison pertinente.")

        # Heuristique: numérique vs catégorielle
        def is_numeric_feat(feat, sample_map, sample_limit=40):
            vals = []
            for feats in list(sample_map.values())[:sample_limit]:
                v = feats["features"].get(feat, None)
                if v is None:
                    continue
                vals.append(v)
            if not vals:
                return False
            convertible = 0
            for v in vals:
                try:
                    float(v)
                    convertible += 1
                except Exception:
                    pass
            return (convertible / len(vals)) >= 0.8

        numeric_prior = [f for f in present_prior if is_numeric_feat(f, sample_rows)]
        categorical_prior = [f for f in present_prior if f not in numeric_prior]

        # UI pour scaling: Simple vs Avancé
        mode_scaling = st.radio(
            "Mode d'affichage des options de mise à l'échelle",
            options=["Simple (recommandé pour chargés de relation)", "Avancé (options techniques)"],
            index=0,
            horizontal=True
        )

        if mode_scaling.startswith("Simple"):
            friendly_choice = st.selectbox(
                "Comparer les caractéristiques — choix rapide",
                options=[
                    "Auto (recommandé)", "Brut — valeurs réelles", "Compacter montants (AMT_*)"
                ],
                index=0
            )
            if friendly_choice == "Auto (recommandé)":
                st.caption("Auto : l'application choisit une mise à l'échelle lisible selon les variables sélectionnées.")
                scale_choice_internal = "auto"
            elif friendly_choice == "Brut — valeurs réelles":
                st.caption("Affiche les valeurs réelles sans transformation.")
                scale_choice_internal = "none"
            else:
                st.caption("Compacter montants : applique une transformation adaptée aux montants pour rendre la lecture plus claire.")
                scale_choice_internal = "money_compact"
        else:
            scale_choice_internal = st.selectbox(
                "Transformation & scaling (mode avancé)",
                options=[
                    "Log1p puis Standard", "Standard (z-score)", "MinMax (0-1)", "Aucun (brut)"
                ],
                index=0
            )

        # Fonctions de mapping entre choix utilisateur et méthode concrète
        def resolve_internal_scaler(choice, x_feat=None, y_feat=None):
            money_feats = {"AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY"}
            if choice == "none":
                return "None"
            if choice == "money_compact":
                return "Log1p+Standard"
            if choice == "auto":
                if (x_feat in money_feats) or (y_feat in money_feats):
                    return "Log1p+Standard"
                else:
                    return "Standard"
            # options avancées
            if choice == "Log1p puis Standard":
                return "Log1p+Standard"
            if choice == "Standard (z-score)":
                return "Standard"
            if choice == "MinMax (0-1)":
                return "MinMax"
            if choice == "Aucun (brut)":
                return "None"
            return "Standard"

        # Sélection des axes (limités aux features numériques pour scatter)
        if not numeric_prior:
            st.info("Aucune feature numérique parmi les 9 priorisées pour un scatter. Essayez d'utiliser le mode boxplot pour catégories.")
        else:
            col_x, col_y = st.columns(2)
            with col_x:
                x_feature = st.selectbox("Axe X (numérique)", options=numeric_prior, index=0, format_func=lambda v: FEATURE_DESCRIPTIONS.get(v, v))
            with col_y:
                y_options = [f for f in numeric_prior if f != x_feature]
                if not y_options:
                    st.info("Pas d'autre feature numérique à afficher sur l'axe Y.")
                    y_feature = None
                else:
                    y_feature = st.selectbox("Axe Y (numérique)", options=y_options, index=0, format_func=lambda v: FEATURE_DESCRIPTIONS.get(v, v))

            if x_feature and y_feature:
                # Construction du dataset sur l'échantillon
                rows = []
                for cid, rec in sample_rows.items():
                    feats = rec["features"]
                    p = rec["prediction"]
                    x_raw = feats.get(x_feature, None)
                    y_raw = feats.get(y_feature, None)
                    if x_raw is None or y_raw is None:
                        continue
                    prob = p.get("probability", 0)
                    threshold = p.get("threshold", 0.5)
                    decision = "ACCEPTÉ" if prob < threshold else "REFUSÉ"
                    rows.append({
                        "client_id": int(cid),
                        "x_raw": x_raw,
                        "y_raw": y_raw,
                        "probability": prob,
                        "decision": decision
                    })

                if not rows:
                    st.info("Pas assez de données pour tracer le scatter.")
                else:
                    df_all = pd.DataFrame(rows)
                    df_all["x_num"] = pd.to_numeric(df_all["x_raw"], errors="coerce")
                    df_all["y_num"] = pd.to_numeric(df_all["y_raw"], errors="coerce")
                    df_all = df_all.dropna(subset=["x_num", "y_num"]).reset_index(drop=True)

                    # transformation log safe
                    def safe_log1p_series(s):
                        s = pd.to_numeric(s, errors="coerce")
                        out = s.copy()
                        mask_pos = s > 0
                        out.loc[mask_pos] = np.log1p(s.loc[mask_pos])
                        return out

                    concrete_method = resolve_internal_scaler(scale_choice_internal, x_feature, y_feature)

                    if concrete_method == "None":
                        df_all["x_plot"] = df_all["x_num"]
                        df_all["y_plot"] = df_all["y_num"]
                        scaler_x = scaler_y = None
                        x_label = FEATURE_DESCRIPTIONS.get(x_feature, x_feature)
                        y_label = FEATURE_DESCRIPTIONS.get(y_feature, y_feature)
                    else:
                        if concrete_method == "Log1p+Standard":
                            x_vals = safe_log1p_series(df_all["x_num"]).values.reshape(-1, 1)
                            y_vals = safe_log1p_series(df_all["y_num"]).values.reshape(-1, 1)
                        else:
                            x_vals = df_all["x_num"].values.reshape(-1, 1)
                            y_vals = df_all["y_num"].values.reshape(-1, 1)

                        if concrete_method == "MinMax":
                            scaler_x = MinMaxScaler()
                            scaler_y = MinMaxScaler()
                        else:
                            scaler_x = StandardScaler()
                            scaler_y = StandardScaler()

                        df_all["x_plot"] = scaler_x.fit_transform(x_vals).flatten()
                        df_all["y_plot"] = scaler_y.fit_transform(y_vals).flatten()
                        x_label = f"{FEATURE_DESCRIPTIONS.get(x_feature, x_feature)} ({concrete_method})"
                        y_label = f"{FEATURE_DESCRIPTIONS.get(y_feature, y_feature)} ({concrete_method})"

                    # Indiquer clients sélectionnés et référence
                    selected_set = set([int(x) for x in selected_clients])
                    df_all["is_selected"] = df_all["client_id"].isin(selected_set)
                    df_all["is_reference"] = df_all["client_id"] == int(reference_client)

                    # Construire la figure
                    fig_bi = go.Figure()

                    df_other = df_all[~df_all["is_selected"] & ~df_all["is_reference"]]
                    df_sel = df_all[df_all["is_selected"] & ~df_all["is_reference"]]
                    df_ref = df_all[df_all["is_reference"]]

                    # Autres clients
                    if not df_other.empty:
                        fig_bi.add_trace(go.Scatter(
                            x=df_other["x_plot"],
                            y=df_other["y_plot"],
                            mode="markers",
                            marker=dict(
                                size=9,
                                symbol="circle",
                                color=[COLORBLIND_FRIENDLY_PALETTE.get('accepted','#2ca02c') if d == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused','#d62728') for d in df_other["decision"]],
                                line=dict(width=0.3, color='rgba(0,0,0,0.12)')
                            ),
                            hovertemplate=(
                                "Client #%{customdata[0]}<br>"
                                f"{FEATURE_DESCRIPTIONS.get(x_feature,x_feature)}: " + "%{customdata[1]}<br>"
                                f"{FEATURE_DESCRIPTIONS.get(y_feature,y_feature)}: " + "%{customdata[2]}<br>"
                                "Probabilité: %{customdata[3]:.1%}<extra></extra>"
                            ),
                            customdata=np.stack([df_other["client_id"], df_other["x_raw"], df_other["y_raw"], df_other["probability"]], axis=-1),
                            name="Autres clients",
                            showlegend=True
                        ))

                    # Clients sélectionnés
                    if not df_sel.empty:
                        fig_bi.add_trace(go.Scatter(
                            x=df_sel["x_plot"],
                            y=df_sel["y_plot"],
                            mode="markers+text",
                            marker=dict(
                                size=14,
                                symbol="diamond",
                                color=[COLORBLIND_FRIENDLY_PALETTE.get('accepted','#2ca02c') if d == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused','#d62728') for d in df_sel["decision"]],
                                line=dict(width=1, color='black')
                            ),
                            text=[f"#{int(c)}" for c in df_sel["client_id"]],
                            textposition="top center",
                            hovertemplate=(
                                "Client #%{customdata[0]}<br>"
                                f"{FEATURE_DESCRIPTIONS.get(x_feature,x_feature)}: " + "%{customdata[1]}<br>"
                                f"{FEATURE_DESCRIPTIONS.get(y_feature,y_feature)}: " + "%{customdata[2]}<br>"
                                "Probabilité: %{customdata[3]:.1%}<extra></extra>"
                            ),
                            customdata=np.stack([df_sel["client_id"], df_sel["x_raw"], df_sel["y_raw"], df_sel["probability"]], axis=-1),
                            name="Clients choisis",
                            showlegend=True
                        ))

                    # Client référence
                    if not df_ref.empty:
                        r = df_ref.iloc[0]
                        fig_bi.add_trace(go.Scatter(
                            x=[r["x_plot"]],
                            y=[r["y_plot"]],
                            mode="markers+text",
                            marker=dict(size=22, symbol="star", color="black", line=dict(width=2, color='white')),
                            text=[f"Réf #{int(r['client_id'])}"],
                            textposition="bottom center",
                            hovertemplate=(
                                f"Client #{int(r['client_id'])}<br>"
                                f"{FEATURE_DESCRIPTIONS.get(x_feature,x_feature)}: {r['x_raw']}<br>"
                                f"{FEATURE_DESCRIPTIONS.get(y_feature,y_feature)}: {r['y_raw']}<br>"
                                f"Probabilité: {r['probability']:.1%}<extra></extra>"
                            ),
                            name="Client référence",
                            showlegend=True
                        ))

                    fig_bi.update_layout(
                        title=f"{FEATURE_DESCRIPTIONS.get(x_feature,x_feature)} vs {FEATURE_DESCRIPTIONS.get(y_feature,y_feature)} (échantillon {len(df_all)} clients)",
                        xaxis_title=x_label,
                        yaxis_title=y_label,
                        template="simple_white",
                        height=650,
                        legend=dict(itemsizing='constant')
                    )

                    # Options droites (trendline, identity)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        show_trend = st.checkbox("Afficher droite de tendance (régression linéaire)", value=False)
                    with col_b:
                        show_identity = st.checkbox("Afficher ligne y = x (identité)", value=False)

                    if show_trend:
                        if len(df_all) < 2:
                            st.warning("Pas assez de points pour calculer une régression.")
                        else:
                            try:
                                coeffs = np.polyfit(df_all["x_plot"], df_all["y_plot"], 1)
                                x_line = np.linspace(df_all["x_plot"].min(), df_all["x_plot"].max(), 200)
                                y_line = np.polyval(coeffs, x_line)
                                fig_bi.add_trace(go.Scatter(
                                    x=x_line, y=y_line, mode="lines",
                                    line=dict(color="black", dash="dash"),
                                    name="Droite de tendance"
                                ))
                            except Exception as e:
                                st.warning(f"Impossible de tracer la droite de tendance : {e}")

                    if show_identity:
                        try:
                            x0, x1 = df_all["x_plot"].min(), df_all["x_plot"].max()
                            fig_bi.add_trace(go.Scatter(
                                x=[x0, x1], y=[x0, x1],
                                mode="lines",
                                line=dict(color="gray", dash="dot"),
                                name="y = x"
                            ))
                        except Exception:
                            pass

                    # Option percentile référence
                    if st.checkbox("Afficher percentile de la référence sur la probabilité (échantillon)", value=False):
                        if not df_all.empty and int(reference_client) in df_all["client_id"].values:
                            ref_prob_val = df_all.loc[df_all["client_id"] == int(reference_client), "probability"].values[0]
                            percentile_ref = (np.sum(df_all["probability"] <= ref_prob_val) / len(df_all)) * 100
                            st.markdown(f"Client Réf #{reference_client} : percentile = {percentile_ref:.1f}% (sur l'échantillon affiché)")

                    st.plotly_chart(fig_bi, use_container_width=True)

                    # Boxplot alternatif si catégorielle vs numérique
                    if categorical_prior:
                        if st.checkbox("Afficher boxplot alternatif si catégorielle vs numérique ?", value=False):
                            by_cat = None
                            num = None
                            if x_feature in categorical_prior and y_feature not in categorical_prior:
                                by_cat = x_feature; num = y_feature
                            elif y_feature in categorical_prior and x_feature not in categorical_prior:
                                by_cat = y_feature; num = x_feature

                            if by_cat and num:
                                cat_rows = []
                                for cid, rec in sample_rows.items():
                                    feats = rec["features"]
                                    p = rec["prediction"]
                                    cat_val = feats.get(by_cat, None)
                                    num_val = feats.get(num, None)
                                    if cat_val is None or num_val is None:
                                        continue
                                    try:
                                        cat_rows.append({
                                            "client_id": int(cid),
                                            "category": str(cat_val),
                                            "value": float(num_val),
                                            "probability": p.get("probability", 0)
                                        })
                                    except Exception:
                                        continue
                                if cat_rows:
                                    df_cat = pd.DataFrame(cat_rows)
                                    fig_box = px.box(df_cat, x="category", y="value", points="all",
                                                     title=f"Boxplot: {FEATURE_DESCRIPTIONS.get(num,num)} par {FEATURE_DESCRIPTIONS.get(by_cat, by_cat)}",
                                                     labels={"category": FEATURE_DESCRIPTIONS.get(by_cat,by_cat), "value": FEATURE_DESCRIPTIONS.get(num,num)},
                                                     color_discrete_sequence=[COLORBLIND_FRIENDLY_PALETTE.get("primary","#636EFA")])
                                    st.plotly_chart(fig_box, use_container_width=True)
                                else:
                                    st.info("Pas assez de données pour le boxplot alternatif.")
                            else:
                                st.info("Le boxplot alternatif nécessite une feature catégorielle et une numérique.")

# -----------------------------
# Explications des caractéristiques sélectionnées
# -----------------------------
if st.button("Explication des caractéristiques sélectionnées", key="exp_feat"):
    for f in [locals().get("x_feature"), locals().get("y_feature")]:
        if f:
            if f in FEATURE_DESCRIPTIONS:
                st.markdown(f"**{FEATURE_DESCRIPTIONS.get(f,f)}**: {FEATURE_DESCRIPTIONS.get(f)}")
            else:
                st.markdown(f"**{f}**: Pas de description disponible")

# -----------------------------
# Comparaison probabilités (bar chart)
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
    color = COLORBLIND_FRIENDLY_PALETTE.get('accepted','#2ca02c') if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused','#d62728')
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
