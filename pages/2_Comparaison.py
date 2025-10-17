"""
Page de comparaison entre clients
Permet de comparer les profils de différents clients et leurs scores
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.api_client import get_client_prediction, get_client_details, get_available_clients

# Import de la configuration
from config import (
    COLORBLIND_FRIENDLY_PALETTE, UI_CONFIG, 
    FEATURE_DESCRIPTIONS
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
    
    /* Contraste amélioré pour les tableaux */
    .dataframe th {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
    
    .dataframe td {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Adaptation des polices et tailles */
    body, .stMarkdown, .stText {
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.75rem !important; }
    h3 { font-size: 1.5rem !important; }
    
    .nav-button {
        display: inline-block;
        padding: 0.5rem 1rem !important;
        margin-right: 0.75rem !important;
        border-radius: 0.25rem !important;
        text-decoration: none !important;
        font-weight: 500 !important;
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        border: 1px solid #777777 !important;
    }
    
    .nav-button.active {
        background-color: #3366ff !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour afficher la barre de navigation commune
def display_navigation():
    st.markdown(
        """
        <nav aria-label="Navigation principale" role="navigation">
            <div style="margin-bottom: 1rem;">
                <a href="/" class="nav-button" role="button" aria-label="Accueil">Accueil</a>
                <a href="/Profil_Client" class="nav-button" role="button" aria-label="Profil Client">Profil Client</a>
                <a href="/Comparaison" class="nav-button active" role="button" aria-current="page" aria-label="Page actuelle: Comparaison">Comparaison</a>
                <a href="/Simulation" class="nav-button" role="button" aria-label="Simulation">Simulation</a>
            </div>
        </nav>
        """,
        unsafe_allow_html=True
    )

# Affichage de la barre de navigation
display_navigation()

# Titre de la page
st.title("Comparaison de profils clients")
st.markdown("""
Cette page vous permet de comparer plusieurs profils clients côte à côte, 
pour analyser les différences entre leurs caractéristiques et comprendre les variations dans les décisions de crédit.
""")

# Chargement de la liste des clients disponibles
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
    key="client_selection"  # Clé unique pour le widget
)

if len(selected_clients) < 2:
    st.warning("Veuillez sélectionner au moins deux clients pour la comparaison.")
    st.stop()

# Chargement des données des clients sélectionnés
with st.spinner("Chargement des données des clients sélectionnés..."):
    client_data = {}
    
    for client_id in selected_clients:
        prediction = get_client_prediction(client_id)
        details = get_client_details(client_id)
        
        if prediction and details:
            client_data[client_id] = {
                "prediction": prediction,
                "details": details
            }
        else:
            st.error(f"Impossible de charger les données pour le client {client_id}.")

if not client_data:
    st.error("Aucune donnée client n'a pu être chargée.")
    st.stop()

# Affichage des cartes de statut des clients
st.subheader("Statut des demandes de crédit")

# Créer un layout flexible pour les cartes
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

# Comparaison des informations personnelles
st.subheader("Comparaison des informations personnelles")

# Préparer les données pour la comparaison
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

# Affichage du tableau comparatif
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

# Comparaison des informations de crédit
st.subheader("Comparaison des crédits demandés")

# Préparer les données pour la comparaison des crédits
credit_data = []

for client_id, data in client_data.items():
    details = data["details"]
    credit_info = details.get("credit_info", {})
    
    # Calcul des ratios
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

# Affichage du tableau comparatif des crédits
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

# Visualisation comparative
st.subheader("Analyse comparative des caractéristiques")

# Sélection des caractéristiques à visualiser
available_features = list(client_data[list(client_data.keys())[0]]["details"]["features"].keys())
selected_features = st.multiselect(
    "Sélectionnez les caractéristiques à comparer:",
    options=available_features,
    default=["EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH"],
    key="features_selection"  # Clé unique pour le widget
)

if not selected_features:
    st.info("Veuillez sélectionner au moins une caractéristique à comparer.")
else:
    # Préparation des données pour le graphique
    chart_data = []
    
    for client_id, data in client_data.items():
        features = data["details"]["features"]
        
        for feature in selected_features:
            # Formater les noms des features pour l'affichage
            if feature in FEATURE_DESCRIPTIONS:
                display_name = FEATURE_DESCRIPTIONS[feature]
            else:
                display_name = feature
                
            # Traitement spécial pour certaines features
            value = features.get(feature, 0)
            
            if feature == "DAYS_BIRTH":
                # Convertir les jours négatifs en âge en années
                value_display = abs(value) / 365.25
                unit = "ans"
            elif feature == "DAYS_EMPLOYED":
                # Convertir les jours négatifs en années d'emploi
                if value == 365243:
                    value_display = 0
                    unit = "ans (sans emploi)"
                else:
                    value_display = abs(value) / 365.25
                    unit = "ans"
            else:
                value_display = value
                unit = ""
                
            chart_data.append({
                "Client ID": f"Client #{client_id}",
                "Caractéristique": display_name,
                "Valeur": value_display,
                "Unité": unit
            })
    
    # Création du graphique de comparaison
    chart_df = pd.DataFrame(chart_data)
    
    fig = px.bar(
        chart_df, 
        x="Caractéristique", 
        y="Valeur", 
        color="Client ID",
        barmode="group",
        color_discrete_sequence=list(COLORBLIND_FRIENDLY_PALETTE.values())[:len(client_data)],
        labels={"Valeur": "Valeur", "Caractéristique": "Caractéristique"},
        title="Comparaison des caractéristiques entre clients"
    )
    
    # Amélioration du graphique pour l'accessibilité
    fig.update_layout(
        font_family="Arial",
        font_size=14,
        legend_title_font_size=16,
        title_font_size=18,
        height=500,
        margin=dict(l=20, r=20, t=50, b=50)
    )
    
    # Affichage du graphique
    st.plotly_chart(fig, use_container_width=True, key="features_chart")
    
    # Explication des caractéristiques comparées
    with st.expander("Explication des caractéristiques"):
        for feature in selected_features:
            if feature in FEATURE_DESCRIPTIONS:
                st.markdown(f"**{feature}**: {FEATURE_DESCRIPTIONS[feature]}")
            else:
                st.markdown(f"**{feature}**: Pas de description disponible")

# Probabilités de défaut
st.subheader("Comparaison des probabilités de défaut")

# Données pour le graphique
proba_data = []
for client_id, data in client_data.items():
    probability = data["prediction"].get("probability", 0)
    threshold = data["prediction"].get("threshold", 0.5)
    
    proba_data.append({
        "client_id": f"Client #{client_id}",
        "probability": probability
    })

# Création du dataframe
proba_df = pd.DataFrame(proba_data)

# Graphique de comparaison des probabilités
fig = px.bar(
    proba_df,
    x="client_id",
    y="probability",
    color="client_id",
    color_discrete_sequence=list(COLORBLIND_FRIENDLY_PALETTE.values())[:len(proba_df)],
    labels={"probability": "Probabilité de défaut", "client_id": "Client"},
    title="Probabilité de défaut par client"
)

# Ajout d'une ligne horizontale pour le seuil
if client_data:
    threshold = client_data[list(client_data.keys())[0]]["prediction"].get("threshold", 0.5)
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(proba_df) - 0.5,
        y0=threshold,
        y1=threshold,
        line=dict(
            color=COLORBLIND_FRIENDLY_PALETTE["threshold"],
            width=2,
            dash="dash",
        )
    )
    
    # Ajouter une annotation pour le seuil
    fig.add_annotation(
        x=len(proba_df) - 1,
        y=threshold * 1.1,
        text=f"Seuil: {threshold:.2f}",
        showarrow=False,
        font=dict(
            size=14,
            color=COLORBLIND_FRIENDLY_PALETTE["threshold"]
        )
    )

# Amélioration du graphique pour l'accessibilité
fig.update_layout(
    font_family="Arial",
    font_size=14,
    legend_title_font_size=16,
    title_font_size=18,
    height=400,
    margin=dict(l=20, r=20, t=50, b=50)
)

# Format des étiquettes en pourcentage
fig.update_yaxes(tickformat=".1%")

# Ajouter les valeurs sur les barres
fig.update_traces(
    texttemplate='%{y:.1%}',
    textposition='outside'
)

# Affichage du graphique
st.plotly_chart(fig, use_container_width=True, key="probabilities_chart")

# Explication du graphique
st.markdown("""
**Comment interpréter ce graphique:**
- Les barres représentent la probabilité qu'un client ne rembourse pas son prêt
- La ligne pointillée montre le seuil au-delà duquel un prêt est généralement refusé
- Plus la probabilité est basse, meilleur est le profil du client
""")

# Conclusion
st.header("Analyse comparative")

# Génération automatique d'un résumé comparatif simple
st.write("### Résumé de la comparaison", key="comparison_title")

if len(client_data) >= 2:
    # Générer une clé unique basée sur les clients sélectionnés
    comparison_key = "_".join(map(str, sorted(selected_clients)))
    
    # Trouver le client avec la probabilité la plus basse (meilleur profil)
    best_client_id = min(client_data.items(), key=lambda x: x[1]["prediction"].get("probability", 1))[0]
    best_client_proba = client_data[best_client_id]["prediction"].get("probability", 0)
    
    # Trouver le client avec la probabilité la plus haute (pire profil)
    worst_client_id = max(client_data.items(), key=lambda x: x[1]["prediction"].get("probability", 0))[0]
    worst_client_proba = client_data[worst_client_id]["prediction"].get("probability", 0)
    
    # Utiliser empty containers pour remplacer dynamiquement le contenu
    summary_container = st.empty()
    differences_container = st.container()
    
    summary_container.markdown(f"""
    D'après l'analyse comparative:
    
    - Le client #{best_client_id} présente le meilleur profil avec une probabilité de défaut de **{best_client_proba:.1%}**
    - Le client #{worst_client_id} présente le profil le plus risqué avec une probabilité de défaut de **{worst_client_proba:.1%}**
    
    Les principales différences observées concernent:
    """, key=f"summary_{comparison_key}")
    
    # Identifier les différences les plus marquantes
    best_features = client_data[best_client_id]["details"]["features"]
    worst_features = client_data[worst_client_id]["details"]["features"]
    
    # Comparaison des sources externes (généralement très importantes)
    differences = []
    
    for feature in ["EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1", "DAYS_BIRTH", "AMT_INCOME_TOTAL"]:
        if feature in best_features and feature in worst_features:
            best_value = best_features[feature]
            worst_value = worst_features[feature]
            
            # Format différent selon la feature
            if feature == "DAYS_BIRTH":
                best_age = abs(best_value) / 365
                worst_age = abs(worst_value) / 365
                diff_pct = abs(best_age - worst_age) / max(best_age, worst_age) * 100
                
                if diff_pct > 10:  # Différence significative d'âge
                    differences.append((f"- L'âge: **{best_age:.0f} ans** contre **{worst_age:.0f} ans**", diff_pct))
            
            elif feature.startswith("EXT_SOURCE"):
                diff_pct = abs(best_value - worst_value) / max(abs(best_value), abs(worst_value), 0.01) * 100
                
                if diff_pct > 15:  # Différence significative de score externe
                    feature_name = FEATURE_DESCRIPTIONS.get(feature, feature)
                    differences.append((f"- {feature_name}: **{best_value:.2f}** contre **{worst_value:.2f}**", diff_pct))
            
            elif feature == "AMT_INCOME_TOTAL":
                diff_pct = abs(best_value - worst_value) / max(abs(best_value), abs(worst_value), 0.01) * 100
                
                if diff_pct > 20:  # Différence significative de revenu
                    differences.append((f"- Le revenu annuel: **{best_value:,.0f} {UI_CONFIG['currency_symbol']}** contre **{worst_value:,.0f} {UI_CONFIG['currency_symbol']}**", diff_pct))
    
    # Trier les différences par importance
    differences.sort(key=lambda x: x[1], reverse=True)
    
    with differences_container:
        # Afficher les différences ou un message par défaut
        if differences:
            for i, (diff_text, _) in enumerate(differences[:3]):  # Limiter à 3 différences pour la clarté
                st.markdown(diff_text, key=f"diff_{i}_{comparison_key}")
        else:
            st.markdown("- Les profils présentent des caractéristiques relativement similaires malgré les scores différents.", 
                      key=f"no_diff_{comparison_key}")
    
else:
    st.info("Sélectionnez au moins 2 clients pour obtenir une analyse comparative.")

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
