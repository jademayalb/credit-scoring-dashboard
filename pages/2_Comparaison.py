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
    
    /* Style pour la jauge de score */
    .score-gauge {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .score-marker {
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Titre et présentation
st.title("Dashboard Credit Scoring")

# Alternative textuelle pour l'icône - Critère 1.1.1
st.markdown('<span class="visually-hidden" aria-hidden="false">Icône représentant une carte de crédit pour le dashboard de scoring</span>', unsafe_allow_html=True)

# Barre de navigation principale
tabs = ["Accueil", "Profil Client", "Comparaison", "Simulation"]
selected_tab = st.tabs(tabs)

# Déterminer l'index de l'onglet actif
active_tab_index = 2  # Pour la page Comparaison

# Gestion de la navigation
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

# Chargement de la liste des clients disponibles
with st.spinner("Chargement de la liste des clients..."):
    available_clients = get_available_clients(limit=UI_CONFIG["default_limit"])

if not available_clients:
    st.error("Impossible de charger la liste des clients.")
    st.stop()

# Créer un identifiant unique pour cette session si nécessaire
if "session_id" not in st.session_state:
    import random
    st.session_state.session_id = str(random.randint(10000, 99999))

# Sélection des clients à comparer (multiselect)
selected_clients = st.multiselect(
    "Sélectionnez 2 à 4 clients à comparer:",
    options=available_clients,
    default=[available_clients[0], available_clients[1]] if len(available_clients) > 1 else [available_clients[0]],
    max_selections=4,
    key=f"client_selection_{st.session_state.session_id}"
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

# Liste des features SHAP les plus importantes par défaut - MODIFIÉE POUR NE GARDER QUE LES 3 SCORES EXTERNES
default_features = ["EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1"]

# Sélection des caractéristiques à visualiser
available_features = list(client_data[list(client_data.keys())[0]]["details"]["features"].keys())

# Vérifier que les features par défaut existent dans les données
default_features_available = [f for f in default_features if f in available_features]

# Sélection des caractéristiques à visualiser avec les valeurs par défaut
selected_features = st.multiselect(
    "Sélectionnez les caractéristiques à comparer:",
    options=available_features,
    default=default_features_available,
    key=f"features_selection_{st.session_state.session_id}"
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
    st.plotly_chart(fig, use_container_width=True)
    
    # Explication des caractéristiques comparées
    with st.expander("Explication des caractéristiques"):
        for feature in selected_features:
            if feature in FEATURE_DESCRIPTIONS:
                st.markdown(f"**{feature}**: {FEATURE_DESCRIPTIONS[feature]}")
            else:
                st.markdown(f"**{feature}**: Pas de description disponible")

# NOUVEAU GRAPHIQUE: Comparaison des probabilités de défaut avec une jauge visuelle - VERSION AMÉLIORÉE
st.subheader("Comparaison des risques de défaut")

# Récupérer le seuil pour tous les clients (ils devraient avoir le même)
threshold = 0.5
if client_data:
    threshold = client_data[list(client_data.keys())[0]]["prediction"].get("threshold", 0.5)

# Trier les clients par probabilité croissante
sorted_clients = sorted(
    [(client_id, data["prediction"].get("probability", 0)) 
     for client_id, data in client_data.items()],
    key=lambda x: x[1]
)

# ========== VERSION ALTERNATIVE DU GRAPHIQUE ==========
# Création d'un graphique horizontal plus simple et plus efficace pour l'accessibilité
fig = go.Figure()

# Définir les différentes zones de risque
risk_zones = [
    {"name": "RISQUE TRÈS FAIBLE", "min": 0, "max": 0.2, "color": "rgba(1, 133, 113, 0.4)"},
    {"name": "RISQUE FAIBLE", "min": 0.2, "max": 0.4, "color": "rgba(1, 133, 113, 0.6)"},
    {"name": "RISQUE MODÉRÉ", "min": 0.4, "max": threshold, "color": "rgba(1, 133, 113, 0.8)"},
    {"name": "RISQUE ÉLEVÉ", "min": threshold, "max": 0.7, "color": "rgba(166, 97, 26, 0.6)"},
    {"name": "RISQUE TRÈS ÉLEVÉ", "min": 0.7, "max": 1, "color": "rgba(166, 97, 26, 0.8)"}
]

# Ajouter les zones de risque comme barres horizontales
for zone in risk_zones:
    fig.add_shape(
        type="rect",
        x0=zone["min"],
        x1=zone["max"],
        y0=-1,  # Position en bas
        y1=len(sorted_clients),  # Hauteur en fonction du nombre de clients
        fillcolor=zone["color"],
        line=dict(width=0),
        layer="below"
    )
    
    # Ajouter une annotation pour chaque zone
    fig.add_annotation(
        x=(zone["min"] + zone["max"]) / 2,  # Centre de la zone
        y=-1.5,  # Position sous l'axe
        text=zone["name"],
        showarrow=False,
        font=dict(size=16, color='black', family='Arial', weight='bold'),
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='black',
        borderwidth=1,
        borderpad=3,
        align='center'
    )

# Ajouter une ligne pour le seuil
fig.add_shape(
    type="line",
    x0=threshold,
    x1=threshold,
    y0=-2,
    y1=len(sorted_clients),
    line=dict(
        color="black",
        width=3,
        dash="dash",
    )
)

# Ajouter une annotation pour le seuil
fig.add_annotation(
    x=threshold,
    y=-2.5,
    text=f"SEUIL: {threshold:.2f}",
    showarrow=False,
    font=dict(
        size=18,
        color="black",
        family="Arial",
        weight="bold"
    ),
    bgcolor="rgba(255, 255, 255, 0.8)",
    bordercolor="black",
    borderwidth=1,
    borderpad=4,
    align='center'
)

# Ajouter des barres horizontales pour chaque client avec annotations
for i, (client_id, probability) in enumerate(sorted_clients):
    decision = "ACCEPTÉ" if probability < threshold else "REFUSÉ"
    color = COLORBLIND_FRIENDLY_PALETTE['accepted'] if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE['refused']
    
    # Ajouter une barre horizontale pour ce client
    fig.add_trace(go.Bar(
        y=[i],  # Position verticale
        x=[probability],  # Longueur de la barre (probabilité)
        orientation='h',
        marker=dict(
            color=color,
            line=dict(color='rgba(0,0,0,0.5)', width=1)
        ),
        # Retirer le texte sur la barre pour éviter les chevauchements
        hovertemplate=f"Client #{client_id}<br>Probabilité: {probability:.1%}<br>Décision: {decision}<extra></extra>",
        name=f"Client #{client_id}",
        showlegend=False
    ))
    
    # Ajouter une annotation avec l'ID client à gauche de la barre pour plus de clarté
    fig.add_annotation(
        x=-0.05,  # Légèrement à gauche de l'axe
        y=i,
        text=f"#{client_id}",
        showarrow=False,
        xanchor="right",
        font=dict(size=14, color='black', family='Arial', weight='bold'),
        bgcolor='white',
        borderpad=2
    )
    
    # Ajouter le pourcentage à droite de la barre (avec décalage pour éviter le trait vertical du seuil)
    position_x = probability + 0.03 if abs(probability - threshold) > 0.05 else probability + 0.06
    fig.add_annotation(
        x=position_x,
        y=i,
        text=f"{probability:.1%}",
        showarrow=False,
        xanchor="left",
        font=dict(size=14, color='black', family='Arial'),
        bgcolor='rgba(255, 255, 255, 0.7)',  # Fond semi-transparent
        bordercolor='rgba(0,0,0,0.3)',
        borderwidth=1,
        borderpad=2
    )

# Configurer la mise en page du graphique
fig.update_layout(
    # Titre au même style que le premier graphique
    title={
        'text': "Comparaison des risques de défaut par client",
        'font': {'size': 18, 'family': 'Arial', 'color': 'black'},
        'x': 0.5,
        'xanchor': 'center'
    },
    height=max(400, 100 + 50 * len(sorted_clients)),  # Ajuster la hauteur en fonction du nombre de clients
    bargap=0.3,
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=60, r=100, t=80, b=100),  # Augmenter la marge droite pour les annotations
    xaxis=dict(
        title="Probabilité de défaut",
        titlefont=dict(size=14),
        range=[-0.1, 1.1],  # Élargir légèrement pour les annotations
        tickformat='.0%',
        tickvals=[0, 0.2, 0.4, threshold, 0.7, 1],
        ticktext=['0%', '20%', '40%', f'{threshold:.0%}', '70%', '100%'],
        tickfont=dict(size=14),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)'
    ),
    yaxis=dict(
        title="",
        showticklabels=False,
        showgrid=False,
        range=[-3, len(sorted_clients)]
    ),
    showlegend=False
)

# Afficher le graphique
st.plotly_chart(fig, use_container_width=True)

# Explication du graphique
st.markdown("""
**Comment interpréter cette échelle de risque:**
- L'échelle représente la probabilité qu'un client ne rembourse pas son prêt, de 0% à 100%
- La ligne pointillée indique le seuil au-delà duquel un prêt est généralement refusé (52%)
- Les zones de couleur représentent différents niveaux de risque (du vert au rouge)
- Chaque barre représente un client, avec sa probabilité de défaut
""")

# Plus d'informations sur l'échelle de risque
with st.expander("En savoir plus sur l'échelle de risque"):
    st.markdown("""
    L'échelle de risque est divisée en 5 niveaux:
    
    1. **Risque très faible** (0-20%): Clients avec une excellente solvabilité, présentant un risque minimal de défaut.
    2. **Risque faible** (20-40%): Clients avec une bonne solvabilité, présentant un faible risque de défaut.
    3. **Risque modéré** (40-52%): Clients avec une solvabilité acceptable mais nécessitant une attention particulière.
    4. **Risque élevé** (52-70%): Clients présentant un risque significatif de défaut, généralement refusés.
    5. **Risque très élevé** (70-100%): Clients présentant un risque majeur de défaut, systématiquement refusés.
    
    Le seuil de décision est fixé à 52%. Ce seuil a été déterminé lors de l'entraînement du modèle pour optimiser l'équilibre entre l'acceptation de bons clients et le refus de clients à risque.
    """)

# Navigation vers les pages détaillées avec attributs d'accessibilité
st.markdown('<h3 class="section-header">Outils d\'analyse pour le chargé de relation</h3>', unsafe_allow_html=True)
col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    # Bouton avec icône ET texte (1.4.1)
    if st.button("📋 Profil détaillé et facteurs décisifs", key="btn_profile", use_container_width=True):
        st.switch_page("pages/1_Profil_Client.py")
        
with col_nav2:
    # Bouton avec icône ET texte (1.4.1)
    if st.button("📊 Comparaison avec clients similaires", key="btn_compare", use_container_width=True):
        st.switch_page("pages/2_Comparaison.py")
        
with col_nav3:
    # Bouton avec icône ET texte (1.4.1)
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
