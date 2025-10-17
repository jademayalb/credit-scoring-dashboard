"""
Page de profil détaillé client
Permet d'analyser en profondeur les facteurs influençant la décision de crédit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.api_client import get_client_prediction, get_client_details, get_feature_importance

# Import de la configuration
from config import (
    COLORBLIND_FRIENDLY_PALETTE, UI_CONFIG, 
    FEATURE_DESCRIPTIONS, DEFAULT_THRESHOLD
)

# Configuration de la page - Critère 2.4.2 (Titre de page descriptif)
st.set_page_config(
    page_title="Profil Client Détaillé - Dashboard de Scoring Crédit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS pour l'accessibilité - Critère 1.4.3 (Contraste) et 1.4.4 (Redimensionnement du texte)
st.markdown("""
<style>
    /* Classe pour les textes destinés aux lecteurs d'écran */
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
    
    /* Adaptation des polices et tailles pour faciliter le redimensionnement */
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

# Titre et présentation
st.title("Dashboard Credit Scoring")

# Alternative textuelle pour l'icône - Critère 1.1.1
st.markdown('<span class="visually-hidden" aria-hidden="false">Icône représentant une carte de crédit pour le dashboard de scoring</span>', unsafe_allow_html=True)

# Barre de navigation principale
tabs = ["Accueil", "Profil Client", "Comparaison", "Simulation"]
selected_tab = st.tabs(tabs)

# Déterminer l'index de l'onglet actif
active_tab_index = 1  # Pour la page Profil Client

# Gestion de la navigation
if selected_tab[0].button("Accueil", key="nav_home", use_container_width=True):
    st.switch_page("Home.py")
elif selected_tab[2].button("Comparaison", key="nav_compare", use_container_width=True):
    st.switch_page("pages/2_Comparaison.py")
elif selected_tab[3].button("Simulation", key="nav_simulation", use_container_width=True):
    st.switch_page("pages/3_Simulation.py")

# Vérification de l'ID client dans la session
if "client_id" not in st.session_state:
    st.warning("Aucun client sélectionné. Veuillez retourner à la page d'accueil pour sélectionner un client.")
    if st.button("Retour à l'accueil"):
        st.switch_page("Home.py")
    st.stop()

# Récupération de l'ID client de la session
client_id = st.session_state.client_id

# Titre de la page avec ID client
st.title(f"Profil détaillé du client #{client_id}")
# Description pour les lecteurs d'écran - Critère 1.1.1 (Contenu non textuel)
st.markdown(f'<span class="visually-hidden">Cette page présente les informations détaillées et l\'analyse de la demande de crédit pour le client numéro {client_id}.</span>', unsafe_allow_html=True)

# Chargement des données client
with st.spinner("Chargement des données détaillées..."):
    prediction = get_client_prediction(client_id)
    details = get_client_details(client_id)
    feature_importance = get_feature_importance(client_id)

if not prediction or not details:
    st.error("Impossible de récupérer les informations du client.")
    if st.button("Retour à l'accueil", key="btn_back_error"):
        st.switch_page("Home.py")
    st.stop()

# Affichage du statut de la demande (Acceptée/Refusée)
decision = prediction.get('decision', 'INCONNU')
probability = prediction.get('probability', 0)
threshold = prediction.get('threshold', DEFAULT_THRESHOLD)

status_color = COLORBLIND_FRIENDLY_PALETTE['accepted'] if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE['refused']
# Utiliser à la fois icône et texte - Critère 1.4.1 (Utilisation de la couleur)
status_icon = "✅" if decision == "ACCEPTÉ" else "❌"
status_text = "Accepté" if decision == "ACCEPTÉ" else "Refusé" 

# Bannière de statut en haut de la page avec contraste amélioré et texte explicite
st.markdown(
    f"""
    <div style="padding: 0.75rem 1.25rem; border-radius: 0.5rem; background-color: {status_color}22; border: 2px solid {status_color}; margin-bottom: 1.5rem;">
        <h2 style="color: {status_color}; margin: 0; display: flex; align-items: center; font-size: 1.5rem;">
            <span aria-hidden="true">{status_icon}</span> 
            <span>Décision: <strong>{status_text}</strong> • Probabilité de défaut: <strong>{probability:.1%}</strong></span>
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)
# Version accessible pour les lecteurs d'écran - Critère 1.1.1
st.markdown(f'<div class="visually-hidden">La demande de crédit a été {status_text}. La probabilité de défaut calculée est de {probability:.1%}.</div>', unsafe_allow_html=True)

# Organisation en tabs pour les différentes sections
tab1, tab2, tab3 = st.tabs(["Profil client", "Facteurs décisionnels", "Analyse bivariée"])

with tab1:
    # Section 1: Informations détaillées du client
    st.header("Informations personnelles et financières")
    
    # Colonnes pour les informations
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.subheader("Informations personnelles")
            
            # Création d'un tableau pour afficher les informations personnelles
            personal_data = {
                "Caractéristique": [
                    "Genre", "Âge", "Éducation", "Statut familial",
                    "Nombre d'enfants", "Nombre de membres dans le foyer",
                    "Revenu annuel", "Ancienneté d'emploi"
                ],
                "Valeur": [
                    details['personal_info'].get('gender', ''),
                    f"{details['personal_info'].get('age', '')} ans",
                    details['personal_info'].get('education', ''),
                    details['personal_info'].get('family_status', ''),
                    details['personal_info'].get('children_count', 0),
                    details['personal_info'].get('family_size', 0),
                    f"{details['personal_info'].get('income', 0):,.0f} {UI_CONFIG['currency_symbol']}",
                    f"{details['personal_info'].get('employment_years', 0)} ans"
                ]
            }
            
            # Affichage du tableau avec style
            personal_df = pd.DataFrame(personal_data)
            st.dataframe(
                personal_df,
                column_config={
                    "Caractéristique": st.column_config.TextColumn("Caractéristique"),
                    "Valeur": st.column_config.TextColumn("Valeur")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Description pour lecteurs d'écran - Critère 1.1.1
            personal_summary = ", ".join([f"{row['Caractéristique']}: {row['Valeur']}" for _, row in personal_df.iterrows()])
            st.markdown(f'<div class="visually-hidden">Tableau des informations personnelles: {personal_summary}</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container(border=True):
            st.subheader("Détails de la demande de crédit")
            
            # Calcul de ratios pertinents
            payment_ratio = details['credit_info'].get('annuity', 0) * 12 / max(details['personal_info'].get('income', 1), 1)
            credit_goods_ratio = details['credit_info'].get('amount', 0) / max(details['credit_info'].get('goods_price', 1), 1)
            
            # Création d'un tableau pour les informations de crédit
            credit_data = {
                "Caractéristique": [
                    "Montant demandé", "Durée du crédit", "Mensualité",
                    "Valeur du bien", "Ratio mensualité/revenu",
                    "Ratio montant/valeur du bien"
                ],
                "Valeur": [
                    f"{details['credit_info'].get('amount', 0):,.0f} {UI_CONFIG['currency_symbol']}",
                    f"{details['credit_info'].get('credit_term', 0)} mois",
                    f"{details['credit_info'].get('annuity', 0):,.0f} {UI_CONFIG['currency_symbol']}/mois",
                    f"{details['credit_info'].get('goods_price', 0):,.0f} {UI_CONFIG['currency_symbol']}",
                    f"{payment_ratio:.1%}",
                    f"{credit_goods_ratio:.2f}"
                ]
            }
            
            # Affichage du tableau de crédit
            credit_df = pd.DataFrame(credit_data)
            st.dataframe(
                credit_df,
                column_config={
                    "Caractéristique": st.column_config.TextColumn("Caractéristique"),
                    "Valeur": st.column_config.TextColumn("Valeur")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Description pour lecteurs d'écran - Critère 1.1.1
            credit_summary = ", ".join([f"{row['Caractéristique']}: {row['Valeur']}" for _, row in credit_df.iterrows()])
            st.markdown(f'<div class="visually-hidden">Tableau des informations de crédit: {credit_summary}</div>', unsafe_allow_html=True)
    
    # Section 2: Données externes
    st.header("Données externes")
    
    # Colonnes pour l'affichage des données
    col_ext1, col_ext2 = st.columns(2)
    
    with col_ext1:
        with st.container(border=True):
            st.subheader("Scores normalisés")
            
            # Utiliser uniquement des données réelles du client avec les noms exacts demandés
            real_scores_data = {
                "Indicateur": [
                    "Score normalisé - Source externe 1",
                    "Score normalisé - Source externe 2", 
                    "Score normalisé - Source externe 3"
                ],
                "Valeur": [
                    f"{details.get('features', {}).get('EXT_SOURCE_1', 'N/A'):.3f}" if details.get('features', {}).get('EXT_SOURCE_1') is not None else "N/A",
                    f"{details.get('features', {}).get('EXT_SOURCE_2', 'N/A'):.3f}" if details.get('features', {}).get('EXT_SOURCE_2') is not None else "N/A",
                    f"{details.get('features', {}).get('EXT_SOURCE_3', 'N/A'):.3f}" if details.get('features', {}).get('EXT_SOURCE_3') is not None else "N/A"
                ]
            }
            
            # Affichage du tableau avec style
            ext_scores_df = pd.DataFrame(real_scores_data)
            st.dataframe(
                ext_scores_df,
                column_config={
                    "Indicateur": st.column_config.TextColumn("Indicateur"),
                    "Valeur": st.column_config.TextColumn("Valeur")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Description pour lecteurs d'écran - Critère 1.1.1
            scores_summary = ", ".join([f"{row['Indicateur']}: {row['Valeur']}" for _, row in ext_scores_df.iterrows()])
            st.markdown(f'<div class="visually-hidden">Tableau des scores externes: {scores_summary}</div>', unsafe_allow_html=True)
    
    with col_ext2:
        with st.container(border=True):
            st.subheader("Informations administratives")
            
            # Utiliser uniquement des données réelles, sans Jours depuis publication ID
            admin_data = {
                "Indicateur": [
                    "Possession d'une voiture",
                    "Possession d'un bien immobilier"
                ],
                "Valeur": [
                    "Oui" if details.get('features', {}).get('FLAG_OWN_CAR') == 'Y' else "Non",
                    "Oui" if details.get('features', {}).get('FLAG_OWN_REALTY') == 'Y' else "Non"
                ]
            }
            
            # Affichage du tableau avec style
            admin_df = pd.DataFrame(admin_data)
            st.dataframe(
                admin_df,
                column_config={
                    "Indicateur": st.column_config.TextColumn("Indicateur"),
                    "Valeur": st.column_config.TextColumn("Valeur")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Description pour lecteurs d'écran - Critère 1.1.1
            admin_summary = ", ".join([f"{row['Indicateur']}: {row['Valeur']}" for _, row in admin_df.iterrows()])
            st.markdown(f'<div class="visually-hidden">Tableau des informations administratives: {admin_summary}</div>', unsafe_allow_html=True)

with tab2:
    # Section 3: Analyse des facteurs d'importance (Graphique interactif #1)
    st.header("Facteurs influençant la décision")
    
    # Vérifier si des valeurs d'importance sont disponibles
    if feature_importance:
        # Trier les features par importance absolue
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Limiter aux 5 features les plus importantes pour simplifier
        top_features = sorted_features[:5]
        
        # Préparer les données pour le graphique
        feature_names = []
        feature_values = []
        colors = []
        
        for feature, value in top_features:
            # Utiliser les nouveaux noms pour les scores externes
            if feature == "EXT_SOURCE_1":
                display_name = "Score normalisé - Source externe 1"
            elif feature == "EXT_SOURCE_2":
                display_name = "Score normalisé - Source externe 2"
            elif feature == "EXT_SOURCE_3":
                display_name = "Score normalisé - Source externe 3"
            else:
                display_name = FEATURE_DESCRIPTIONS.get(feature, feature)
                
            feature_names.append(display_name)
            feature_values.append(value)
            
            # Couleur basée sur l'impact (positif ou négatif)
            colors.append(
                COLORBLIND_FRIENDLY_PALETTE["positive"] if value < 0 else
                COLORBLIND_FRIENDLY_PALETTE["negative"]
            )
        
        # Créer un graphique simplifié
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=feature_values,
            y=feature_names,
            orientation='h',
            marker_color=colors,
            text=[f"{abs(v):.2f}" for v in feature_values],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Impact: %{x:.2f}<extra></extra>'
        ))
        
        # Layout simplifié
        fig.update_layout(
            title="Principaux facteurs influençant la décision",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                title="Impact"
            )
        )
        
        # Afficher le graphique interactif
        st.plotly_chart(fig, use_container_width=True)
        
        # Légende explicative simplifiée
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border: 1px solid #dee2e6;">
            <h4 style="margin-top: 0;">Comment interpréter ce graphique?</h4>
            <ul style="margin-bottom: 0;">
                <li><span style="color: #018571; font-weight: bold;">Les barres vertes</span> représentent des facteurs favorables au client.</li>
                <li><span style="color: #a6611a; font-weight: bold;">Les barres rouges</span> représentent des facteurs défavorables au client.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Ajout de la note explicative avant le tableau
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <strong>Note :</strong> Les valeurs dans le graphique ci-dessus représentent l'<em>impact</em> de chaque facteur sur la décision.
            Le tableau ci-dessous montre à la fois la <em>valeur réelle</em> de chaque facteur pour ce client et son impact sur la décision.
        </div>
        """, unsafe_allow_html=True)
        
        # SOLUTION 5: MAPPING DIRECT DANS LA PAGE STREAMLIT
        # Préparer les données pour le tableau simplifié
        table_data = []
        
        for feature, value in top_features:
            # Déterminer le nom d'affichage
            if feature == "EXT_SOURCE_1":
                display_name = "Score normalisé - Source externe 1"
            elif feature == "EXT_SOURCE_2":
                display_name = "Score normalisé - Source externe 2"
            elif feature == "EXT_SOURCE_3":
                display_name = "Score normalisé - Source externe 3"
            else:
                display_name = FEATURE_DESCRIPTIONS.get(feature, feature)
            
            # Valeur d'impact arrondie
            impact_value = abs(round(value, 2))
            impact_direction = "Favorable" if value < 0 else "Défavorable"
            
            # Récupérer directement la valeur réelle
            real_value = "N/A"
            try:
                if feature in details.get('features', {}):
                    if feature == "DAYS_BIRTH":
                        real_value = f"{abs(int(details['features'][feature] / 365))} ans"
                    elif feature == "DAYS_EMPLOYED":
                        if details['features'][feature] == 365243:
                            real_value = "Sans emploi"
                        else:
                            real_value = f"{abs(int(details['features'][feature] / 365))} ans"
                    else:
                        real_value = details['features'].get(feature, "N/A")
                        if isinstance(real_value, (int, float)):
                            real_value = round(real_value, 2)
            except Exception as e:
                real_value = "Erreur"
            
            table_data.append({
                "Facteur": display_name,
                "Valeur du client": real_value,
                "Impact": f"{impact_direction} ({impact_value})"
            })
        
        # Afficher le tableau simplifié
        st.dataframe(
            pd.DataFrame(table_data),
            column_config={
                "Facteur": st.column_config.TextColumn("Facteur"),
                "Valeur du client": st.column_config.TextColumn("Valeur du client"),
                "Impact": st.column_config.TextColumn("Impact sur la décision")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Description textuelle pour lecteurs d'écran
        st.markdown(f'<div class="visually-hidden">Graphique montrant les 5 facteurs principaux influençant la décision de crédit.</div>', unsafe_allow_html=True)
    
    else:
        st.info("Les valeurs d'importance des caractéristiques ne sont pas disponibles pour ce client.")
    
    # Section d'analyse comparative des caractéristiques (Graphique interactif #2)
    st.header("Analyse comparative des caractéristiques")
    
    # Sélection des caractéristiques à visualiser (interactivité)
    selected_features = st.multiselect(
        label="Sélectionner des caractéristiques à comparer:",
        options=list(details.get('features', {}).keys()),
        default=list(details.get('features', {}).keys())[:3],
        help="Sélectionnez une ou plusieurs caractéristiques pour voir comment les valeurs du client se positionnent."
    )
    
    if selected_features:
        # Préparation des données pour le graphique
        feature_data = []
        
        for feature in selected_features:
            if feature in details.get('features', {}):
                # Utiliser les nouveaux noms pour les scores externes
                if feature == "EXT_SOURCE_1":
                    display_name = "Score normalisé - Source externe 1"
                elif feature == "EXT_SOURCE_2":
                    display_name = "Score normalisé - Source externe 2"
                elif feature == "EXT_SOURCE_3":
                    display_name = "Score normalisé - Source externe 3"
                else:
                    display_name = FEATURE_DESCRIPTIONS.get(feature, feature)
                    
                feature_data.append({
                    "Caractéristique": display_name,
                    "Valeur client": details['features'][feature],
                })
        
        # Création du graphique comparatif - Critères 1.4.3 et 1.4.4
        fig = px.bar(
            pd.DataFrame(feature_data),
            x="Caractéristique",
            y="Valeur client",
            text="Valeur client",
            color_discrete_sequence=[COLORBLIND_FRIENDLY_PALETTE["primary"]],
            title="Valeurs des caractéristiques sélectionnées"
        )
        
        fig.update_layout(
            title_font=dict(size=20),
            xaxis_title="",
            yaxis_title="Valeur",
            height=400,
            margin=dict(l=20, r=20, t=50, b=100),
            xaxis=dict(tickangle=-45, tickfont=dict(size=14)),
            yaxis=dict(tickfont=dict(size=14)),
            font=dict(size=14),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            )
        )
        
        fig.update_traces(
            texttemplate='%{y:.2f}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Valeur: %{y:.2f}<extra></extra>'
        )
        
        # Afficher le graphique interactif
        st.plotly_chart(fig, use_container_width=True)
        
        # Description textuelle pour les lecteurs d'écran - Critère 1.1.1
        features_values = [f"La caractéristique {d['Caractéristique']} a une valeur de {d['Valeur client']:.2f}" for d in feature_data]
        features_text = ". ".join(features_values)
        st.markdown(f'<div class="visually-hidden">Graphique comparatif des caractéristiques sélectionnées. {features_text}</div>', unsafe_allow_html=True)
        
        # Tableau des valeurs
        st.dataframe(
            pd.DataFrame(feature_data),
            column_config={
                "Caractéristique": st.column_config.TextColumn("Caractéristique"),
                "Valeur client": st.column_config.NumberColumn("Valeur client", format="%.2f")
            },
            hide_index=True,
            use_container_width=True
        )
        
    else:
        st.info("Veuillez sélectionner au moins une caractéristique pour l'analyse comparative.")

# Nouvel onglet pour l'analyse bivariée
with tab3:
    st.header("Analyse bivariée des caractéristiques")
    
    st.markdown("""
    Cette section vous permet d'explorer la relation entre deux caractéristiques. 
    Sélectionnez deux variables ci-dessous pour visualiser leur relation à l'aide d'un nuage de points.
    """)
    
    # Filtrer pour ne garder que les features numériques
    features_dict = details.get('features', {})
    numeric_features = []
    for feature, value in features_dict.items():
        if isinstance(value, (int, float)) and feature not in ['SK_ID_CURR']:
            numeric_features.append(feature)
    
    # Sélection des deux caractéristiques pour l'analyse bivariée
    col_select1, col_select2 = st.columns(2)
    
    with col_select1:
        x_feature = st.selectbox(
            "Sélectionner la caractéristique pour l'axe X:",
            options=numeric_features,
            format_func=lambda f: FEATURE_DESCRIPTIONS.get(f, f),
            key="bivar_x_feature",
            index=0 if len(numeric_features) > 0 else None
        )
    
    with col_select2:
        # Exclure la feature déjà sélectionnée pour X
        y_features = [f for f in numeric_features if f != x_feature]
        y_feature = st.selectbox(
            "Sélectionner la caractéristique pour l'axe Y:",
            options=y_features,
            format_func=lambda f: FEATURE_DESCRIPTIONS.get(f, f),
            key="bivar_y_feature",
            index=0 if len(y_features) > 0 else None
        )
    
    # Vérifier si les deux caractéristiques sont sélectionnées
    if x_feature and y_feature:
        # Obtenir les données pour créer le nuage de points
        x_value = features_dict.get(x_feature)
        y_value = features_dict.get(y_feature)
        
        # Obtenir les noms d'affichage pour les caractéristiques
        x_display = FEATURE_DESCRIPTIONS.get(x_feature, x_feature)
        y_display = FEATURE_DESCRIPTIONS.get(y_feature, y_feature)
        
        # Gestion des cas spéciaux pour les jours
        if x_feature == "DAYS_BIRTH":
            x_value = abs(x_value) / 365.25
            x_display = "Âge (années)"
        elif x_feature == "DAYS_EMPLOYED":
            if x_value == 365243:
                x_value = 0
                x_display = "Ancienneté d'emploi (années)"
            else:
                x_value = abs(x_value) / 365.25
                x_display = "Ancienneté d'emploi (années)"
        
        if y_feature == "DAYS_BIRTH":
            y_value = abs(y_value) / 365.25
            y_display = "Âge (années)"
        elif y_feature == "DAYS_EMPLOYED":
            if y_value == 365243:
                y_value = 0
                y_display = "Ancienneté d'emploi (années)"
            else:
                y_value = abs(y_value) / 365.25
                y_display = "Ancienneté d'emploi (années)"
        
        # Créer un DataFrame pour le point du client actuel
        client_point = pd.DataFrame({
            "x": [x_value],
            "y": [y_value],
            "client": [f"Client #{client_id}"]
        })
        
        # Création du nuage de points avec contexte
        try:
            # Essayer de charger un échantillon de données pour ajouter du contexte
            # Ceci pourrait être remplacé par un appel API pour récupérer des données similaires
            # Pour cet exemple, nous allons simuler quelques points de contexte
            import random
            
            # Générer des points de contexte autour des valeurs du client
            # En production, ces valeurs viendraient de l'API ou d'une base de données
            context_size = 50
            x_std = max(abs(x_value) * 0.2, 0.1)  # écart-type de 20% de la valeur ou au moins 0.1
            y_std = max(abs(y_value) * 0.2, 0.1)  # écart-type de 20% de la valeur ou au moins 0.1
            
            # Générer des valeurs normalement distribuées autour des valeurs du client
            context_x = np.random.normal(x_value, x_std, context_size)
            context_y = np.random.normal(y_value, y_std, context_size)
            
            # Assurer que les valeurs soient positives si nécessaire
            if x_feature in ["DAYS_BIRTH", "DAYS_EMPLOYED"]:
                context_x = np.abs(context_x)
            if y_feature in ["DAYS_BIRTH", "DAYS_EMPLOYED"]:
                context_y = np.abs(context_y)
            
            # Créer un DataFrame pour les points de contexte
            context_df = pd.DataFrame({
                "x": context_x,
                "y": context_y,
                "client": ["Autres clients" for _ in range(context_size)]
            })
            
            # Combiner le point client avec les points de contexte
            plot_df = pd.concat([client_point, context_df])
            
            # Créer un nuage de points avec distinction claire du client actuel
            fig = px.scatter(
                plot_df,
                x="x",
                y="y",
                color="client",
                color_discrete_map={
                    f"Client #{client_id}": COLORBLIND_FRIENDLY_PALETTE["primary"],
                    "Autres clients": "rgba(180, 180, 180, 0.5)"  # Points de contexte en gris transparent
                },
                labels={
                    "x": x_display,
                    "y": y_display,
                    "client": "Client"
                },
                title=f"Relation entre {x_display} et {y_display}",
                height=600
            )
            
            # Mise en forme du graphique pour une meilleure lisibilité
            fig.update_traces(
                marker=dict(
                    size=[12 if c == f"Client #{client_id}" else 8 for c in plot_df["client"]],
                    opacity=[1 if c == f"Client #{client_id}" else 0.6 for c in plot_df["client"]],
                    line=dict(
                        width=[2 if c == f"Client #{client_id}" else 0 for c in plot_df["client"]],
                        color=[COLORBLIND_FRIENDLY_PALETTE["primary"] if c == f"Client #{client_id}" else "lightgrey" for c in plot_df["client"]]
                    )
                )
            )
            
            # Amélioration de la mise en page
            fig.update_layout(
                xaxis=dict(
                    title=dict(text=x_display, font=dict(size=16)),
                    tickfont=dict(size=14),
                    zeroline=True,
                    zerolinecolor='rgba(0,0,0,0.2)',
                    gridcolor='rgba(0,0,0,0.05)'
                ),
                yaxis=dict(
                    title=dict(text=y_display, font=dict(size=16)),
                    tickfont=dict(size=14),
                    zeroline=True,
                    zerolinecolor='rgba(0,0,0,0.2)',
                    gridcolor='rgba(0,0,0,0.05)'
                ),
                legend=dict(
                    font=dict(size=14),
                    bordercolor='rgba(0,0,0,0.1)',
                    borderwidth=1
                ),
                title=dict(
                    text=f"Relation entre {x_display} et {y_display}",
                    font=dict(size=20),
                    x=0.5,
                    xanchor='center'
                ),
                hovermode='closest',
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=14,
                    font_family="Arial"
                )
            )
            
            # Ajouter une ligne de tendance pour visualiser la relation
            fig.add_trace(
                go.Scatter(
                    x=[min(plot_df["x"]), max(plot_df["x"])],
                    y=[min(plot_df["y"]), max(plot_df["y"])] if np.corrcoef(plot_df["x"], plot_df["y"])[0, 1] > 0 else [max(plot_df["y"]), min(plot_df["y"])],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0.3)', dash='dash'),
                    name='Tendance',
                    hoverinfo='skip'
                )
            )
            
            # Afficher le graphique
            st.plotly_chart(fig, use_container_width=True)
            
            # Calcul du coefficient de corrélation
            corr = np.corrcoef(plot_df["x"], plot_df["y"])[0, 1]
            
            # Affichage du coefficient de corrélation et interprétation
            corr_strength = ""
            if abs(corr) < 0.3:
                corr_strength = "faible"
            elif abs(corr) < 0.7:
                corr_strength = "modérée"
            else:
                corr_strength = "forte"
                
            corr_direction = "positive" if corr >= 0 else "négative"
            
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border: 1px solid #dee2e6;">
                <h4 style="margin-top: 0;">Analyse de la relation</h4>
                <p>Le coefficient de corrélation entre ces deux caractéristiques est de <strong>{corr:.2f}</strong>.</p>
                <p>Cela indique une relation <strong>{corr_strength} {corr_direction}</strong> entre {x_display} et {y_display}.</p>
                <p><em>Note: Les points gris représentent des clients simulés pour illustrer le contexte. Dans une version complète, ces données proviendraient de clients réels similaires.</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Description textuelle pour les lecteurs d'écran - Critère 1.1.1
            st.markdown(f"""
            <div class="visually-hidden">
                Nuage de points montrant la relation entre {x_display} et {y_display}. 
                Le client #{client_id} a une valeur de {x_value:.2f} pour {x_display} et {y_value:.2f} pour {y_display}.
                La corrélation entre ces caractéristiques est {corr:.2f}, indiquant une relation {corr_strength} {corr_direction}.
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Impossible de générer le graphique d'analyse bivariée: {str(e)}")
            st.info("Essayez de sélectionner d'autres caractéristiques ou vérifiez que les données sont bien numériques.")
    
    else:
        st.info("Veuillez sélectionner deux caractéristiques différentes pour visualiser leur relation.")
    
    # Explication de l'utilité de l'analyse bivariée
    with st.expander("En savoir plus sur l'analyse bivariée"):
        st.markdown("""
        ### Qu'est-ce que l'analyse bivariée?
        
        L'analyse bivariée permet d'explorer la relation entre deux variables. Dans le contexte du crédit, cette analyse peut révéler des corrélations importantes qui aident à comprendre les facteurs de risque.
        
        ### Comment utiliser cette analyse?
        
        - **Pour identifier des relations importantes**: Par exemple, voir si l'âge est corrélé avec le score externe
        - **Pour comprendre les compensations**: Certains facteurs négatifs peuvent être compensés par d'autres positifs
        - **Pour guider les conseils aux clients**: Si un client est refusé, vous pouvez identifier quelles combinaisons de facteurs seraient plus favorables
        
        ### Exemples d'analyses pertinentes:
        
        - **Revenu vs Mensualité**: Pour voir si le montant de la mensualité est proportionnel au revenu
        - **Âge vs Score externe**: Pour voir si l'âge influence la fiabilité du score
        - **Montant du crédit vs Valeur du bien**: Pour évaluer le ratio de couverture
        """)

# Notes et actions du chargé de relation
st.header("Notes et actions")

col_notes1, col_notes2 = st.columns([2, 1])

with col_notes1:
    # Système de notes (sauvegardé dans la session)
    if "detailed_notes" not in st.session_state:
        st.session_state.detailed_notes = {}
    
    current_notes = st.session_state.detailed_notes.get(client_id, "")
    
    # Améliorer l'accessibilité du champ de texte
    new_notes = st.text_area(
        label="Notes de suivi détaillées",
        value=current_notes,
        height=150,
        placeholder="Saisissez ici vos observations, échanges avec le client, ou actions de suivi...",
        help="Ces notes sont sauvegardées automatiquement dans votre session",
        key="detailed_notes_field"
    )
    
    if new_notes != current_notes:
        st.session_state.detailed_notes[client_id] = new_notes
        st.success("Notes enregistrées")

with col_notes2:
    # Actions possibles
    with st.container(border=True):
        st.subheader("Actions rapides")
        
        # Boutons avec icône ET texte - Critère 1.4.1
        if st.button("📧 Envoyer un récapitulatif", 
                     help="Envoie un résumé de cette analyse au client par email",
                     use_container_width=True):
            st.info("Fonctionnalité d'envoi d'email à implémenter.")
            
        if decision == "REFUSÉ" and st.button("📝 Demander une révision", 
                                              help="Demande une nouvelle évaluation du dossier",
                                              use_container_width=True):
            st.info("Redirection vers le formulaire de révision.")
            
        if st.button("🔙 Retour à l'accueil", 
                     help="Retourner à la page d'accueil",
                     use_container_width=True):
            st.switch_page("Home.py")

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

# Footer avec informations de version
st.markdown("""
<hr>
<div style="text-align: center; color: #333333; background-color: #f8f9fa; padding: 0.75rem; border-radius: 0.5rem; margin-top: 1rem;">
    <div>
        <strong>Profil client détaillé</strong> | Prêt à dépenser
    </div>
    <div>
        <span>Montants exprimés en roubles (₽)</span> | 
        <span>Contact support: poste 4242</span>
    </div>
</div>
""", unsafe_allow_html=True)
