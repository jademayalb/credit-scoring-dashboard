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

# Fonction pour afficher la barre de navigation commune avec attributs ARIA
def display_navigation():
    st.markdown(
        """
        <nav aria-label="Navigation principale" role="navigation">
            <div style="margin-bottom: 1rem;">
                <a href="/" class="nav-button" role="button" aria-label="Accueil">Accueil</a>
                <a href="/Profil_Client" class="nav-button active" role="button" aria-current="page" aria-label="Page actuelle: Profil Client">Profil Client</a>
                <a href="/Comparaison" class="nav-button" role="button" aria-label="Comparaison">Comparaison</a>
                <a href="/Simulation" class="nav-button" role="button" aria-label="Simulation">Simulation</a>
            </div>
        </nav>
        """,
        unsafe_allow_html=True
    )

# Affichage de la barre de navigation
display_navigation()

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
tab1, tab2 = st.tabs(["Profil client", "Facteurs décisionnels"])

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
    # Section 3: Analyse des facteurs d'importance (Graphique interactif #1) - Version simplifiée
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
            text=[f"{abs(v):.2f}" for v in feature_values],  # Format simplifié à 2 décimales
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
        
        # Table simplifiée des facteurs principaux - Version corrigée avec distinction claire
        simple_explanations = []

        for feature, value in top_features:
            if feature == "EXT_SOURCE_1":
                display_name = "Score normalisé - Source externe 1"
            elif feature == "EXT_SOURCE_2":
                display_name = "Score normalisé - Source externe 2"
            elif feature == "EXT_SOURCE_3":
                display_name = "Score normalisé - Source externe 3"
            else:
                display_name = FEATURE_DESCRIPTIONS.get(feature, feature)
                
            # Valeur d'impact (SHAP) arrondie pour affichage
            impact_value = abs(round(value, 2))
            impact_direction = "Favorable" if value < 0 else "Défavorable"
            
            # Valeur réelle de la caractéristique
            real_value = "N/A"
            if feature in details.get('features', {}) and details['features'][feature] is not None:
                if isinstance(details['features'][feature], (int, float)):
                    real_value = round(details['features'][feature], 2)
                else:
                    real_value = details['features'][feature]
            
            simple_explanations.append({
                "Facteur": display_name,
                "Valeur réelle": real_value,
                "Impact": f"{impact_direction} ({impact_value})"
            })

        # Afficher le tableau avec des noms de colonnes plus clairs
        st.dataframe(
            pd.DataFrame(simple_explanations),
            column_config={
                "Facteur": st.column_config.TextColumn("Facteur"),
                "Valeur réelle": st.column_config.TextColumn("Valeur du client"),
                "Impact": st.column_config.TextColumn("Impact sur la décision")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Description textuelle pour lecteurs d'écran - Version simplifiée
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
