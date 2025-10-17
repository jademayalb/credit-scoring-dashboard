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

# Configuration de la page
st.set_page_config(
    page_title="Profil Client Détaillé - Dashboard de Scoring Crédit",  # Titre plus descriptif
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS pour l'accessibilité
st.markdown("""
<style>
    /* Styles pour l'accessibilité */
    .high-contrast-text {
        color: #000000 !important;
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 0.2rem;
        border: 1px solid #cccccc;
    }
    
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
    
    /* Amélioration du contraste pour les éléments interactifs */
    button, .stButton>button {
        color: #000000 !important;
        background-color: #f8f8f8 !important;
        border: 2px solid #666666 !important;
        font-weight: 500 !important;
    }
    
    button:hover, .stButton>button:hover {
        background-color: #e0e0e0 !important;
    }
    
    /* Amélioration du focus pour la navigation clavier */
    a:focus, button:focus, select:focus, textarea:focus, input:focus {
        outline: 3px solid #4b9fff !important;
        outline-offset: 2px !important;
    }
    
    /* Style pour les liens */
    a {
        color: #0066cc !important;
        text-decoration: underline !important;
    }

    /* Adaptation des polices et tailles pour faciliter le redimensionnement */
    body, .stMarkdown, .stText {
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        line-height: 1.3 !important;
    }
    
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
    
    /* Couleurs avec contraste renforcé */
    .alert-success {
        color: #155724 !important;
        background-color: #d4edda !important;
        border: 1px solid #155724 !important;
    }
    
    .alert-warning {
        color: #856404 !important;
        background-color: #fff3cd !important;
        border: 1px solid #856404 !important;
    }
    
    .alert-danger {
        color: #721c24 !important;
        background-color: #f8d7da !important;
        border: 1px solid #721c24 !important;
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
    st.markdown('<div class="alert-warning" style="padding: 1rem; border-radius: 0.5rem;">', unsafe_allow_html=True)
    st.warning("Aucun client sélectionné. Veuillez retourner à la page d'accueil pour sélectionner un client.")
    if st.button("Retour à l'accueil", key="btn_back_home"):
        st.switch_page("Home.py")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Récupération de l'ID client de la session
client_id = st.session_state.client_id

# Titre de la page avec ID client
st.title(f"Profil détaillé du client #{client_id}")
# Description pour les lecteurs d'écran
st.markdown(f'<span class="visually-hidden">Cette page présente les informations détaillées et l\'analyse de la demande de crédit pour le client numéro {client_id}.</span>', unsafe_allow_html=True)

# Chargement des données client avec barre de progression
with st.spinner("Chargement des données détaillées..."):
    progress_bar = st.progress(0)
    st.markdown('<div class="visually-hidden" aria-live="polite">Chargement des données en cours...</div>', unsafe_allow_html=True)
    
    # Chargement progressif des données
    progress_bar.progress(33)
    prediction = get_client_prediction(client_id)
    progress_bar.progress(66)
    details = get_client_details(client_id)
    progress_bar.progress(100)
    feature_importance = get_feature_importance(client_id)
    st.markdown('<div class="visually-hidden" aria-live="polite">Chargement des données terminé.</div>', unsafe_allow_html=True)
    progress_bar.empty()

if not prediction or not details:
    st.markdown('<div class="alert-danger" style="padding: 1rem; border-radius: 0.5rem;">', unsafe_allow_html=True)
    st.error("Impossible de récupérer les informations du client.")
    if st.button("Retour à l'accueil", key="btn_back_error"):
        st.switch_page("Home.py")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Affichage du statut de la demande (Acceptée/Refusée)
decision = prediction.get('decision', 'INCONNU')
probability = prediction.get('probability', 0)
threshold = prediction.get('threshold', DEFAULT_THRESHOLD)

status_color = COLORBLIND_FRIENDLY_PALETTE['accepted'] if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE['refused']
# Utiliser à la fois icône et texte pour l'accessibilité
status_icon = "✅" if decision == "ACCEPTÉ" else "❌"
status_text = "Accepté" if decision == "ACCEPTÉ" else "Refusé" 

# Bannière de statut en haut de la page avec contraste amélioré et texte explicite
st.markdown(
    f"""
    <div style="padding: 0.75rem 1.25rem; border-radius: 0.5rem; background-color: {status_color}22; border: 2px solid {status_color}; margin-bottom: 1.5rem;" role="status" aria-live="polite">
        <h2 style="color: {status_color}; margin: 0; display: flex; align-items: center; font-size: 1.5rem;">
            <span aria-hidden="true">{status_icon}</span> 
            <span>Décision: <strong>{status_text}</strong> • Probabilité de défaut: <strong>{probability:.1%}</strong></span>
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)
# Version accessible pour les lecteurs d'écran
st.markdown(f'<div class="visually-hidden">La demande de crédit a été {status_text}. La probabilité de défaut calculée est de {probability:.1%}.</div>', unsafe_allow_html=True)

# Organisation en tabs pour les différentes sections avec attributs ARIA
tab1, tab2, tab3 = st.tabs(["Profil client", "Facteurs décisionnels", "Historique"])

with tab1:
    # Section 1: Informations détaillées du client
    st.header("Informations personnelles et financières")
    st.markdown('<div class="visually-hidden">Cette section présente les informations personnelles et financières du client.</div>', unsafe_allow_html=True)
    
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
                    "Revenu annuel", "Type d'emploi", "Ancienneté d'emploi"
                ],
                "Valeur": [
                    details['personal_info'].get('gender', ''),
                    f"{details['personal_info'].get('age', '')} ans",
                    details['personal_info'].get('education', ''),
                    details['personal_info'].get('family_status', ''),
                    details['personal_info'].get('children_count', 0),
                    details['personal_info'].get('family_size', 0),
                    f"{details['personal_info'].get('income', 0):,.0f} {UI_CONFIG['currency_symbol']}",
                    details['personal_info'].get('employment_type', ''),
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
            
            # Description pour lecteurs d'écran
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
                    "Valeur du bien", "Type de bien", "Ratio mensualité/revenu",
                    "Ratio montant/valeur du bien"
                ],
                "Valeur": [
                    f"{details['credit_info'].get('amount', 0):,.0f} {UI_CONFIG['currency_symbol']}",
                    f"{details['credit_info'].get('credit_term', 0)} mois",
                    f"{details['credit_info'].get('annuity', 0):,.0f} {UI_CONFIG['currency_symbol']}/mois",
                    f"{details['credit_info'].get('goods_price', 0):,.0f} {UI_CONFIG['currency_symbol']}",
                    details['credit_info'].get('name_goods_category', ''),
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
            
            # Description pour lecteurs d'écran
            credit_summary = ", ".join([f"{row['Caractéristique']}: {row['Valeur']}" for _, row in credit_df.iterrows()])
            st.markdown(f'<div class="visually-hidden">Tableau des informations de crédit: {credit_summary}</div>', unsafe_allow_html=True)
    
    # Section 2: Historique et comportement client
    st.header("Historique du client")
    
    # Colonnes pour historique et comportement
    col_hist1, col_hist2 = st.columns(2)
    
    with col_hist1:
        with st.container(border=True):
            st.subheader("Antécédents de crédit")
            
            # Simuler quelques données d'historique (à remplacer par des données réelles)
            history_data = {
                "Indicateur": [
                    "Nombre de prêts précédents",
                    "Défauts de paiement antérieurs",
                    "Retards de paiement (30+ jours)",
                    "Score bureau de crédit",
                    "Ancienneté relation (années)"
                ],
                "Valeur": [
                    details.get('credit_history', {}).get('previous_loans_count', 0),
                    details.get('credit_history', {}).get('previous_defaults', 0),
                    details.get('credit_history', {}).get('late_payments', 0),
                    details.get('credit_history', {}).get('credit_score', 'N/A'),
                    details.get('credit_history', {}).get('years_with_bank', 0)
                ]
            }
            
            # Affichage du tableau d'historique
            history_df = pd.DataFrame(history_data)
            st.dataframe(
                history_df,
                column_config={
                    "Indicateur": st.column_config.TextColumn("Indicateur"),
                    "Valeur": st.column_config.TextColumn("Valeur")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Description pour lecteurs d'écran
            history_summary = ", ".join([f"{row['Indicateur']}: {row['Valeur']}" for _, row in history_df.iterrows()])
            st.markdown(f'<div class="visually-hidden">Tableau des antécédents de crédit: {history_summary}</div>', unsafe_allow_html=True)
    
    with col_hist2:
        with st.container(border=True):
            st.subheader("Documents et vérifications")
            
            # Simuler les données de vérification (à remplacer par des données réelles)
            verification_data = {
                "Document/Vérification": [
                    "Pièce d'identité",
                    "Justificatif de revenus",
                    "Justificatif de domicile",
                    "Vérification d'emploi",
                    "Vérification téléphonique"
                ],
                "Statut": [
                    "✅ Vérifié",  # Utiliser à la fois icône et texte
                    "✅ Vérifié",
                    "✅ Vérifié",
                    "⚠️ En attente",
                    "❌ Non effectuée"
                ]
            }
            
            # Affichage du tableau de vérifications
            verification_df = pd.DataFrame(verification_data)
            st.dataframe(
                verification_df,
                column_config={
                    "Document/Vérification": st.column_config.TextColumn("Document/Vérification"),
                    "Statut": st.column_config.TextColumn("Statut")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Description pour lecteurs d'écran
            verification_summary = ", ".join([f"{row['Document/Vérification']}: {row['Statut']}" for _, row in verification_df.iterrows()])
            st.markdown(f'<div class="visually-hidden">Tableau des documents et vérifications: {verification_summary}</div>', unsafe_allow_html=True)

with tab2:
    # Section 3: Analyse des facteurs d'importance
    st.header("Facteurs influençant la décision")
    st.markdown('<div class="visually-hidden">Cette section analyse les facteurs qui ont le plus d\'impact sur la décision de crédit.</div>', unsafe_allow_html=True)
    
    # Vérifier si des valeurs d'importance sont disponibles
    if feature_importance:
        # Trier les features par importance absolue
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Limiter aux features les plus importantes
        top_features = sorted_features[:UI_CONFIG["max_features_display"]]
        
        # Préparer les données pour le graphique
        feature_names = []
        feature_values = []
        colors = []
        
        for feature, value in top_features:
            # Obtenir un nom convivial pour la feature à partir des descriptions
            display_name = FEATURE_DESCRIPTIONS.get(feature, feature)
            feature_names.append(display_name)
            feature_values.append(value)
            
            # Couleur basée sur l'impact (positif ou négatif)
            colors.append(
                COLORBLIND_FRIENDLY_PALETTE["positive"] if value < 0 else
                COLORBLIND_FRIENDLY_PALETTE["negative"]
            )
        
        # Créer le graphique d'importance des features avec accessibilité améliorée
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=feature_values,
            y=feature_names,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.3f}" for v in feature_values],
            textposition='auto',
            # Améliorer info-bulles pour l'accessibilité
            hovertemplate='<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Impact des caractéristiques sur la décision",
            xaxis_title="Impact sur la probabilité de défaut (valeurs SHAP)",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1
            ),
            font=dict(size=14),  # Augmenter taille de police pour lisibilité
            # Améliorer info-bulles
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            )
        )
        
        # Afficher le graphique
        st.plotly_chart(fig, use_container_width=True)
        
        # Description textuelle pour les lecteurs d'écran
        feature_impact_description = []
        for feature, value in top_features:
            display_name = FEATURE_DESCRIPTIONS.get(feature, feature)
            impact_type = "favorable (réduisant le risque)" if value < 0 else "défavorable (augmentant le risque)"
            feature_impact_description.append(f"{display_name} a un impact {impact_type} de {abs(value):.3f}")
        
        features_text = ". ".join(feature_impact_description)
        st.markdown(f'<div class="visually-hidden" aria-hidden="false">Graphique montrant l\'impact des caractéristiques sur la décision de crédit. {features_text}</div>', unsafe_allow_html=True)
        
        # Légende explicative avec contraste amélioré
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border: 1px solid #dee2e6;">
            <h4 style="margin-top: 0;">Comment interpréter ce graphique?</h4>
            <ul style="margin-bottom: 0;">
                <li><span style="color: #018571; font-weight: bold;">Les barres vertes</span> représentent des facteurs qui <strong>réduisent</strong> la probabilité de défaut (favorable au client).</li>
                <li><span style="color: #a6611a; font-weight: bold;">Les barres rouges</span> représentent des facteurs qui <strong>augmentent</strong> la probabilité de défaut (défavorable au client).</li>
                <li>Plus la barre est longue, plus l'impact du facteur est important.</li>
                <li>Les valeurs représentent la contribution de chaque caractéristique à la probabilité finale.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Table d'explication détaillée des facteurs
        st.subheader("Explication détaillée des facteurs principaux")
        
        explanations = []
        
        for feature, value in top_features:
            display_name = FEATURE_DESCRIPTIONS.get(feature, feature)
            direction = "favorable" if value < 0 else "défavorable"
            impact = "réduit" if value < 0 else "augmente"
            magnitude = "fortement" if abs(value) > 0.1 else "modérément" if abs(value) > 0.05 else "légèrement"
            
            # Obtenir la valeur réelle de la feature
            feature_value = None
            if feature in details.get('features', {}):
                feature_value = details['features'][feature]
            
            # Ajouter des icônes pour l'impact visuel (plus accessibilité)
            impact_icon = "✅ " if value < 0 else "❌ "
            impact_text = f"{impact_icon}{abs(value):.3f}"
            
            if feature_value is not None:
                explanation = f"La valeur de **{display_name}** est **{feature_value}**, ce qui {impact} {magnitude} le risque de défaut. Ce facteur est {direction} à la demande."
            else:
                explanation = f"Le facteur **{display_name}** {impact} {magnitude} le risque de défaut. Ce facteur est {direction} à la demande."
            
            explanations.append({
                "Facteur": display_name,
                "Impact": impact_text,
                "Explication": explanation
            })
        
        # Créer un DataFrame pour l'affichage
        explanations_df = pd.DataFrame(explanations)
        
        # Utiliser st.dataframe pour un affichage formaté
        st.dataframe(
            explanations_df,
            column_config={
                "Facteur": st.column_config.TextColumn("Facteur"),
                "Impact": st.column_config.TextColumn("Impact"),
                "Explication": st.column_config.TextColumn("Explication", width="large")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Résumé textuel pour les lecteurs d'écran
        explanations_summary = ". ".join([f"{row['Facteur']}: {row['Explication']}" for _, row in explanations_df.iterrows()])
        st.markdown(f'<div class="visually-hidden">Tableau d\'explications des facteurs principaux: {explanations_summary}</div>', unsafe_allow_html=True)
        
    else:
        # Message si les valeurs SHAP ne sont pas disponibles
        st.markdown('<div class="alert-warning" style="padding: 1rem; border-radius: 0.5rem;">', unsafe_allow_html=True)
        st.info("""
        Les valeurs d'importance des caractéristiques (SHAP) ne sont pas disponibles pour ce client.
        
        Motifs possibles:
        - L'API de calcul SHAP est temporairement indisponible
        - Le modèle n'a pas pu calculer les valeurs pour ce client spécifique
        - Le client possède des caractéristiques atypiques
        
        Vous pouvez toujours analyser les autres informations du profil client.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section d'analyse comparative des caractéristiques
    st.header("Analyse comparative des caractéristiques")
    
    # Sélection des caractéristiques à visualiser avec label plus descriptif
    selected_features = st.multiselect(
        label="Sélectionner des caractéristiques à comparer aux seuils:",
        options=list(details.get('features', {}).keys()),
        default=list(details.get('features', {}).keys())[:3],
        help="Sélectionnez une ou plusieurs caractéristiques pour voir comment les valeurs du client se comparent aux seuils favorable et défavorable."
    )
    
    if selected_features:
        # Préparation des données
        feature_data = []
        
        for feature in selected_features:
            if feature in details.get('features', {}):
                # Simuler des seuils (à remplacer par des données réelles)
                good_threshold = np.random.uniform(0.2, 0.8) * details['features'][feature]
                bad_threshold = np.random.uniform(1.2, 1.8) * details['features'][feature]
                
                feature_data.append({
                    "Caractéristique": FEATURE_DESCRIPTIONS.get(feature, feature),
                    "Valeur client": details['features'][feature],
                    "Seuil favorable": good_threshold,
                    "Seuil défavorable": bad_threshold
                })
        
        # Création du graphique avec améliorations d'accessibilité
        fig = go.Figure()
        
        for data in feature_data:
            fig.add_trace(go.Scatter(
                x=[data["Caractéristique"]],
                y=[data["Valeur client"]],
                mode='markers',
                name=data["Caractéristique"],
                marker=dict(size=12, color=COLORBLIND_FRIENDLY_PALETTE["primary"]),
                # Améliorer info-bulles
                hovertemplate='<b>%{x}</b><br>Valeur client: %{y:.2f}<extra></extra>'
            ))
            
            fig.add_shape(
                type="line",
                x0=data["Caractéristique"],
                y0=data["Seuil favorable"],
                x1=data["Caractéristique"],
                y1=data["Seuil défavorable"],
                line=dict(
                    color="gray",
                    width=2,
                )
            )
            
            # Marquer le seuil favorable
            fig.add_trace(go.Scatter(
                x=[data["Caractéristique"]],
                y=[data["Seuil favorable"]],
                mode='markers',
                marker=dict(
                    symbol="triangle-up",
                    size=10,
                    color=COLORBLIND_FRIENDLY_PALETTE["positive"]
                ),
                name="Seuil favorable",
                showlegend=False,
                # Améliorer info-bulles
                hovertemplate='<b>%{x}</b><br>Seuil favorable: %{y:.2f}<extra></extra>'
            ))
            
            # Marquer le seuil défavorable
            fig.add_trace(go.Scatter(
                x=[data["Caractéristique"]],
                y=[data["Seuil défavorable"]],
                mode='markers',
                marker=dict(
                    symbol="triangle-down",
                    size=10,
                    color=COLORBLIND_FRIENDLY_PALETTE["negative"]
                ),
                name="Seuil défavorable",
                showlegend=False,
                # Améliorer info-bulles
                hovertemplate='<b>%{x}</b><br>Seuil défavorable: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Positionnement du client par rapport aux seuils",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
            font=dict(size=14)  # Amélioration de la taille des polices
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Description textuelle pour les lecteurs d'écran
        comparison_description = []
        for data in feature_data:
            if data["Valeur client"] < data["Seuil favorable"]:
                position = "en dessous du seuil favorable"
                status = "très favorable"
            elif data["Valeur client"] > data["Seuil défavorable"]:
                position = "au-dessus du seuil défavorable"
                status = "défavorable"
            else:
                position = "entre les seuils favorable et défavorable"
                status = "acceptable"
            
            comparison_description.append(f"Pour {data['Caractéristique']}, la valeur du client ({data['Valeur client']:.2f}) est {position}, ce qui est {status}")
        
        comparison_text = ". ".join(comparison_description)
        st.markdown(f'<div class="visually-hidden" aria-hidden="false">Graphique montrant le positionnement du client par rapport aux seuils pour les caractéristiques sélectionnées: {", ".join([f["Caractéristique"] for f in feature_data])}. {comparison_text}.</div>', unsafe_allow_html=True)
        
        # Tableau des valeurs avec formatage amélioré
        st.dataframe(
            pd.DataFrame(feature_data),
            column_config={
                "Caractéristique": st.column_config.TextColumn("Caractéristique"),
                "Valeur client": st.column_config.NumberColumn("Valeur client", format="%.2f"),
                "Seuil favorable": st.column_config.NumberColumn("Seuil favorable", format="%.2f"),
                "Seuil défavorable": st.column_config.NumberColumn("Seuil défavorable", format="%.2f")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Résumé textuel pour les lecteurs d'écran
        feature_data_summary = ". ".join([f"{d['Caractéristique']}: valeur client {d['Valeur client']:.2f}, seuil favorable {d['Seuil favorable']:.2f}, seuil défavorable {d['Seuil défavorable']:.2f}" for d in feature_data])
        st.markdown(f'<div class="visually-hidden">Tableau des caractéristiques sélectionnées: {feature_data_summary}</div>', unsafe_allow_html=True)
        
    else:
        st.info("Veuillez sélectionner au moins une caractéristique pour l'analyse comparative.")

with tab3:
    # Section 4: Historique des décisions
    st.header("Historique des décisions pour ce client")
    st.markdown('<div class="visually-hidden">Cette section présente l\'historique des demandes de crédit précédentes du client.</div>', unsafe_allow_html=True)
    
    # Simuler un historique (à remplacer par des données réelles)
    decision_history = [
        {"Date": "2025-09-15", "Score": 0.559, "Décision": "REFUSÉ", "Montant": 360000, "Durée": 19},
        {"Date": "2024-11-22", "Score": 0.48, "Décision": "ACCEPTÉ", "Montant": 280000, "Durée": 24},
        {"Date": "2023-05-07", "Score": 0.52, "Décision": "REFUSÉ", "Montant": 400000, "Durée": 36}
    ]
    
    # Vérifier s'il y a un historique à afficher
    if decision_history:
        # Créer un DataFrame pour l'historique
        history_df = pd.DataFrame(decision_history)
        
        # Améliorer les décisions pour l'accessibilité (ajouter des icônes)
        history_df["Décision"] = history_df["Décision"].apply(
            lambda x: f"✅ {x}" if x == "ACCEPTÉ" else f"❌ {x}"
        )
        
        # Afficher le tableau avec style
        st.dataframe(
            history_df,
            column_config={
                "Date": st.column_config.DateColumn("Date"),
                "Score": st.column_config.NumberColumn("Score de risque", format="%.3f"),
                "Décision": st.column_config.TextColumn("Décision"),
                "Montant": st.column_config.NumberColumn(f"Montant ({UI_CONFIG['currency_symbol']})", format="%d"),
                "Durée": st.column_config.NumberColumn("Durée (mois)")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Résumé textuel pour les lecteurs d'écran
        history_summary = ". ".join([f"Date: {row['Date']}, Décision: {row['Décision']}, Score: {row['Score']:.3f}, Montant: {row['Montant']} {UI_CONFIG['currency_symbol']}, Durée: {row['Durée']} mois" for _, row in history_df.iterrows()])
        st.markdown(f'<div class="visually-hidden">Tableau des décisions historiques: {history_summary}</div>', unsafe_allow_html=True)
        
        # Graphique d'évolution des scores avec améliorations d'accessibilité
        fig = px.line(
            history_df,
            x="Date",
            y="Score",
            markers=True,
            title="Évolution du score de risque au fil du temps",
            color_discrete_sequence=[COLORBLIND_FRIENDLY_PALETTE["primary"]]
        )
        
        # Ajouter la ligne de seuil
        fig.add_hline(
            y=DEFAULT_THRESHOLD,
            line_dash="dash",
            line_color=COLORBLIND_FRIENDLY_PALETTE["threshold"],
            annotation_text=f"Seuil ({DEFAULT_THRESHOLD:.2f})",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Score de risque",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(size=14),  # Amélioration de la taille des polices
            # Améliorer info-bulles
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Description textuelle du graphique pour les lecteurs d'écran
        score_trend = "augmente" if history_df["Score"].iloc[-1] > history_df["Score"].iloc[0] else "diminue"
        st.markdown(f'<div class="visually-hidden" aria-hidden="false">Graphique d\'évolution du score de risque au fil du temps. La tendance générale du score {score_trend}. Le seuil de décision est fixé à {DEFAULT_THRESHOLD:.2f}. Les scores inférieurs au seuil correspondent à des décisions favorables.</div>', unsafe_allow_html=True)
        
    else:
        st.info("Aucun historique de décision disponible pour ce client.")

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
        st.markdown('<div class="alert-success" style="padding: 0.75rem; border-radius: 0.5rem; margin-top: 1rem;">', unsafe_allow_html=True)
        st.success("Notes enregistrées")
        st.markdown('</div>', unsafe_allow_html=True)
        # Pour les lecteurs d'écran
        st.markdown('<div class="visually-hidden" aria-live="polite">Vos notes ont été enregistrées avec succès.</div>', unsafe_allow_html=True)

with col_notes2:
    # Actions possibles
    with st.container(border=True):
        st.subheader("Actions rapides")
        
        # Rendre les boutons plus accessibles
        if st.button("📧 Envoyer un récapitulatif", 
                     help="Envoie un résumé de cette analyse au client par email",
                     use_container_width=True):
            st.info("Fonctionnalité d'envoi d'email à implémenter.")
            st.markdown('<div class="visually-hidden" aria-live="polite">La fonctionnalité d\'envoi de récapitulatif par email sera implémentée prochainement.</div>', unsafe_allow_html=True)
            
        if decision == "REFUSÉ" and st.button("📝 Demander une révision", 
                                              help="Demande une nouvelle évaluation du dossier",
                                              use_container_width=True):
            st.info("Redirection vers le formulaire de révision.")
            st.markdown('<div class="visually-hidden" aria-live="polite">Vous serez redirigé vers le formulaire de révision.</div>', unsafe_allow_html=True)
            
        if st.button("🔙 Retour à l'accueil", 
                     help="Retourner à la page d'accueil",
                     use_container_width=True):
            st.switch_page("Home.py")

# Footer avec informations de version (amélioré pour l'accessibilité)
st.markdown("""
<hr aria-hidden="true">
<div style="text-align: center; color: #333333; background-color: #f8f9fa; padding: 0.75rem; border-radius: 0.5rem; margin-top: 1rem;">
    <div>
        <strong>Profil client détaillé</strong> | Dernière mise à jour: 2025-10-17 07:38:53
    </div>
    <div>
        <span>Montants exprimés en roubles (₽)</span> | 
        <span>Contact support: <a href="tel:+XXXXXXXXXX" style="color: #0066cc;">poste 4242</a></span>
    </div>
</div>
""", unsafe_allow_html=True)