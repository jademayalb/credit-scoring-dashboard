"""
Page de simulation
Permet de simuler l'impact des modifications de caractéristiques sur la décision de crédit
et de prédire la décision pour un nouveau dossier client
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from utils.api_client import get_client_prediction, get_client_details, get_available_clients, get_feature_importance

# Import de la configuration
from config import (
    COLORBLIND_FRIENDLY_PALETTE, UI_CONFIG, 
    FEATURE_DESCRIPTIONS, DEFAULT_THRESHOLD
)

# Configuration de la page
st.set_page_config(
    page_title="Simulation de Décisions - Dashboard de Scoring Crédit",
    page_icon="🔄",
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
    
    /* Style pour les cartes */
    .info-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    
    /* Style pour les sections importantes */
    .important-section {
        border-left: 4px solid #3366ff;
        padding-left: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Style pour les alertes */
    .custom-alert {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    .custom-alert.info {
        background-color: #e6f3ff;
        border: 1px solid #b8daff;
        color: #004085;
    }
    
    .custom-alert.warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    
    .custom-alert.success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
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
active_tab_index = 3  # Pour la page Simulation

# Gestion de la navigation
if selected_tab[0].button("Accueil", key="nav_home", use_container_width=True):
    st.switch_page("Home.py")
elif selected_tab[1].button("Profil Client", key="nav_profile", use_container_width=True):
    st.switch_page("pages/1_Profil_Client.py")
elif selected_tab[2].button("Comparaison", key="nav_compare", use_container_width=True):
    st.switch_page("pages/2_Comparaison.py")

# Titre de la page
st.title("Simulation de décisions de crédit")
st.markdown("""
Cette page vous permet de simuler l'impact de modifications sur un dossier existant ou de prédire la décision pour un nouveau client.
Utilisez les onglets ci-dessous pour choisir le mode de simulation souhaité.
""")

# Choix du mode de simulation
simulation_mode = st.tabs(["Modification d'un client existant", "Nouveau dossier client"])

# Mode 1: Modification d'un client existant
with simulation_mode[0]:
    st.header("Simulation de modifications pour un client existant")
    st.markdown("""
    <div class="custom-alert info">
        <strong>💡 Comment utiliser cette fonctionnalité:</strong> Sélectionnez un client, modifiez une ou plusieurs caractéristiques et lancez la simulation pour voir l'impact sur la décision de crédit.
    </div>
    """, unsafe_allow_html=True)
    
    # Sélection du client
    col_select, col_info = st.columns([2, 1])
    
    with col_select:
        # Chargement de la liste des clients disponibles
        with st.spinner("Chargement de la liste des clients..."):
            available_clients = get_available_clients(limit=UI_CONFIG["default_limit"])
        
        if not available_clients:
            st.error("Impossible de charger la liste des clients.")
            st.stop()
        
        # Sélection du client à simuler
        selected_client = st.selectbox(
            "Sélectionnez un client pour la simulation:",
            options=available_clients,
            key="simulation_client_select"
        )
    
    with col_info:
        st.markdown("""
        <div class="important-section">
            <small>Date de simulation: {}</small><br>
            <small>Cette simulation est à but informatif uniquement et ne constitue pas une décision finale.</small>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
    
    # Chargement des données du client sélectionné
    with st.spinner("Chargement des données du client..."):
        client_details = get_client_details(selected_client)
        client_prediction = get_client_prediction(selected_client)
        feature_importance = get_feature_importance(selected_client)
    
    if not client_details or not client_prediction:
        st.error(f"Impossible de charger les données pour le client {selected_client}.")
        st.stop()
    
    # Affichage des informations actuelles du client
    col_current1, col_current2 = st.columns([3, 2])
    
    with col_current1:
        st.subheader("Informations actuelles du client")
        
        # Informations personnelles
        st.markdown("<strong>Informations personnelles:</strong>", unsafe_allow_html=True)
        personal_info = client_details.get("personal_info", {})
        
        col_personal1, col_personal2 = st.columns(2)
        
        with col_personal1:
            st.markdown(f"• Genre: {personal_info.get('gender', 'N/A')}")
            st.markdown(f"• Âge: {personal_info.get('age', 'N/A')} ans")
            st.markdown(f"• Éducation: {personal_info.get('education', 'N/A')}")
            
        with col_personal2:
            st.markdown(f"• Statut familial: {personal_info.get('family_status', 'N/A')}")
            st.markdown(f"• Revenu annuel: {personal_info.get('income', 0):,.0f} {UI_CONFIG['currency_symbol']}")
            st.markdown(f"• Ancienneté d'emploi: {personal_info.get('employment_years', 0)} ans")
        
        # Informations crédit
        st.markdown("<strong>Informations de crédit:</strong>", unsafe_allow_html=True)
        credit_info = client_details.get("credit_info", {})
        
        col_credit1, col_credit2 = st.columns(2)
        
        with col_credit1:
            st.markdown(f"• Montant demandé: {credit_info.get('amount', 0):,.0f} {UI_CONFIG['currency_symbol']}")
            st.markdown(f"• Durée du crédit: {credit_info.get('credit_term', 0)} mois")
            
        with col_credit2:
            st.markdown(f"• Mensualité: {credit_info.get('annuity', 0):,.0f} {UI_CONFIG['currency_symbol']}/mois")
            st.markdown(f"• Valeur du bien: {credit_info.get('goods_price', 0):,.0f} {UI_CONFIG['currency_symbol']}")
    
    with col_current2:
        # Affichage de la décision actuelle
        decision = client_prediction.get("decision", "INCONNU")
        probability = client_prediction.get("probability", 0)
        threshold = client_prediction.get("threshold", DEFAULT_THRESHOLD)
        
        status_color = COLORBLIND_FRIENDLY_PALETTE['accepted'] if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE['refused']
        status_icon = "✅" if decision == "ACCEPTÉ" else "❌"
        
        st.markdown(
            f"""
            <div style="padding: 1rem; border-radius: 0.5rem; background-color: {status_color}22; border: 2px solid {status_color}; margin-bottom: 1.5rem;">
                <h3 style="margin: 0; color: {status_color};">
                    <span aria-hidden="true">{status_icon}</span> 
                    Décision actuelle: <strong>{decision}</strong>
                </h3>
                <p style="margin: 0.5rem 0 0 0;">Probabilité de défaut: <strong>{probability:.1%}</strong> (Seuil: {threshold:.1%})</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Facteurs importants pour ce client
        if feature_importance:
            st.markdown("<strong>Facteurs les plus influents:</strong>", unsafe_allow_html=True)
            
            # Obtenir les 3 facteurs les plus importants
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            
            for feature, value in sorted_features:
                # Formater le nom de la feature
                if feature in FEATURE_DESCRIPTIONS:
                    display_name = FEATURE_DESCRIPTIONS.get(feature)
                elif feature == "EXT_SOURCE_1":
                    display_name = "Score normalisé - Source externe 1"
                elif feature == "EXT_SOURCE_2":
                    display_name = "Score normalisé - Source externe 2"
                elif feature == "EXT_SOURCE_3":
                    display_name = "Score normalisé - Source externe 3"
                else:
                    display_name = feature
                
                # Déterminer l'impact (positif ou négatif)
                impact = "positif" if value < 0 else "négatif"
                impact_color = "#018571" if value < 0 else "#a6611a"
                
                st.markdown(f"• <span style='color: {impact_color};'>{display_name}</span> (impact {impact})", unsafe_allow_html=True)
    
    # Section de modification des caractéristiques
    st.subheader("Modification des caractéristiques")
    st.markdown("Sélectionnez les caractéristiques que vous souhaitez modifier et ajustez leurs valeurs:")
    
    # Déterminer les caractéristiques les plus importantes à proposer
    important_features = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED", 
                         "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
    
    features_data = client_details.get("features", {})
    modifiable_features = [f for f in important_features if f in features_data]
    
    # Sélection des caractéristiques à modifier
    selected_features = st.multiselect(
        "Sélectionnez jusqu'à 3 caractéristiques à modifier:",
        options=modifiable_features,
        format_func=lambda f: FEATURE_DESCRIPTIONS.get(f) if f in FEATURE_DESCRIPTIONS else (
            "Score normalisé - Source externe 1" if f == "EXT_SOURCE_1" else
            "Score normalisé - Source externe 2" if f == "EXT_SOURCE_2" else
            "Score normalisé - Source externe 3" if f == "EXT_SOURCE_3" else f
        ),
        max_selections=3
    )
    
    # Interface pour modifier les valeurs sélectionnées
    if selected_features:
        st.markdown("<strong>Nouvelles valeurs:</strong>", unsafe_allow_html=True)
        
        modified_values = {}
        col_mod1, col_mod2 = st.columns(2)
        
        # Distribuer les contrôles en colonnes
        features_per_column = len(selected_features) // 2 + len(selected_features) % 2
        
        with col_mod1:
            for i, feature in enumerate(selected_features[:features_per_column]):
                current_value = features_data.get(feature, 0)
                
                # Formater le nom d'affichage
                if feature in FEATURE_DESCRIPTIONS:
                    display_name = FEATURE_DESCRIPTIONS.get(feature)
                elif feature == "EXT_SOURCE_1":
                    display_name = "Score normalisé - Source externe 1"
                elif feature == "EXT_SOURCE_2":
                    display_name = "Score normalisé - Source externe 2"
                elif feature == "EXT_SOURCE_3":
                    display_name = "Score normalisé - Source externe 3"
                else:
                    display_name = feature
                
                # Créer le contrôle approprié selon le type de feature
                if feature.startswith("EXT_SOURCE"):
                    # Scores externes (entre 0 et 1)
                    modified_values[feature] = st.slider(
                        f"{display_name} [actuelle: {current_value:.2f}]",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(current_value),
                        step=0.01
                    )
                elif feature == "DAYS_BIRTH":
                    # Âge (jours négatifs convertis en années)
                    age_years = abs(current_value) / 365.25
                    modified_values[feature] = -st.slider(
                        f"Âge (ans) [actuel: {age_years:.0f} ans]",
                        min_value=18.0,
                        max_value=100.0,
                        value=float(age_years),
                        step=1.0
                    ) * 365.25
                elif feature == "DAYS_EMPLOYED":
                    # Ancienneté d'emploi (jours négatifs convertis en années)
                    if current_value == 365243:  # Code spécial pour "sans emploi"
                        emp_status = st.selectbox(
                            "Statut d'emploi [actuel: Sans emploi]",
                            options=["Sans emploi", "Employé"]
                        )
                        if emp_status == "Sans emploi":
                            modified_values[feature] = 365243
                        else:
                            # Si maintenant employé, demander l'ancienneté
                            emp_years = st.slider(
                                "Ancienneté d'emploi (ans)",
                                min_value=0.0,
                                max_value=50.0,
                                value=1.0,
                                step=1.0
                            )
                            modified_values[feature] = -emp_years * 365.25
                    else:
                        emp_years = abs(current_value) / 365.25
                        modified_values[feature] = -st.slider(
                            f"Ancienneté d'emploi (ans) [actuelle: {emp_years:.0f} ans]",
                            min_value=0.0,
                            max_value=50.0,
                            value=float(emp_years),
                            step=1.0
                        ) * 365.25
                elif feature == "AMT_INCOME_TOTAL":
                    # Revenu (valeur positive)
                    modified_values[feature] = st.number_input(
                        f"Revenu annuel ({UI_CONFIG['currency_symbol']}) [actuel: {current_value:,.0f}]",
                        min_value=0.0,
                        max_value=float(current_value * 10) if current_value > 0 else 10000000.0,
                        value=float(current_value),
                        step=10000.0,
                        format="%.0f"
                    )
                elif feature in ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
                    # Montants de crédit (valeur positive)
                    unit = "/mois" if feature == "AMT_ANNUITY" else ""
                    feature_name = {
                        "AMT_CREDIT": "Montant du crédit",
                        "AMT_ANNUITY": "Mensualité",
                        "AMT_GOODS_PRICE": "Valeur du bien"
                    }.get(feature, feature)
                    
                    modified_values[feature] = st.number_input(
                        f"{feature_name} ({UI_CONFIG['currency_symbol']}{unit}) [actuel: {current_value:,.0f}]",
                        min_value=0.0,
                        max_value=float(current_value * 5) if current_value > 0 else 100000000.0,
                        value=float(current_value),
                        step=5000.0,
                        format="%.0f"
                    )
                else:
                    # Autres caractéristiques numériques
                    modified_values[feature] = st.number_input(
                        f"{display_name} [actuelle: {current_value}]",
                        value=float(current_value),
                        step=0.01
                    )
        
        with col_mod2:
            for i, feature in enumerate(selected_features[features_per_column:]):
                current_value = features_data.get(feature, 0)
                
                # Formater le nom d'affichage
                if feature in FEATURE_DESCRIPTIONS:
                    display_name = FEATURE_DESCRIPTIONS.get(feature)
                elif feature == "EXT_SOURCE_1":
                    display_name = "Score normalisé - Source externe 1"
                elif feature == "EXT_SOURCE_2":
                    display_name = "Score normalisé - Source externe 2"
                elif feature == "EXT_SOURCE_3":
                    display_name = "Score normalisé - Source externe 3"
                else:
                    display_name = feature
                
                # Créer le contrôle approprié selon le type de feature
                if feature.startswith("EXT_SOURCE"):
                    # Scores externes (entre 0 et 1)
                    modified_values[feature] = st.slider(
                        f"{display_name} [actuelle: {current_value:.2f}]",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(current_value),
                        step=0.01
                    )
                elif feature == "DAYS_BIRTH":
                    # Âge (jours négatifs convertis en années)
                    age_years = abs(current_value) / 365.25
                    modified_values[feature] = -st.slider(
                        f"Âge (ans) [actuel: {age_years:.0f} ans]",
                        min_value=18.0,
                        max_value=100.0,
                        value=float(age_years),
                        step=1.0
                    ) * 365.25
                elif feature == "DAYS_EMPLOYED":
                    # Ancienneté d'emploi (jours négatifs convertis en années)
                    if current_value == 365243:  # Code spécial pour "sans emploi"
                        emp_status = st.selectbox(
                            "Statut d'emploi [actuel: Sans emploi]",
                            options=["Sans emploi", "Employé"]
                        )
                        if emp_status == "Sans emploi":
                            modified_values[feature] = 365243
                        else:
                            # Si maintenant employé, demander l'ancienneté
                            emp_years = st.slider(
                                "Ancienneté d'emploi (ans)",
                                min_value=0.0,
                                max_value=50.0,
                                value=1.0,
                                step=1.0
                            )
                            modified_values[feature] = -emp_years * 365.25
                    else:
                        emp_years = abs(current_value) / 365.25
                        modified_values[feature] = -st.slider(
                            f"Ancienneté d'emploi (ans) [actuelle: {emp_years:.0f} ans]",
                            min_value=0.0,
                            max_value=50.0,
                            value=float(emp_years),
                            step=1.0
                        ) * 365.25
                elif feature == "AMT_INCOME_TOTAL":
                    # Revenu (valeur positive)
                    modified_values[feature] = st.number_input(
                        f"Revenu annuel ({UI_CONFIG['currency_symbol']}) [actuel: {current_value:,.0f}]",
                        min_value=0.0,
                        max_value=float(current_value * 10) if current_value > 0 else 10000000.0,
                        value=float(current_value),
                        step=10000.0,
                        format="%.0f"
                    )
                elif feature in ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
                    # Montants de crédit (valeur positive)
                    unit = "/mois" if feature == "AMT_ANNUITY" else ""
                    feature_name = {
                        "AMT_CREDIT": "Montant du crédit",
                        "AMT_ANNUITY": "Mensualité",
                        "AMT_GOODS_PRICE": "Valeur du bien"
                    }.get(feature, feature)
                    
                    modified_values[feature] = st.number_input(
                        f"{feature_name} ({UI_CONFIG['currency_symbol']}{unit}) [actuel: {current_value:,.0f}]",
                        min_value=0.0,
                        max_value=float(current_value * 5) if current_value > 0 else 100000000.0,
                        value=float(current_value),
                        step=5000.0,
                        format="%.0f"
                    )
                else:
                    # Autres caractéristiques numériques
                    modified_values[feature] = st.number_input(
                        f"{display_name} [actuelle: {current_value}]",
                        value=float(current_value),
                        step=0.01
                    )
        
        # Bouton pour lancer la simulation
        if st.button("Simuler l'impact des modifications", type="primary"):
            # Dans une application réelle, ici vous appelleriez votre API de prédiction
            # avec les valeurs modifiées. Pour ce prototype, nous allons simuler une réponse.
            
            with st.spinner("Calcul de l'impact des modifications..."):
                # Simuler un délai pour donner l'impression que le calcul est en cours
                import time
                time.sleep(1.5)
                
                # Fonction de simulation de l'impact des modifications
                def simulate_impact(original_prob, modified_values, features_data, feature_importance):
                    """Simule l'impact des modifications sur la probabilité de défaut."""
                    
                    impact = 0.0
                    impact_details = {}
                    
                    # Règles d'impact approximatives basées sur le type de feature
                    for feature, new_value in modified_values.items():
                        old_value = features_data.get(feature, 0)
                        feature_imp = feature_importance.get(feature, 0) if feature_importance else 0
                        
                        # Pas de modification = pas d'impact
                        if new_value == old_value:
                            impact_details[feature] = 0
                            continue
                        
                        # Impact basé sur l'importance de la feature et la direction du changement
                        if feature.startswith("EXT_SOURCE"):
                            # Les sources externes sont inversement corrélées au risque
                            # (plus le score est élevé, plus le risque est faible)
                            delta = new_value - old_value
                            feature_impact = -delta * 0.15  # Simuler un impact significatif
                            impact += feature_impact
                            impact_details[feature] = feature_impact
                            
                        elif feature == "DAYS_BIRTH":
                            # Convertir en âge en années (les jours sont négatifs)
                            old_age = abs(old_value) / 365.25
                            new_age = abs(new_value) / 365.25
                            delta_age = new_age - old_age
                            
                            # Les clients plus âgés sont généralement moins risqués
                            feature_impact = -delta_age * 0.003
                            impact += feature_impact
                            impact_details[feature] = feature_impact
                            
                        elif feature == "DAYS_EMPLOYED":
                            # Traitement spécial pour le code "sans emploi"
                            if old_value == 365243 and new_value != 365243:
                                # Passage de sans emploi à employé (réduction du risque)
                                feature_impact = -0.05
                                impact += feature_impact
                                impact_details[feature] = feature_impact
                            elif old_value != 365243 and new_value == 365243:
                                # Passage d'employé à sans emploi (augmentation du risque)
                                feature_impact = 0.05
                                impact += feature_impact
                                impact_details[feature] = feature_impact
                            elif old_value != 365243 and new_value != 365243:
                                # Changement dans l'ancienneté
                                old_years = abs(old_value) / 365.25
                                new_years = abs(new_value) / 365.25
                                delta_years = new_years - old_years
                                
                                # Plus d'ancienneté = moins de risque
                                feature_impact = -delta_years * 0.005
                                impact += feature_impact
                                impact_details[feature] = feature_impact
                            
                        elif feature == "AMT_INCOME_TOTAL":
                            # Variation relative du revenu
                            if old_value > 0:
                                pct_change = (new_value - old_value) / old_value
                                
                                # Plus de revenu = moins de risque
                                feature_impact = -pct_change * 0.1
                                impact += feature_impact
                                impact_details[feature] = feature_impact
                                
                        elif feature == "AMT_CREDIT":
                            # Variation relative du montant de crédit
                            if old_value > 0:
                                pct_change = (new_value - old_value) / old_value
                                
                                # Plus de crédit = plus de risque
                                feature_impact = pct_change * 0.05
                                impact += feature_impact
                                impact_details[feature] = feature_impact
                                
                        elif feature == "AMT_ANNUITY":
                            # Variation relative de la mensualité
                            if old_value > 0:
                                pct_change = (new_value - old_value) / old_value
                                
                                # Mensualité plus élevée = plus de risque
                                feature_impact = pct_change * 0.08
                                impact += feature_impact
                                impact_details[feature] = feature_impact
                                
                        elif feature == "AMT_GOODS_PRICE":
                            # Variation relative de la valeur du bien
                            if old_value > 0:
                                pct_change = (new_value - old_value) / old_value
                                
                                # Bien plus cher = impact mitigé (dépend du ratio crédit/valeur)
                                feature_impact = pct_change * 0.02
                                impact += feature_impact
                                impact_details[feature] = feature_impact
                                
                        else:
                            # Pour les autres features, simuler un impact minimal
                            feature_impact = feature_imp * 0.1
                            impact += feature_impact
                            impact_details[feature] = feature_impact
                    
                    # Calculer la nouvelle probabilité
                    new_prob = max(0.01, min(0.99, original_prob + impact))
                    
                    return new_prob, impact_details
                
                # Simuler l'impact des modifications
                original_prob = client_prediction.get("probability", 0)
                new_prob, impact_details = simulate_impact(
                    original_prob, 
                    modified_values, 
                    features_data, 
                    feature_importance
                )
                
                # Déterminer la nouvelle décision
                threshold = client_prediction.get("threshold", DEFAULT_THRESHOLD)
                new_decision = "ACCEPTÉ" if new_prob < threshold else "REFUSÉ"
                original_decision = client_prediction.get("decision", "INCONNU")
                
                # Afficher les résultats de la simulation
                st.subheader("Résultat de la simulation")
                
                col_result1, col_result2 = st.columns([1, 1])
                
                with col_result1:
                    # Statut original
                    original_color = COLORBLIND_FRIENDLY_PALETTE['accepted'] if original_decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE['refused']
                    original_icon = "✅" if original_decision == "ACCEPTÉ" else "❌"
                    
                    st.markdown(
                        f"""
                        <div style="padding: 1rem; border-radius: 0.5rem; background-color: {original_color}22; border: 2px solid {original_color}; margin-bottom: 1rem;">
                            <h4 style="margin: 0; color: {original_color};">
                                <span aria-hidden="true">{original_icon}</span> 
                                Décision originale: <strong>{original_decision}</strong>
                            </h4>
                            <p style="margin: 0.5rem 0 0 0;">Probabilité: <strong>{original_prob:.1%}</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with col_result2:
                    # Nouvelle décision simulée
                    new_color = COLORBLIND_FRIENDLY_PALETTE['accepted'] if new_decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE['refused']
                    new_icon = "✅" if new_decision == "ACCEPTÉ" else "❌"
                    
                    st.markdown(
                        f"""
                        <div style="padding: 1rem; border-radius: 0.5rem; background-color: {new_color}22; border: 2px solid {new_color}; margin-bottom: 1rem;">
                            <h4 style="margin: 0; color: {new_color};">
                                <span aria-hidden="true">{new_icon}</span> 
                                Décision simulée: <strong>{new_decision}</strong>
                            </h4>
                            <p style="margin: 0.5rem 0 0 0;">Probabilité: <strong>{new_prob:.1%}</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Graphique de comparaison
                st.subheader("Comparaison des probabilités")
                
                # Création d'une jauge avant/après
                fig = go.Figure()
                
                # Ajouter des zones de risque
                fig.add_shape(
                    type="rect",
                    x0=0, x1=threshold,
                    y0=0, y1=1,
                    fillcolor=COLORBLIND_FRIENDLY_PALETTE['accepted'] + "44",
                    line=dict(width=0),
                    layer="below"
                )
                fig.add_shape(
                    type="rect",
                    x0=threshold, x1=1,
                    y0=0, y1=1,
                    fillcolor=COLORBLIND_FRIENDLY_PALETTE['refused'] + "44",
                    line=dict(width=0),
                    layer="below"
                )
                
                # Ajouter une ligne pour le seuil
                fig.add_shape(
                    type="line",
                    x0=threshold, x1=threshold,
                    y0=0, y1=1,
                    line=dict(
                        color="black",
                        width=2,
                        dash="dash",
                    )
                )
                
                # Ajouter des barres pour les probabilités
                fig.add_trace(go.Bar(
                    x=[original_prob, new_prob],
                    y=["Avant", "Après"],
                    orientation='h',
                    marker_color=[original_color, new_color],
                    text=[f"{original_prob:.1%}", f"{new_prob:.1%}"],
                    textposition='outside',
                    hoverinfo='text',
                    showlegend=False
                ))
                
                # Mettre en forme le graphique
                fig.update_layout(
                    title="Impact des modifications sur la probabilité de défaut",
                    xaxis=dict(
                        title="Probabilité de défaut",
                        range=[0, 1],
                        tickformat='.0%',
                        tickvals=[0, 0.2, 0.4, threshold, 0.6, 0.8, 1],
                        ticktext=['0%', '20%', '40%', f'{threshold:.0%}', '60%', '80%', '100%']
                    ),
                    yaxis=dict(
                        title=""
                    ),
                    height=300,
                    margin=dict(l=0, r=0, t=50, b=0),
                    annotations=[
                        dict(
                            x=threshold,
                            y=1.1,
                            xref="x",
                            yref="paper",
                            text=f"Seuil: {threshold:.0%}",
                            showarrow=False,
                            font=dict(
                                size=14,
                                color="black"
                            ),
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1,
                            borderpad=3
                        )
                    ]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyse détaillée de l'impact
                st.subheader("Analyse détaillée des modifications")
                
                # Générer un message de synthèse
                if new_decision != original_decision:
                    if new_decision == "ACCEPTÉ":
                        st.markdown(
                            f"""
                            <div class="custom-alert success">
                                <strong>Changement positif :</strong> Les modifications simulées ont fait passer la décision de <strong>refusé</strong> à <strong>accepté</strong>.
                                La probabilité de défaut a diminué de <strong>{original_prob:.1%}</strong> à <strong>{new_prob:.1%}</strong>, passant sous le seuil de {threshold:.0%}.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="custom-alert warning">
                                <strong>Changement négatif :</strong> Les modifications simulées ont fait passer la décision de <strong>accepté</strong> à <strong>refusé</strong>.
                                La probabilité de défaut a augmenté de <strong>{original_prob:.1%}</strong> à <strong>{new_prob:.1%}</strong>, dépassant le seuil de {threshold:.0%}.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    delta = new_prob - original_prob
                    if abs(delta) < 0.01:
                        st.markdown(
                            f"""
                            <div class="custom-alert info">
                                <strong>Impact négligeable :</strong> Les modifications n'ont pas significativement changé la probabilité de défaut ni la décision finale.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    elif delta > 0:
                        st.markdown(
                            f"""
                            <div class="custom-alert warning">
                                <strong>Impact négatif :</strong> Les modifications ont augmenté la probabilité de défaut de <strong>{original_prob:.1%}</strong> à <strong>{new_prob:.1%}</strong>,
                                mais la décision finale reste la même ({original_decision}).
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="custom-alert success">
                                <strong>Impact positif :</strong> Les modifications ont diminué la probabilité de défaut de <strong>{original_prob:.1%}</strong> à <strong>{new_prob:.1%}</strong>,
                                mais la décision finale reste la même ({original_decision}).
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                # Tableau d'impact des modifications
                impact_data = []
                
                for feature, impact_value in impact_details.items():
                    # Formater le nom de la feature
                    if feature in FEATURE_DESCRIPTIONS:
                        display_name = FEATURE_DESCRIPTIONS.get(feature)
                    elif feature == "EXT_SOURCE_1":
                        display_name = "Score normalisé - Source externe 1"
                    elif feature == "EXT_SOURCE_2":
                        display_name = "Score normalisé - Source externe 2"
                    elif feature == "EXT_SOURCE_3":
                        display_name = "Score normalisé - Source externe 3"
                    else:
                        display_name = feature
                    
                    # Formater les valeurs pour l'affichage
                    old_value = features_data.get(feature, 0)
                    new_value = modified_values.get(feature, old_value)
                    
                    if feature == "DAYS_BIRTH":
                        old_display = f"{abs(old_value / 365.25):.0f} ans"
                        new_display = f"{abs(new_value / 365.25):.0f} ans"
                    elif feature == "DAYS_EMPLOYED":
                        if old_value == 365243:
                            old_display = "Sans emploi"
                        else:
                            old_display = f"{abs(old_value / 365.25):.0f} ans"
                            
                        if new_value == 365243:
                            new_display = "Sans emploi"
                        else:
                            new_display = f"{abs(new_value / 365.25):.0f} ans"
                    elif feature in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
                        old_display = f"{old_value:,.0f} {UI_CONFIG['currency_symbol']}"
                        new_display = f"{new_value:,.0f} {UI_CONFIG['currency_symbol']}"
                    elif feature.startswith("EXT_SOURCE"):
                        old_display = f"{old_value:.2f}"
                        new_display = f"{new_value:.2f}"
                    else:
                        old_display = f"{old_value}"
                        new_display = f"{new_value}"
                    
                    # Déterminer la direction de l'impact
                    if abs(impact_value) < 0.001:
                        impact_text = "Aucun impact"
                        impact_class = ""
                    elif impact_value > 0:
                        if impact_value > 0.02:
                            impact_text = "Augmentation significative du risque"
                        else:
                            impact_text = "Légère augmentation du risque"
                        impact_class = "text-danger"
                    else:
                        if impact_value < -0.02:
                            impact_text = "Réduction significative du risque"
                        else:
                            impact_text = "Légère réduction du risque"
                        impact_class = "text-success"
                    
                    impact_data.append({
                        "Caractéristique": display_name,
                        "Ancienne valeur": old_display,
                        "Nouvelle valeur": new_display,
                        "Impact": impact_text
                    })
                
                # Afficher le tableau d'impact
                if impact_data:
                    impact_df = pd.DataFrame(impact_data)
                    st.dataframe(impact_df, use_container_width=True)
                
                # Note d'avertissement
                st.info("""
                **Note importante:** Cette simulation est basée sur des règles simplifiées et ne reflète pas exactement le comportement du modèle réel.
                Dans une version complète du système, cette simulation appellerait l'API pour recalculer la prédiction.
                """)
    else:
        st.info("👆 Veuillez sélectionner au moins une caractéristique à modifier pour simuler son impact.")

# Mode 2: Nouveau dossier client
with simulation_mode[1]:
    st.header("Prédiction pour un nouveau dossier client")
    st.markdown("""
    <div class="custom-alert info">
        <strong>📌 Fonctionnalité en développement :</strong> Cette partie de l'application sera disponible dans une prochaine version.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    La prédiction pour un nouveau dossier client permettra de :
    
    - Saisir les informations personnelles et financières d'un client potentiel
    - Obtenir une prédiction sur sa probabilité de défaut
    - Recevoir des recommandations sur les ajustements à apporter au dossier pour améliorer ses chances d'acceptation
    
    Cette fonctionnalité nécessite une intégration spécifique avec l'API de prédiction pour traiter de nouveaux clients sans historique dans le système.
    """)
    
    # Mockup d'interface future
    st.markdown("""
    <div style="border: 1px dashed #aaa; padding: 1rem; margin: 1rem 0; border-radius: 0.5rem;">
        <h4 style="color: #888; margin-top: 0;">Aperçu de l'interface à venir</h4>
        <div style="opacity: 0.6;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                <div style="width: 48%; background: #f8f9fa; padding: 0.5rem; border-radius: 0.25rem;">
                    <div style="font-weight: bold; margin-bottom: 0.5rem;">Informations personnelles</div>
                    <div style="height: 150px; background: #eee; border-radius: 0.25rem;"></div>
                </div>
                <div style="width: 48%; background: #f8f9fa; padding: 0.5rem; border-radius: 0.25rem;">
                    <div style="font-weight: bold; margin-bottom: 0.5rem;">Informations financières</div>
                    <div style="height: 150px; background: #eee; border-radius: 0.25rem;"></div>
                </div>
            </div>
            <div style="width: 100%; background: #f8f9fa; padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 1rem;">
                <div style="font-weight: bold; margin-bottom: 0.5rem;">Scores externes</div>
                <div style="height: 50px; background: #eee; border-radius: 0.25rem;"></div>
            </div>
            <div style="text-align: center;">
                <div style="display: inline-block; padding: 0.5rem 2rem; background: #ddd; border-radius: 0.25rem; font-weight: bold;">Prédire la décision</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        <strong>Simulation de décisions</strong> | Prêt à dépenser
    </div>
    <div>
        <span>Montants exprimés en roubles (₽)</span> | 
        <span>Contact support: poste 4242</span>
    </div>
</div>
""", unsafe_allow_html=True)
