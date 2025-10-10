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
    page_title="Profil Client - Dashboard de Scoring Crédit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour afficher la barre de navigation commune
def display_navigation():
    st.markdown(
        """
        <style>
        .nav-button {
            display: inline-block;
            padding: 5px 15px;
            margin-right: 10px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: 500;
            background-color: #f0f0f0;
            color: #404040;
        }
        .nav-button.active {
            background-color: #3366ff;
            color: white;
        }
        </style>
        <div style="margin-bottom: 1rem;">
            <a href="/" class="nav-button">Accueil</a>
            <a href="/Profil_Client" class="nav-button active">Profil Client</a>
            <a href="/Comparaison" class="nav-button">Comparaison</a>
            <a href="/Simulation" class="nav-button">Simulation</a>
        </div>
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
status_icon = "✅" if decision == "ACCEPTÉ" else "❌"

# Bannière de statut en haut de la page
st.markdown(
    f"""
    <div style="padding: 0.5rem 1rem; border-radius: 5px; background-color: {status_color}22; border: 1px solid {status_color};">
        <h2 style="color: {status_color}; margin: 0; display: flex; align-items: center;">
            {status_icon} Décision: {decision} • Probabilité de défaut: {probability:.1%}
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Organisation en tabs pour les différentes sections
tab1, tab2, tab3 = st.tabs(["Profil client", "Facteurs décisionnels", "Historique"])

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
                    "✅ Vérifié",
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

with tab2:
    # Section 3: Analyse des facteurs d'importance
    st.header("Facteurs influençant la décision")
    
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
        
        # Créer le graphique d'importance des features
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=feature_values,
            y=feature_names,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.3f}" for v in feature_values],
            textposition='auto'
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
            )
        )
        
        # Afficher le graphique
        st.plotly_chart(fig, use_container_width=True)
        
        # Légende explicative
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
            <h4>Comment interpréter ce graphique?</h4>
            <ul>
                <li><span style="color: #018571;">Les barres vertes</span> représentent des facteurs qui réduisent la probabilité de défaut (favorable au client).</li>
                <li><span style="color: #a6611a;">Les barres rouges</span> représentent des facteurs qui augmentent la probabilité de défaut (défavorable au client).</li>
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
            
            if feature_value is not None:
                explanation = f"La valeur de **{display_name}** est **{feature_value}**, ce qui {impact} {magnitude} le risque de défaut. Ce facteur est {direction} à la demande."
            else:
                explanation = f"Le facteur **{display_name}** {impact} {magnitude} le risque de défaut. Ce facteur est {direction} à la demande."
            
            explanations.append({
                "Facteur": display_name,
                "Impact": f"{'➖' if value < 0 else '➕'} {abs(value):.3f}",
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
        
    else:
        # Message si les valeurs SHAP ne sont pas disponibles
        st.info("""
        Les valeurs d'importance des caractéristiques (SHAP) ne sont pas disponibles pour ce client.
        
        Motifs possibles:
        - L'API de calcul SHAP est temporairement indisponible
        - Le modèle n'a pas pu calculer les valeurs pour ce client spécifique
        - Le client possède des caractéristiques atypiques
        
        Vous pouvez toujours analyser les autres informations du profil client.
        """)
    
    # Section d'analyse comparative des caractéristiques
    st.header("Analyse comparative des caractéristiques")
    
    # Sélection des caractéristiques à visualiser
    selected_features = st.multiselect(
        "Sélectionner des caractéristiques à comparer aux seuils:",
        options=list(details.get('features', {}).keys()),
        default=list(details.get('features', {}).keys())[:3]
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
        
        # Création du graphique
        fig = go.Figure()
        
        for data in feature_data:
            fig.add_trace(go.Scatter(
                x=[data["Caractéristique"]],
                y=[data["Valeur client"]],
                mode='markers',
                name=data["Caractéristique"],
                marker=dict(size=12, color=COLORBLIND_FRIENDLY_PALETTE["primary"])
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
                showlegend=False
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
                showlegend=False
            ))
        
        fig.update_layout(
            title="Positionnement du client par rapport aux seuils",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau des valeurs
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
    else:
        st.info("Veuillez sélectionner au moins une caractéristique pour l'analyse comparative.")

with tab3:
    # Section 4: Historique des décisions
    st.header("Historique des décisions pour ce client")
    
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
        
        # Appliquer un formatage conditionnel pour les décisions
        def color_decision(val):
            if val == "ACCEPTÉ":
                return f'background-color: {COLORBLIND_FRIENDLY_PALETTE["accepted"]}22; color: {COLORBLIND_FRIENDLY_PALETTE["accepted"]}'
            elif val == "REFUSÉ":
                return f'background-color: {COLORBLIND_FRIENDLY_PALETTE["refused"]}22; color: {COLORBLIND_FRIENDLY_PALETTE["refused"]}'
            else:
                return ''
        
        # Ajouter formatage pour le score
        def color_score(val):
            if val > DEFAULT_THRESHOLD:
                return f'color: {COLORBLIND_FRIENDLY_PALETTE["refused"]}'
            else:
                return f'color: {COLORBLIND_FRIENDLY_PALETTE["accepted"]}'
        
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
        
        # Graphique d'évolution des scores
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
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
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
    
    new_notes = st.text_area(
        "Notes de suivi détaillées",
        value=current_notes,
        height=150,
        placeholder="Saisissez ici vos observations, échanges avec le client, ou actions de suivi..."
    )
    
    if new_notes != current_notes:
        st.session_state.detailed_notes[client_id] = new_notes
        st.success("Notes enregistrées")

with col_notes2:
    # Actions possibles
    with st.container(border=True):
        st.subheader("Actions rapides")
        
        if st.button("📧 Envoyer un récapitulatif", use_container_width=True):
            st.info("Fonctionnalité d'envoi d'email à implémenter.")
            
        if decision == "REFUSÉ" and st.button("📝 Demander une révision", use_container_width=True):
            st.info("Redirection vers le formulaire de révision.")
            
        if st.button("🔙 Retour à l'accueil", use_container_width=True):
            st.switch_page("Home.py")

# Footer avec informations de version
st.markdown("""
<hr>
<div style="text-align: center; color: #666;">
    <small>
        Profil client détaillé | 2025-10-10 09:30:45 | 
        <span aria-label="Symbole monétaire utilisé: Rouble russe">Montants en roubles (₽)</span> | 
        Contact support: poste 4242
    </small>
</div>
""", unsafe_allow_html=True)