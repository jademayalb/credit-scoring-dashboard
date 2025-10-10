"""
Dashboard de Scoring Crédit pour Chargés de Relation Client
Facilite l'explication des décisions de crédit aux clients et leur révision si nécessaire
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import datetime
from utils.api_client import get_client_prediction, get_available_clients, get_client_details

# Import de la configuration
from config import (
    COLORBLIND_FRIENDLY_PALETTE, UI_CONFIG, 
    CSV_PATHS, DEFAULT_THRESHOLD
)

# Configuration de la page
st.set_page_config(
    page_title="Outil d'Analyse pour Chargés de Relation Client",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Barre de navigation principale
tabs = ["Accueil", "Profil Client", "Comparaison", "Simulation"]
selected_tab = st.tabs(tabs)

with selected_tab[0]:  # Accueil
    # Titre et description adaptés aux chargés de relation client
    st.title("Outil d'Analyse des Décisions de Crédit")
    
    st.markdown("""
    <div role="contentinfo" aria-label="Description de l'application">
    <p>Bienvenue dans votre outil d'analyse de décisions de crédit. Cette interface vous permet de:</p>
    <ul>
        <li><strong>Consulter</strong> rapidement les décisions d'octroi de crédit</li>
        <li><strong>Expliquer</strong> clairement aux clients les facteurs déterminants</li>
        <li><strong>Comparer</strong> le profil d'un client à des profils similaires</li>
        <li><strong>Simuler</strong> l'impact de modifications sur la décision finale</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Section de recherche client - prominente pour les chargés de relation
    st.header("Rechercher un dossier client")
    
    # Conteneur principal de recherche avec bordure
    with st.container(border=True):
        col_search_1, col_search_2 = st.columns([3, 1])
        
        with col_search_1:
            # Récupération des clients
            all_clients = get_available_clients(limit=UI_CONFIG["default_limit"])
            
            if all_clients:
                # Menu déroulant avec recherche
                selected_client_id = st.selectbox(
                    "Entrez l'identifiant du client:",
                    options=all_clients,
                    format_func=lambda x: f"Client #{x}",
                    help="Cliquez pour voir la liste ou commencez à taper pour filtrer",
                    key="main_client_search"
                )
            else:
                st.error("Aucun client disponible. Vérifiez la connexion à la base de données.")
                selected_client_id = None
        
        with col_search_2:
            # Bouton d'action principal pour les chargés de relation
            if selected_client_id and st.button("Consulter ce dossier", key="btn_analyze_main", type="primary", use_container_width=True):
                st.session_state.client_id = selected_client_id
                st.rerun()
    
    # Affichage des derniers clients consultés - utile pour les chargés qui travaillent avec plusieurs clients
    st.subheader("Dossiers récemment consultés")
    
    # Initialiser l'historique s'il n'existe pas
    if "recent_clients" not in st.session_state:
        st.session_state.recent_clients = []
    
    # Mettre à jour l'historique si un client est consulté
    if "client_id" in st.session_state:
        client_id = st.session_state.client_id
        if client_id in st.session_state.recent_clients:
            st.session_state.recent_clients.remove(client_id)
        st.session_state.recent_clients.insert(0, client_id)
        st.session_state.recent_clients = st.session_state.recent_clients[:5]
    
    # Afficher les clients récents
    if st.session_state.recent_clients:
        recent_cols = st.columns(5)
        for i, recent_id in enumerate(st.session_state.recent_clients):
            with recent_cols[i]:
                if st.button(f"Dossier #{recent_id}", key=f"recent_{recent_id}", use_container_width=True):
                    st.session_state.client_id = recent_id
                    st.rerun()
    else:
        st.info("Aucun dossier client consulté récemment")

# Fonctions d'affichage
def display_client_overview(client_id):
    """Affiche un aperçu des informations client adapté aux besoins des chargés de relation"""
    
    with st.spinner(f"Chargement du dossier client {client_id}..."):
        prediction = get_client_prediction(client_id)
        details = get_client_details(client_id)
    
    if not prediction or not details:
        st.error(f"Impossible de récupérer le dossier client {client_id}.")
        if st.button("Retour à l'accueil"):
            if "client_id" in st.session_state:
                del st.session_state.client_id
            st.rerun()
        return
    
    # En-tête avec ID et statut - informations essentielles pour le chargé
    col_header_1, col_header_2 = st.columns([3, 1])
    
    with col_header_1:
        st.header(f"Dossier client #{client_id}")
        
    with col_header_2:
        decision = prediction.get('decision', 'INCONNU')
        icon = "✅" if decision == "ACCEPTÉ" else "❌"
        color = COLORBLIND_FRIENDLY_PALETTE['accepted'] if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE['refused']
        st.markdown(f"""
        <h3 style='color: {color}; text-align: right;'>
            {icon} {decision}
        </h3>
        <div role="status" aria-live="polite" class="visually-hidden">
            Décision pour ce dossier: {decision}
        </div>
        """, unsafe_allow_html=True)
    
    # Notes du chargé de relation (nouvelle fonctionnalité)
    if "client_notes" not in st.session_state:
        st.session_state.client_notes = {}
    
    with st.expander("📝 Notes sur ce dossier client", expanded=False):
        current_note = st.session_state.client_notes.get(client_id, "")
        note = st.text_area(
            "Notes de suivi (visibles uniquement par les chargés de relation):",
            value=current_note,
            height=100,
            help="Documentez ici vos échanges avec le client ou vos observations"
        )
        if note != current_note:
            st.session_state.client_notes[client_id] = note
            st.success("Note enregistrée")
    
    # Informations principales - organisées pour faciliter l'explication au client
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Informations personnelles - ce que le chargé peut discuter avec le client
        with st.container(border=True):
            st.subheader("Profil du demandeur")
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown(f"**Genre:** {details['personal_info'].get('gender', '')}")
                st.markdown(f"**Âge:** {details['personal_info'].get('age', '')} ans")
                st.markdown(f"**Éducation:** {details['personal_info'].get('education', '')}")
                st.markdown(f"**Statut familial:** {details['personal_info'].get('family_status', '')}")
            
            with col_info2:
                st.markdown(f"**Revenu annuel:** {details['personal_info'].get('income', 0):,.0f} {UI_CONFIG['currency_symbol']}")
                st.markdown(f"**Ancienneté d'emploi:** {details['personal_info'].get('employment_years', 0)} ans")
                
        # Informations crédit - détails que le chargé peut expliquer
        with st.container(border=True):
            st.subheader("Détails de la demande de crédit")
            
            col_credit1, col_credit2 = st.columns(2)
            
            with col_credit1:
                st.markdown(f"**Montant demandé:** {details['credit_info'].get('amount', 0):,.0f} {UI_CONFIG['currency_symbol']}")
                st.markdown(f"**Durée du crédit:** {details['credit_info'].get('credit_term', 0)} mois")
            
            with col_credit2:
                st.markdown(f"**Mensualité:** {details['credit_info'].get('annuity', 0):,.0f} {UI_CONFIG['currency_symbol']}/mois")
                st.markdown(f"**Valeur du bien:** {details['credit_info'].get('goods_price', 0):,.0f} {UI_CONFIG['currency_symbol']}")
    
    with col2:
        # Évaluation avec explication pour le chargé de relation client
        with st.container(border=True):
            st.subheader("Analyse du risque à expliquer au client")
            
            probability = prediction.get('probability', 0)
            threshold = prediction.get('threshold', DEFAULT_THRESHOLD)
            
            # CORRECTION: Jauge claire pour faciliter l'explication au client
            # Valeurs en pourcentage pour plus de clarté
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,  # Convertir en pourcentage pour l'affichage
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilité de défaut", 'font': {'size': 18}},
                number={'suffix': "%", 'valueformat': ".1f", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100], 'ticksuffix': "%", 'tickformat': ".0f"},  # Échelle de 0 à 100%
                    'bar': {'color': COLORBLIND_FRIENDLY_PALETTE['primary']},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, threshold * 100], 'color': COLORBLIND_FRIENDLY_PALETTE['positive']},
                        {'range': [threshold * 100, 100], 'color': COLORBLIND_FRIENDLY_PALETTE['negative']}
                    ],
                    'threshold': {
                        'line': {'color': COLORBLIND_FRIENDLY_PALETTE['threshold'], 'width': 2},
                        'thickness': 0.75,
                        'value': threshold * 100  # Seuil en pourcentage
                    }
                }
            ))
            
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            
            gauge_description = f"""
            Jauge montrant la probabilité de défaut du client: {probability:.1%}.
            Le seuil critique est à {threshold:.1%}.
            Le client est {'en dessous' if probability < threshold else 'au-dessus'} du seuil de {abs(probability - threshold):.1%}.
            """
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Aide à l'explication pour le chargé de relation
            st.markdown(f"""
            <div aria-hidden="true">
                <p><strong>Points clés à expliquer au client:</strong></p>
                <ul>
                    <li>Le score de risque est de <strong>{probability:.1%}</strong></li>
                    <li>Notre seuil d'acceptation est fixé à <strong>{threshold:.1%}</strong></li>
                    <li>Le dossier est {'favorable' if probability < threshold else 'défavorable'} avec un écart de <strong>{abs(probability - threshold):.1%}</strong></li>
                </ul>
            </div>
            <div class="visually-hidden">{gauge_description}</div>
            """, unsafe_allow_html=True)
    
    # Section de contestation/révision - fonctionnalité spécifique pour les chargés
    if decision == "REFUSÉ":
        with st.container(border=True):
            st.subheader("Options de révision du dossier")
            
            col_rev1, col_rev2 = st.columns([3, 1])
            
            with col_rev1:
                revision_reason = st.selectbox(
                    "Motif de la demande de révision:",
                    options=["", "Informations complémentaires fournies", "Erreur dans les données saisies", 
                             "Garant ou co-emprunteur ajouté", "Modification du montant/durée", "Autre"]
                )
                if revision_reason == "Autre":
                    revision_reason = st.text_input("Précisez le motif:")
            
            with col_rev2:
                if revision_reason:
                    if st.button("Demander une révision", key="btn_revise", use_container_width=True):
                        # Simuler la demande de révision (à implémenter réellement)
                        st.success("Demande de révision enregistrée! Un analyste de crédit examinera ce dossier.")
    
    # Navigation vers les pages détaillées - adaptée aux tâches du chargé
    st.subheader("Outils d'analyse pour le chargé de relation")
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

def display_global_stats():
    """Affiche des statistiques globales utiles pour les chargés de relation client"""
    
    st.header("Vue d'ensemble du portefeuille de demandes")
    
    # Métriques clés utiles pour les chargés
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Taux d'approbation actuel", value="73%", delta="2%")
    
    with col2:
        st.metric(label="Montant moyen accordé", value=f"630 000 {UI_CONFIG['currency_symbol']}", delta=f"-15 000 {UI_CONFIG['currency_symbol']}")
    
    with col3:
        st.metric(label="Durée moyenne de traitement", value="2.5 jours", delta="-0.3 jour")
    
    with col4:
        st.metric(label="Taux de révision acceptée", value="12%", delta="1.5%")
    
    # Graphiques utiles pour le contexte du chargé
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Distribution des montants accordés")
        
        try:
            df = None
            for path in CSV_PATHS:
                try:
                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        break
                except:
                    continue
                
            if df is not None:
                fig = px.histogram(
                    df, 
                    x="AMT_CREDIT",
                    nbins=30, 
                    labels={"AMT_CREDIT": f"Montant du crédit ({UI_CONFIG['currency_symbol']})", "count": "Nombre de dossiers"},
                    color_discrete_sequence=[COLORBLIND_FRIENDLY_PALETTE["primary"]],
                    height=UI_CONFIG["chart_height"]
                )
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                
                hist_desc = """
                Histogramme montrant la distribution des montants de crédit accordés.
                La plupart des crédits se situent entre 300 000 et 700 000 roubles.
                """
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"""<div class="visually-hidden">{hist_desc}</div>""", unsafe_allow_html=True)
            else:
                st.error("Impossible de charger les données pour l'histogramme.")
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {str(e)}")
    
    with col_chart2:
        st.subheader("Répartition des décisions de crédit")
        
        decisions = pd.DataFrame({
            "Décision": ["Accepté", "Refusé"],
            "Pourcentage": [73, 27]
        })
        
        fig = px.pie(
            decisions,
            names="Décision",
            values="Pourcentage",
            color="Décision",
            color_discrete_map={
                "Accepté": COLORBLIND_FRIENDLY_PALETTE["accepted"], 
                "Refusé": COLORBLIND_FRIENDLY_PALETTE["refused"]
            },
            height=UI_CONFIG["chart_height"]
        )
        
        fig.update_traces(
            textinfo='percent+label',
            textposition='inside',
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        
        pie_desc = """
        Diagramme circulaire montrant la répartition des décisions de crédit.
        73% des demandes sont acceptées et 27% sont refusées.
        """
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""<div class="visually-hidden">{pie_desc}</div>""", unsafe_allow_html=True)
    
    # Ressources pour les chargés de relation client
    st.subheader("Ressources pour les chargés de relation client")
    
    with st.container(border=True):
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("""
            ### Comment utiliser ce dashboard
            - Recherchez un client par son identifiant
            - Consultez la décision et les facteurs déterminants
            - Utilisez la page "Simulation" pour tester des modifications
            - Documentez vos échanges dans la section "Notes"
            - Si nécessaire, demandez une révision de dossier
            """)
            
        with col_res2:
            st.markdown("""
            ### Comment expliquer une décision
            1. Présentez la jauge de risque et le seuil
            2. Expliquez les 3-5 facteurs les plus importants
            3. Proposez des pistes d'amélioration si refusé
            4. Documentez toutes les questions et réponses
            5. Orientez vers les alternatives si nécessaire
            """)

# Logique principale d'affichage
if "client_id" in st.session_state:
    display_client_overview(st.session_state.client_id)
else:
    display_global_stats()

# Ajout d'informations de pied de page
st.markdown(f"""
<hr>
<div style="text-align: center; color: #666;">
    <small>
        Outil d'analyse pour chargés de relation client | 2025-10-10 08:47:37 | 
        <span aria-label="Symbole monétaire utilisé: Rouble russe">Montants en roubles (₽)</span> | 
        Contact support: poste 4242
    </small>
</div>
""", unsafe_allow_html=True)

# CSS pour l'accessibilité
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
</style>
""", unsafe_allow_html=True)