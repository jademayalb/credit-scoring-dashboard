"""Configuration globale de l'application."""

# URLs des APIs
API_URL_BASE = "https://credit-scoring-jademayalb-db8bcc609fed.herokuapp.com"
PREDICT_ENDPOINT = f"{API_URL_BASE}/predict/"
DETAILS_ENDPOINT = f"{API_URL_BASE}/client_details/"
SHAP_ENDPOINT = f"{API_URL_BASE}/shap_values/"

# Paramètres du modèle
DEFAULT_THRESHOLD = 0.52  # Seuil optimal déterminé précédemment

# Configuration des graphiques
COLORBLIND_FRIENDLY_PALETTE = {
    "positive": "#018571",  # Vert-bleu
    "negative": "#a6611a",  # Brun
    "neutral": "#80cdc1",   # Turquoise clair
    "threshold": "#404040"  # Gris foncé
}

# Descriptions des features principales pour l'accessibilité
FEATURE_DESCRIPTIONS = {
    # Top features généralement importantes dans les modèles de scoring crédit
    "EXT_SOURCE_3": "Score externe 3 (plus élevé = meilleur risque)",
    "EXT_SOURCE_2": "Score externe 2 (plus élevé = meilleur risque)",
    "EXT_SOURCE_1": "Score externe 1 (plus élevé = meilleur risque)",
    "DAYS_BIRTH": "Âge du client en jours (négatif)",
    "DAYS_EMPLOYED": "Ancienneté dans l'emploi actuel en jours (négatif)",
    "AMT_INCOME_TOTAL": "Revenu annuel déclaré",
    "AMT_CREDIT": "Montant du crédit demandé",
    "PAYMENT_RATE": "Taux de paiement (annuité/crédit)",
    "CREDIT_INCOME_RATIO": "Ratio crédit/revenu"
}
