"""Configuration globale de l'application."""

# URLs des APIs
API_URL_BASE = "https://api-shap-jademayalb-2a1c1dd6f4bd.herokuapp.com"  # Nouvelle URL
PREDICT_ENDPOINT = f"{API_URL_BASE}/predict/"
DETAILS_ENDPOINT = f"{API_URL_BASE}/client/"  # Mise à jour pour correspondre à la nouvelle structure
SHAP_ENDPOINT = f"{API_URL_BASE}/shap_values/"  # Ajout du nouvel endpoint SHAP
CLIENTS_ENDPOINT = f"{API_URL_BASE}/clients"
CLIENT_DETAILS_ENDPOINT = f"{API_URL_BASE}/client/"  # Ajouter un slash à la fin

# Paramètres du modèle
DEFAULT_THRESHOLD = 0.52  # Seuil optimal déterminé précédemment

# Chemins possibles pour les fichiers CSV
CSV_PATHS = [
    "data/application_test.csv",                      # Si exécuté depuis credit-scoring-dashboard/
    "application_test.csv",                           # Si exécuté depuis credit-scoring-dashboard/data/
    "credit-scoring-dashboard/data/application_test.csv",  # Si exécuté depuis le répertoire parent
    "../data/application_test.csv"                    # Si exécuté depuis un sous-répertoire
]

# Configuration des graphiques et interface
COLORBLIND_FRIENDLY_PALETTE = {
    "positive": "#018571",  # Vert-bleu (pour les valeurs favorables)
    "negative": "#a6611a",  # Brun (pour les valeurs défavorables)
    "neutral": "#80cdc1",   # Turquoise clair (pour les valeurs neutres)
    "threshold": "#404040", # Gris foncé (pour les seuils)
    "primary": "#3366ff",   # Bleu (couleur principale pour graphiques)
    "accepted": "#018571",  # Vert-bleu (pour décisions "Accepté")
    "refused": "#a6611a"    # Brun (pour décisions "Refusé")
}

# Descriptions des features principales pour l'accessibilité
FEATURE_DESCRIPTIONS = {
    # Top features généralement importantes dans les modèles de scoring crédit
    "EXT_SOURCE_1": "Score normalisé - Source externe 1",
    "EXT_SOURCE_2": "Score normalisé - Source externe 2", 
    "EXT_SOURCE_3": "Score normalisé - Source externe 3",
    "DAYS_BIRTH": "Âge du client en jours (négatif)",
    "DAYS_EMPLOYED": "Ancienneté dans l'emploi actuel en jours (négatif)",
    "AMT_INCOME_TOTAL": "Revenu annuel déclaré",
    "AMT_CREDIT": "Montant du crédit demandé",
    "PAYMENT_RATE": "Taux de paiement (annuité/crédit)",
    "CREDIT_INCOME_RATIO": "Ratio crédit/revenu"
}

# Paramètres de l'interface utilisateur
UI_CONFIG = {
    "default_limit": 100,            # Nombre maximum de clients à afficher par défaut
    "chart_height": 300,             # Hauteur par défaut des graphiques
    "max_features_display": 10,      # Nombre maximum de features à afficher dans les visualisations
    "currency_symbol": "₽",          # Symbole de la monnaie (rouble)
    "locale": "fr_FR"                # Paramètres régionaux pour le formatage des nombres
}
