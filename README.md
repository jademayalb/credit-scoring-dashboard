# Credit Scoring Dashboard

Un tableau de bord interactif de scoring de crédit développé en Python avec Streamlit. Il permet d'explorer les données, d'observer les prédictions d'un modèle de scoring, et d'interpréter les décisions pour chaque client.

Démo en ligne (Streamlit Cloud)
- https://credit-scoring-dashboard-lkjr6jumthmgv3amrbzpky.streamlit.app/

Principales fonctionnalités
- Visualisation exploratoire des données (distributions, corrélations, etc.)
- Interface pour entrer les caractéristiques d'un client et obtenir une prédiction de risque
- Explications locales/globales des décisions (ex. SHAP ou LIME si inclus)
- Comparaison des performances du modèle (matrices de confusion, ROC, AUC)

Prérequis
- Python 3.8+
- (Optionnel) virtualenv ou conda pour isoler l'environnement

Installation locale rapide
1. Cloner le dépôt :
   git clone https://github.com/jademayalb/credit-scoring-dashboard.git
2. Se placer dans le dossier :
   cd credit-scoring-dashboard
3. Créer et activer un environnement virtuel (ex. venv) :
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows
4. Installer les dépendances :
   pip install -r requirements.txt
   (Si vous n'avez pas de fichier requirements.txt, installer streamlit et les libs usuelles :)
   pip install streamlit pandas scikit-learn shap matplotlib seaborn plotly
5. Lancer l'application Streamlit :
   streamlit run app.py
   ou
   streamlit run src/app.py
   (Adapter le chemin si l'entrée principale diffère)

Structure du dépôt (exemple)
- app.py ou src/app.py — point d'entrée Streamlit
- requirements.txt — dépendances Python
- data/ — jeux de données (ex. CSV) ou scripts pour les charger
- models/ — modèles sauvegardés (.pkl) ou artefacts ML
- notebooks/ — notebooks d'exploration et d'entraînement
- README.md — ce fichier

Données & modèle
- Ne commitez pas de données sensibles ni d'informations personnelles.
- Si des modèles pré-entraînés sont fournis, ils se trouvent typiquement dans le dossier models/.
- Documenter la provenance des données et la méthode d'entraînement dans notebooks/ ou dans un fichier MODEL.md si nécessaire.

Déploiement
- Déploiement public effectué sur Streamlit Cloud (URL ci-dessus).
- Pour déployer vous-même sur Streamlit Cloud :
  1. Pousser le repo sur GitHub.
  2. Créer une nouvelle app sur https://share.streamlit.io/ en pointant vers le repository et la branche.
  3. Configurer les variables d'environnement (si nécessaire).

Contact
- Auteur : jademayalb (https://github.com/jademayalb)
- Démo : https://credit-scoring-dashboard-lkjr6jumthmgv3amrbzpky.streamlit.app/
