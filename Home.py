import streamlit as st

st.set_page_config(
    page_title="Scoring Crédit Dashboard",
    layout="wide",
)

st.title("Scoring Crédit – Dashboard Interactif")

st.write("""
# Bienvenue sur le Dashboard de Scoring Crédit
Cette application permet aux chargés de relation client d'explorer et d'expliquer 
les décisions d'octroi de crédit aux clients.
""")

# Placeholder pour la sélection client
st.subheader("Rechercher un client")
client_id = st.number_input("Entrez l'identifiant client:", min_value=1)

if st.button("Rechercher"):
    st.session_state.client_id = client_id
    st.success(f"Client {client_id} sélectionné. Naviguez vers la page 'Profil Client' pour voir les détails.")
