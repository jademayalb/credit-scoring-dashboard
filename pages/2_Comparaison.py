"""
Page de comparaison entre clients

"""
import traceback
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from utils.api_client import get_client_prediction, get_client_details, get_available_clients
from config import COLORBLIND_FRIENDLY_PALETTE, UI_CONFIG, FEATURE_DESCRIPTIONS

st.set_page_config(page_title="Comparaison de Clients - Dashboard de Scoring Crédit", page_icon="📊", layout="wide")

# Ensure some missing feature descriptions exist (in-memory) so UI does not show "Pas de description disponible"
FEATURE_DESCRIPTIONS.setdefault("AMT_GOODS_PRICE", "Prix du bien/service financé (montant en roubles). Ex. prix du véhicule ou du bien acheté.")
FEATURE_DESCRIPTIONS.setdefault("AMT_ANNUITY", "Montant de l'annuité / mensualité (exprimé en roubles). Utilisé pour estimer l'effort de paiement du client.")
FEATURE_DESCRIPTIONS.setdefault("NAME_EDUCATION_TYPE", "Niveau d'éducation du client (ex.: Secondary, Higher education). Utile pour segmenter la clientèle et comprendre des différences de profil.")
FEATURE_DESCRIPTIONS.setdefault("DAYS_EMPLOYED", "Ancienneté d'emploi du client (converti automatiquement en années positives pour l'affichage). Indicateur de stabilité professionnelle.")
FEATURE_DESCRIPTIONS.setdefault("AMT_INCOME_TOTAL", "Revenu total annuel du client (en roubles). Base pour calculer les ratios d'endettement.")
FEATURE_DESCRIPTIONS.setdefault("DAYS_BIRTH", "Âge du client (converti automatiquement en années positives pour l'affichage)")
FEATURE_DESCRIPTIONS.setdefault("AMT_CREDIT", "Montant du crédit demandé (en roubles)")

# --- Styles légers pour accessibilité ---
st.markdown("""
<style>
    .dataframe th { background-color: #f0f0f0 !important; color: #000000 !important; font-weight: bold !important; }
    .dataframe td { background-color: #ffffff !important; color: #000000 !important; }
    body, .stMarkdown, .stText { font-size: 1rem !important; line-height: 1.6 !important; }
</style>
""", unsafe_allow_html=True)

# --- Titre ---
st.title("Comparaison de profils clients")
st.markdown("Comparez plusieurs profils clients côte à côte et explorez des relations entre caractéristiques importantes. Les transformations nécessaires sont appliquées automatiquement en arrière-plan.")

# ---------- Helpers ----------
def normalize_id_list(lst):
    out = []
    for v in lst or []:
        try:
            out.append(int(v))
        except Exception:
            try:
                out.append(int(str(v)))
            except Exception:
                out.append(v)
    return out

def convert_days_to_years(series):
    """
    Convertit DAYS_BIRTH et DAYS_EMPLOYED en années positives et compréhensibles
    DAYS_BIRTH: négatif dans les données -> âge positif 
    DAYS_EMPLOYED: négatif dans les données -> ancienneté positive (sauf 365243 = valeur manquante)
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace({365243: np.nan})  # valeur manquante pour DAYS_EMPLOYED
    # Convertir en années positives : -(-19243)/365.25 = +52.7 ans
    return (-s / 365.25)  # 365.25 pour tenir compte des années bissextiles

def format_age_for_display(age_in_years):
    """
    Formate l'âge pour un affichage convivial
    """
    if pd.isna(age_in_years):
        return "N/A"
    return f"{age_in_years:.1f} ans"

def format_employment_for_display(years):
    """
    Formate l'ancienneté d'emploi pour un affichage convivial
    """
    if pd.isna(years):
        return "N/A"
    if years < 1:
        return f"{years*12:.0f} mois"
    return f"{years:.1f} ans"

def format_money_for_display(amount):
    """
    Formate les montants en roubles pour un affichage convivial
    """
    if pd.isna(amount):
        return "N/A"
    return f"{amount:,.0f} ₽"

def backend_prepare_plot(df, x_feat, y_feat):
    """
    Transformations en backend (non visibles) :
    - DAYS_* -> années positives
    - compression log1p pour montants (si présent)
    - standard scaling pour comparabilité visuelle
    Retourne df avec x_plot/y_plot et labels.
    """
    dfp = df.copy()
    dfp["x_num"] = pd.to_numeric(dfp["x_raw"], errors="coerce")
    dfp["y_num"] = pd.to_numeric(dfp["y_raw"], errors="coerce")

    # ✅ CORRECTION : Convert DAYS fields avec meilleur formatage
    if x_feat in ("DAYS_BIRTH", "DAYS_EMPLOYED"):
        dfp["x_num_conv"] = convert_days_to_years(dfp["x_num"])
        # Créer une version formatée pour l'affichage
        if x_feat == "DAYS_BIRTH":
            dfp["x_display"] = dfp["x_num_conv"].apply(format_age_for_display)
        else:  # DAYS_EMPLOYED
            dfp["x_display"] = dfp["x_num_conv"].apply(format_employment_for_display)
    else:
        dfp["x_num_conv"] = dfp["x_num"]
        # Formatter les montants monétaires
        if x_feat in ("AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL"):
            dfp["x_display"] = dfp["x_num"].apply(format_money_for_display)
        else:
            dfp["x_display"] = dfp["x_raw"]

    if y_feat in ("DAYS_BIRTH", "DAYS_EMPLOYED"):
        dfp["y_num_conv"] = convert_days_to_years(dfp["y_num"])
        # Créer une version formatée pour l'affichage
        if y_feat == "DAYS_BIRTH":
            dfp["y_display"] = dfp["y_num_conv"].apply(format_age_for_display)
        else:  # DAYS_EMPLOYED
            dfp["y_display"] = dfp["y_num_conv"].apply(format_employment_for_display)
    else:
        dfp["y_num_conv"] = dfp["y_num"]
        # Formatter les montants monétaires
        if y_feat in ("AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL"):
            dfp["y_display"] = dfp["y_num"].apply(format_money_for_display)
        else:
            dfp["y_display"] = dfp["y_raw"]

    # Money compression heuristic
    money_feats = {"AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL"}
    if (x_feat in money_feats) or (y_feat in money_feats):
        x_pre = np.where(dfp["x_num_conv"] > 0, np.log1p(dfp["x_num_conv"]), dfp["x_num_conv"])
        y_pre = np.where(dfp["y_num_conv"] > 0, np.log1p(dfp["y_num_conv"]), dfp["y_num_conv"])
    else:
        x_pre = dfp["x_num_conv"].values
        y_pre = dfp["y_num_conv"].values

    # Standard scaling to make comparable plotting scales
    try:
        sx = StandardScaler(); sy = StandardScaler()
        dfp["x_plot"] = sx.fit_transform(np.array(x_pre).reshape(-1,1)).flatten()
        dfp["y_plot"] = sy.fit_transform(np.array(y_pre).reshape(-1,1)).flatten()
    except Exception:
        dfp["x_plot"] = dfp["x_num_conv"]
        dfp["y_plot"] = dfp["y_num_conv"]

    # ✅ AMÉLIORATION : Labels plus clairs et spécifiques
    x_label = FEATURE_DESCRIPTIONS.get(x_feat, x_feat)
    y_label = FEATURE_DESCRIPTIONS.get(y_feat, y_feat)
    
    if x_feat == "DAYS_BIRTH":
        x_label = "Âge (années)"
    elif x_feat == "DAYS_EMPLOYED":
        x_label = "Ancienneté d'emploi (années)"
    elif x_feat == "AMT_INCOME_TOTAL":
        x_label = "Revenus annuels déclarés (roubles)"
    elif x_feat == "AMT_CREDIT":
        x_label = "Montant crédit (roubles)"
    elif x_feat == "AMT_ANNUITY":
        x_label = "Montant mensualité (roubles)"
    elif x_feat == "AMT_GOODS_PRICE":
        x_label = "Prix du bien (roubles)"
        
    if y_feat == "DAYS_BIRTH":
        y_label = "Âge (années)"
    elif y_feat == "DAYS_EMPLOYED":
        y_label = "Ancienneté d'emploi (années)"
    elif y_feat == "AMT_INCOME_TOTAL":
        y_label = "Revenus annuels déclarés (roubles)"
    elif y_feat == "AMT_CREDIT":
        y_label = "Montant crédit (roubles)"
    elif y_feat == "AMT_ANNUITY":
        y_label = "Montant mensualité (roubles)"
    elif y_feat == "AMT_GOODS_PRICE":
        y_label = "Prix du bien (roubles)"

    return dfp, x_label, y_label

def get_feature_from_client_data(client_details, feature_name):
    """
    Récupère une feature depuis les données client, en gérant les cas spéciaux
    comme AMT_INCOME_TOTAL qui peut être dans personal_info au lieu de features
    """
    features = client_details.get("features", {})
    
    # Essayer d'abord dans features
    if feature_name in features:
        return features[feature_name]
    
    # Cas spéciaux : certaines features peuvent être dans d'autres sections
    if feature_name == "AMT_INCOME_TOTAL":
        personal_info = client_details.get("personal_info", {})
        return personal_info.get("income", None)
    
    return None

# --- Sanitize helper to avoid pyarrow conversion errors ---
def sanitize_df_for_streamlit(df):
    """
    Ensure DataFrame columns are compatible with pyarrow/Streamlit rendering.
    - decode bytes to str
    - coerce mixed object columns to numeric if mostly numeric else to str
    """
    if df is None:
        return df
    df = df.copy()
    for col in df.columns:
        try:
            # decode bytes in column if any
            if df[col].dtype == object:
                df[col] = df[col].apply(lambda v: v.decode('utf-8', errors='ignore') if isinstance(v, (bytes, bytearray)) else v)
        except Exception:
            try:
                df[col] = df[col].astype(object)
            except Exception:
                pass

        # if still object, try to coerce to numeric when many values numeric
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors='coerce')
            num_count = int(coerced.notna().sum())
            if num_count >= max(3, int(len(df) * 0.4)):
                df[col] = coerced
            else:
                df[col] = df[col].astype(str).replace('None', '')
    return df

# ---------- Chargement liste clients ----------
with st.spinner("Chargement de la liste des clients..."):
    try:
        available_clients = normalize_id_list(get_available_clients(limit=UI_CONFIG.get("default_limit", 100)) or [])
    except Exception:
        available_clients = []

if not available_clients:
    st.error("Impossible de charger la liste des clients.")
    st.stop()

# ---------- Sélection clients ----------
selected_clients = st.multiselect(
    "Sélectionnez 2 à 4 clients à comparer",
    options=available_clients,
    default=available_clients[:2] if len(available_clients) >= 2 else available_clients,
    max_selections=4
)
selected_clients = normalize_id_list(selected_clients)
if len(selected_clients) < 2:
    st.warning("Veuillez sélectionner au moins deux clients pour la comparaison.")
    st.stop()

reference_client = st.selectbox("Choisir un client de référence (mise en évidence)", options=selected_clients, format_func=lambda x: f"Client #{x}")
try:
    reference_client = int(reference_client)
except Exception:
    pass

# ---------- Charger données clients sélectionnés ----------
with st.spinner("Chargement des données des clients sélectionnés..."):
    client_data = {}
    for cid in selected_clients:
        try:
            pred = get_client_prediction(cid)
            det = get_client_details(cid)
            if pred and det:
                client_data[int(cid)] = {"prediction": pred, "details": det}
        except Exception:
            # silencieux ici ; affichage d'erreur en cas d'absence totale plus bas
            pass

if not client_data:
    st.error("Aucune donnée client n'a pu être chargée.")
    st.stop()

# ---------- Cartes statut ----------
st.subheader("Statut des demandes de crédit")
cols = st.columns(len(client_data))
for i, (cid, data) in enumerate(client_data.items()):
    prediction = data["prediction"]
    probability = prediction.get("probability", 0)
    decision = prediction.get("decision", "INCONNU")
    with cols[i]:
        status_color = COLORBLIND_FRIENDLY_PALETTE.get('accepted', '#2ca02c') if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused', '#d62728')
        status_icon = "✅" if decision == "ACCEPTÉ" else "❌"
        st.markdown(
            f"""
            <div style="padding: 0.6rem; border-radius: 8px; background: {status_color}20; border: 1.5px solid {status_color};">
                <strong>Client #{cid}</strong><br>
                <span style="color:{status_color}">{status_icon} <strong>{decision}</strong></span><br>
                Probabilité: <strong>{probability:.1%}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------- Tableaux de comparaison ----------
st.subheader("Comparaison des informations personnelles")
comparison_data = []
for client_id, data in client_data.items():
    details = data["details"]
    pi = details.get("personal_info", {})
    comparison_data.append({
        "ID Client": client_id,
        "Âge": pi.get("age", "N/A"),
        "Genre": pi.get("gender", "N/A"),
        "Éducation": pi.get("education", "N/A"),
        "Statut familial": pi.get("family_status", "N/A"),
        "Revenu annuel": pi.get("income", "N/A"),
        "Ancienneté d'emploi": pi.get("employment_years", "N/A"),
    })
comparison_df = pd.DataFrame(comparison_data)
comparison_df = sanitize_df_for_streamlit(comparison_df)
st.dataframe(comparison_df, width='stretch')

st.subheader("Comparaison des crédits demandés")
credit_data = []
for client_id, data in client_data.items():
    details = data["details"]
    credit_info = details.get("credit_info", {})
    income = details.get("personal_info", {}).get("income", 0) or 0
    annuity = credit_info.get("annuity", 0) or 0
    payment_ratio = annuity * 12 / max(income, 1) if income > 0 else np.nan
    credit_data.append({
        "ID Client": client_id,
        "Montant demandé": credit_info.get("amount", "N/A"),
        "Durée (mois)": credit_info.get("credit_term", "N/A"),
        "Mensualité": annuity,
        "Valeur du bien": credit_info.get("goods_price", "N/A"),
        "Ratio mensualité/revenu": payment_ratio
    })
credit_df = pd.DataFrame(credit_data)
credit_df = sanitize_df_for_streamlit(credit_df)
st.dataframe(credit_df, width='stretch')

# ---------- Analyse univariée ----------
st.subheader("Analyse univariée : distribution des probabilités de défaut")
all_client_ids = normalize_id_list(get_available_clients(limit=UI_CONFIG.get("default_limit", 100)) or [])
probs = []
ids_with_prob = []
with st.spinner("Récupération des probabilités pour la distribution..."):
    for cid in all_client_ids:
        try:
            pred = get_client_prediction(cid)
            if pred:
                probs.append(pred.get("probability", 0))
                ids_with_prob.append((cid, pred.get("probability", 0)))
        except Exception:
            continue

if not probs:
    st.info("Aucune probabilité disponible pour l'instant.")
else:
    dist_df = pd.DataFrame({"client_id": [c for c, p in ids_with_prob], "probability": [p for c, p in ids_with_prob]})
    fig_hist = px.histogram(dist_df, x="probability", nbins=30, title="Distribution des probabilités de défaut (ensemble des clients)",
                            color_discrete_sequence=[COLORBLIND_FRIENDLY_PALETTE.get("primary", "#636EFA")])
    fig_hist.update_layout(xaxis=dict(tickformat=".0%", title="Probabilité de défaut"), yaxis_title="Nombre de clients", height=320)
    try:
        ref_pred = client_data[int(reference_client)]["prediction"]
        ref_prob = ref_pred.get("probability", 0)
        fig_hist.add_vline(x=ref_prob, line_dash="dash", line_color="black", annotation_text=f"Réf #{reference_client}: {ref_prob:.1%}", annotation_position="top right")
    except Exception:
        ref_prob = None
    st.plotly_chart(fig_hist, width='stretch')

    if ref_prob is not None:
        percentile = (np.sum(np.array(probs) <= ref_prob) / len(probs)) * 100
        st.markdown(f"**Interprétation rapide :** Le client #{reference_client} est au {percentile:.1f}e percentile (0% = le plus sûr).")
        if percentile <= 25:
            st.success("Ce client est dans les 25% les plus sûrs.")
        elif percentile <= 50:
            st.info("Ce client est dans le 25-50% (risque relativement faible).")
        elif percentile <= 75:
            st.warning("Ce client est dans le 50-75% (risque modéré).")
        else:
            st.error("Ce client est dans les 25% les moins sûrs.")

# ---------- Analyse bivariée (section intégrée) ----------
st.subheader("Analyse bivariée : comparer deux caractéristiques pertinentes")

# ✅ NOUVELLES PAIRES MÉTIER corrigées et plus pertinentes (axes corrigés, sans scores externes flous)
PAIRS = [
    {"key": "employed_vs_age", "x": "DAYS_EMPLOYED", "y": "DAYS_BIRTH", "label": "Ancienneté d'emploi vs Âge", "type": "employment_vs_age"},
    {"key": "credit_vs_price", "x": "AMT_CREDIT", "y": "AMT_GOODS_PRICE", "label": "Montant crédit vs Prix du bien", "type": "money_vs_money"},
    {"key": "income_vs_annuity", "x": "AMT_INCOME_TOTAL", "y": "AMT_ANNUITY", "label": "Revenus annuels vs Mensualité", "type": "income_vs_payment"},
    {"key": "age_vs_credit", "x": "DAYS_BIRTH", "y": "AMT_CREDIT", "label": "Âge vs Montant crédit", "type": "age_vs_money"},
    {"key": "income_vs_credit", "x": "AMT_INCOME_TOTAL", "y": "AMT_CREDIT", "label": "Revenus vs Montant crédit", "type": "money_vs_money"},
    {"key": "education_vs_credit", "x": "NAME_EDUCATION_TYPE", "y": "AMT_CREDIT", "label": "Niveau d'éducation vs Montant crédit", "type": "cat_vs_score"}
]
pair_map = {p["key"]: p for p in PAIRS}
pair_labels = {p["key"]: p["label"] for p in PAIRS}

choice_key = st.selectbox("Choisir une paire métier à explorer", options=list(pair_labels.keys()), format_func=lambda k: pair_labels[k])
pair = pair_map[choice_key]
x_feature = pair["x"]; y_feature = pair["y"]; pair_type = pair["type"]
st.markdown(f"Comparaison sélectionnée : **{pair_labels[choice_key]}** (transformations automatiques appliquées).")

# construire échantillon pour graphe (jusqu'à 100 clients)
limit_for_bi = 100
with st.spinner("Chargement d'un échantillon de clients pour la bivariée..."):
    sample_ids = normalize_id_list(get_available_clients(limit=limit_for_bi) or [])[:limit_for_bi]
    if not sample_ids:
        sample_ids = available_clients[:limit_for_bi]
    rows = []
    for cid in sample_ids:
        try:
            d = get_client_details(cid)
            p = get_client_prediction(cid)
            if not d or not p:
                continue
            
            # Utiliser la fonction helper pour récupérer les features
            x_raw = get_feature_from_client_data(d, x_feature)
            y_raw = get_feature_from_client_data(d, y_feature)
            
            if x_raw is None or y_raw is None:
                continue
            prob = p.get("probability", 0)
            thr = p.get("threshold", 0.52)
            decision = "ACCEPTÉ" if prob < thr else "REFUSÉ"
            rows.append({"client_id": int(cid), "x_raw": x_raw, "y_raw": y_raw, "probability": prob, "decision": decision})
        except Exception:
            continue

if not rows:
    st.info("Pas assez de données pour cette paire — essayez une autre paire.")
else:
    df = pd.DataFrame(rows)

    # si catégorie vs score => boxplot + résumé adapté (regroupement des petites catégories)
    if pair_type == "cat_vs_score":
        # préparations
        df["category"] = df["x_raw"].astype(str).fillna("Inconnu")
        df["value"] = pd.to_numeric(df["y_raw"], errors="coerce")
        df = df.dropna(subset=["value"]).copy()

        if df.empty:
            st.info("Pas assez de données numériques pour le boxplot.")
        else:
            # effectifs par catégorie
            counts = df["category"].value_counts(dropna=False).rename_axis("category").reset_index(name="n")
            total = counts["n"].sum()
            counts["pct"] = counts["n"] / total

            # seuils métier : regrouper catégories trop petites
            MIN_COUNT = 5         # au moins 5 clients
            MIN_PCT = 0.03        # ou 3% minimum
            rare_cats = counts[(counts["n"] < MIN_COUNT) | (counts["pct"] < MIN_PCT)]["category"].tolist()

            if rare_cats:
                df["category_grouped"] = df["category"].apply(lambda c: "Autre" if c in rare_cats else c)
            else:
                df["category_grouped"] = df["category"]

            # résumé par catégorie groupée
            summary = df.groupby("category_grouped")["value"].agg(
                n="count", median=lambda s: s.median(), q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75)
            ).reset_index()
            summary = summary.sort_values("median", ascending=False)
            ordered_cats = summary["category_grouped"].tolist()
            df["category_grouped"] = pd.Categorical(df["category_grouped"], categories=ordered_cats, ordered=True)

            # afficher tableau d'effectifs et avertissement si petits effectifs
            st.markdown("Effectifs par catégorie (les petites catégories sont regroupées en 'Autre') :")
            summary_disp = summary[["category_grouped", "n"]].rename(columns={"category_grouped": "Catégorie", "n": "Effectif"})
            summary_disp = sanitize_df_for_streamlit(summary_disp)
            st.dataframe(summary_disp, width='stretch')

            small_groups = summary[summary["n"] < MIN_COUNT]
            if not small_groups.empty:
                st.info("Quelques catégories ont un effectif faible après regroupement. Interprète les différences avec prudence.")

            # boxplot ordonné par médiane
            fig_box = px.box(df, x="category_grouped", y="value", points="all",
                             labels={"category_grouped": FEATURE_DESCRIPTIONS.get(x_feature, x_feature),
                                     "value": FEATURE_DESCRIPTIONS.get(y_feature, y_feature)},
                             title=pair_labels[choice_key],
                             color_discrete_sequence=[COLORBLIND_FRIENDLY_PALETTE.get("primary", "#636EFA")])
            fig_box.update_layout(xaxis_title="Catégorie", yaxis_title="Montant crédit (roubles)")
            st.plotly_chart(fig_box, width='stretch')

            # barplot des effectifs
            fig_counts = px.bar(summary, x="category_grouped", y="n",
                                labels={"category_grouped": "Catégorie", "n": "Effectif"},
                                title="Effectifs par catégorie (après regroupement)")
            st.plotly_chart(fig_counts, width='stretch')

            # indiquer la catégorie du client de référence si présente
            try:
                ref_cat = df.loc[df["client_id"] == reference_client, "category_grouped"].iloc[0]
                st.info(f"Le client de référence appartient à la catégorie : **{ref_cat}**")
            except Exception:
                pass

    else:
        df_prep, x_label, y_label = backend_prepare_plot(df, x_feature, y_feature)

        # marquer sélection/référence
        sel_set = set(selected_clients)
        df_prep["is_selected"] = df_prep["client_id"].isin(sel_set)
        df_prep["is_reference"] = df_prep["client_id"] == int(reference_client)

        df_other = df_prep[~df_prep["is_selected"] & ~df_prep["is_reference"]]
        df_sel = df_prep[df_prep["is_selected"] & ~df_prep["is_reference"]]
        df_ref = df_prep[df_prep["is_reference"]]

        fig = go.Figure()
        if not df_other.empty:
            fig.add_trace(go.Scatter(
                x=df_other["x_plot"], y=df_other["y_plot"], mode="markers",
                marker=dict(size=8, symbol="circle",
                            color=[COLORBLIND_FRIENDLY_PALETTE.get('accepted','#2ca02c') if d == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused','#d62728') for d in df_other["decision"]],
                            opacity=0.8),
                customdata=np.stack([df_other["client_id"], df_other.get("x_display", df_other["x_raw"]), df_other.get("y_display", df_other["y_raw"]), df_other["probability"]], axis=-1),
                hovertemplate=("Client #%{customdata[0]}<br>" + f"{x_label}: " + "%{customdata[1]}<br>" + f"{y_label}: " + "%{customdata[2]}<br>Probabilité: %{customdata[3]:.1%}<extra></extra>"),
                name="Autres clients"
            ))
        if not df_sel.empty:
            fig.add_trace(go.Scatter(
                x=df_sel["x_plot"], y=df_sel["y_plot"], mode="markers+text",
                marker=dict(size=13, symbol="diamond", line=dict(width=1, color="black"),
                            color=[COLORBLIND_FRIENDLY_PALETTE.get('accepted','#2ca02c') if d == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused','#d62728') for d in df_sel["decision"]]),
                text=[f"#{int(c)}" for c in df_sel["client_id"]],
                textposition="top center",
                customdata=np.stack([df_sel["client_id"], df_sel.get("x_display", df_sel["x_raw"]), df_sel.get("y_display", df_sel["y_raw"]), df_sel["probability"]], axis=-1),
                hovertemplate=("Client #%{customdata[0]}<br>" + f"{x_label}: " + "%{customdata[1]}<br>" + f"{y_label}: " + "%{customdata[2]}<br>Probabilité: %{customdata[3]:.1%}<extra></extra>"),
                name="Clients sélectionnés"
            ))
        if not df_ref.empty:
            r = df_ref.iloc[0]
            fig.add_trace(go.Scatter(
                x=[r["x_plot"]], y=[r["y_plot"]], mode="markers+text",
                marker=dict(size=20, symbol="star", color="black", line=dict(width=2, color='white')),
                text=[f"Réf #{int(r['client_id'])}"], textposition="bottom center",
                hovertemplate=(f"Client #{int(r['client_id'])}<br>{x_label}: {r.get('x_display', r['x_raw'])}<br>{y_label}: {r.get('y_display', r['y_raw'])}<br>Probabilité: {r['probability']:.1%}<extra></extra>"),
                name="Client référence"
            ))

        fig.update_layout(title=pair_labels[choice_key], xaxis_title=x_label, yaxis_title=y_label, template="simple_white", height=600)

        # Option simple : montrer une droite de tendance purement visuelle
        if st.checkbox("Afficher droite de tendance (aide visuelle)", value=False):
            try:
                lr = LinearRegression().fit(df_prep[["x_plot"]], df_prep["y_plot"])
                x_line = np.linspace(df_prep["x_plot"].min(), df_prep["x_plot"].max(), 200)
                y_line = lr.predict(x_line.reshape(-1,1)).flatten()
                fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="black", dash="dash"), name="Tendance"))
            except Exception:
                st.info("Impossible de tracer la droite de tendance sur ces données.")

        st.plotly_chart(fig, width='stretch')

        # ✅ NOUVELLES interprétations métier plus pertinentes et axes corrigés
        st.markdown("#### 💡 Interprétation métier")
        if choice_key == "employed_vs_age":
            st.info("**Analyse Stabilité Professionnelle :** Jeunes avec longue ancienneté = profils stables et valorisés. Seniors avec faible ancienneté = reconversion ou instabilité professionnelle.")
        elif choice_key == "credit_vs_price":
            st.info("**Analyse Financement :** Points sur la diagonale = crédit égal au prix du bien. Au-dessus = sur-financement (frais inclus), en-dessous = apport personnel important.")
        elif choice_key == "income_vs_annuity":
            st.info("**Analyse Capacité de Paiement :** Ratio fondamental pour évaluer l'endettement. Mensualité > 33% des revenus mensuels = sur-endettement potentiel nécessitant vigilance.")
        elif choice_key == "age_vs_credit":
            st.info("**Analyse Cycle de Vie :** Jeunes avec gros crédits = premiers achats (profils risqués). Seniors avec crédits élevés = potentiel patrimonial établi.")
        elif choice_key == "income_vs_credit":
            st.info("**Analyse Exposition/Capacité :** Ratio crédit/revenu critique pour l'acceptation. Crédits > 5x les revenus annuels = exposition élevée nécessitant conditions particulières.")
        elif choice_key == "education_vs_credit":
            st.info("**Analyse Socio-économique :** Niveau d'éducation généralement corrélé aux montants de crédit acceptés. Utile pour la segmentation et personnalisation des offres commerciales.")

# ---------- Explications simples ----------
if st.button("Explication des caractéristiques sélectionnées", key="exp_feat"):
    for f in [x_feature, y_feature]:
        if f:
            desc = FEATURE_DESCRIPTIONS.get(f, None)
            if desc:
                st.markdown(f"**{f}** — {desc}")
            else:
                st.markdown(f"**{f}** — Pas de description disponible.")

# ---------- Comparaison des risques ----------
st.subheader("Comparaison des risques de défaut")
try:
    threshold = client_data[list(client_data.keys())[0]]["prediction"].get("threshold", 0.52)
except Exception:
    threshold = 0.52
sorted_clients = sorted([(cid, dd["prediction"].get("probability", 0)) for cid, dd in client_data.items()], key=lambda x: x[1])

fig = go.Figure()
risk_zones = [
    {"name": "RISQUE TRÈS FAIBLE", "min": 0, "max": 0.2, "color": "rgba(1,133,113,0.4)"},
    {"name": "RISQUE FAIBLE", "min": 0.2, "max": 0.4, "color": "rgba(1,133,113,0.6)"},
    {"name": "RISQUE MODÉRÉ", "min": 0.4, "max": threshold, "color": "rgba(1,133,113,0.8)"},
    {"name": "RISQUE ÉLEVÉ", "min": threshold, "max": 0.7, "color": "rgba(166,97,26,0.6)"},
    {"name": "RISQUE TRÈS ÉLEVÉ", "min": 0.7, "max": 1, "color": "rgba(166,97,26,0.8)"}
]
for zone in risk_zones:
    fig.add_shape(type="rect", x0=zone["min"], x1=zone["max"], y0=-1, y1=len(sorted_clients), fillcolor=zone["color"], line=dict(width=0), layer="below")
fig.add_shape(type="line", x0=threshold, x1=threshold, y0=-2, y1=len(sorted_clients), line=dict(color="black", width=2, dash="dash"))
fig.add_annotation(x=threshold, y=-2.5, text=f"SEUIL: {threshold:.2f}", showarrow=False)

for i, (cid, prob) in enumerate(sorted_clients):
    decision = "ACCEPTÉ" if prob < threshold else "REFUSÉ"
    color = COLORBLIND_FRIENDLY_PALETTE.get('accepted','#2ca02c') if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused','#d62728')
    fig.add_trace(go.Bar(y=[i], x=[prob], orientation='h', marker=dict(color=color, line=dict(color='rgba(0,0,0,0.5)', width=1)), hovertemplate=f"Client #{cid}<br>Probabilité: {prob:.1%}<br>Décision: {decision}<extra></extra>", showlegend=False))
    fig.add_annotation(x=-0.05, y=i, text=f"#{cid}", showarrow=False, xanchor="right")
    pos_x = prob + 0.03 if abs(prob - threshold) > 0.05 else prob + 0.06
    fig.add_annotation(x=pos_x, y=i, text=f"{prob:.1%}", showarrow=False, xanchor="left")

fig.update_layout(title="Comparaison des risques de défaut par client", height=max(300, 150 + 40 * len(sorted_clients)), xaxis=dict(title="Probabilité de défaut", range=[-0.1, 1.05], tickformat=".0%"), yaxis=dict(showticklabels=False, title="Clients (classés par risque croissant)"))
st.plotly_chart(fig, width='stretch')

# ---------- Footer ----------
st.markdown("""
<hr>
<div style="text-align:center; padding:8px; border-radius:6px; background:#f8f9fa;">
    <strong>Comparaison de clients</strong> — Analyses basées sur des données concrètes et exploitables métier (revenus, âge, crédits). Les scores externes flous (EXT_SOURCE) ont été remplacés par des indicateurs plus pertinents pour les conseillers.
</div>
""", unsafe_allow_html=True)
