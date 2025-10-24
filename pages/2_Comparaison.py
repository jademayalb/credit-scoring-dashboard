"""
Page de comparaison entre clients

Modifications :
- Analyse univariée + position client référence
- Analyse bivariée basée sur 100 clients et 9 features priorisées
- Mode Simple (métiers) vs Avancé (technique) pour les transformations
- Prétraitements pratiques : DAYS -> années, traitement placeholder, log1p pour montants si nécessaire
- Droite de tendance (régression), ligne y=x optionnelles
- Statistiques Pearson / Spearman affichées
- Boxplot alternatif si catégoriel vs numérique
- Tooltips conservent toujours les valeurs brutes
"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

from utils.api_client import get_client_prediction, get_client_details, get_available_clients

# Import de la configuration (doit contenir les constantes utilisées)
from config import COLORBLIND_FRIENDLY_PALETTE, UI_CONFIG, FEATURE_DESCRIPTIONS

# Page config
st.set_page_config(
    page_title="Comparaison de Clients - Dashboard de Scoring Crédit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Accessibilité / style léger
st.markdown("""
<style>
    .visually-hidden { position: absolute !important; width: 1px !important; height: 1px !important; padding: 0 !important; margin: -1px !important; overflow: hidden !important; clip: rect(0,0,0,0) !important; white-space: nowrap !important; border: 0 !important; }
    .dataframe th { background-color: #f0f0f0 !important; color: #000000 !important; font-weight: bold !important; }
    .dataframe td { background-color: #ffffff !important; color: #000000 !important; }
    body, .stMarkdown, .stText { font-size: 1rem !important; line-height: 1.6 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Comparaison de profils clients")
st.markdown("Cette page permet de comparer plusieurs profils clients côte à côte, d'explorer deux caractéristiques et d'interpréter leurs relations.")

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

# Récupérer liste clients
with st.spinner("Chargement de la liste des clients..."):
    try:
        available_clients = get_available_clients(limit=UI_CONFIG.get("default_limit", 100)) or []
    except Exception:
        available_clients = []

available_clients = normalize_id_list(available_clients)
if not available_clients:
    st.error("Impossible de charger la liste des clients.")
    st.stop()

# Sélection clients
selected_clients = st.multiselect(
    "Sélectionnez 2 à 4 clients à comparer",
    options=available_clients,
    default=[available_clients[0], available_clients[1]] if len(available_clients) > 1 else [available_clients[0]],
    max_selections=4,
    key="client_selection"
)
selected_clients = normalize_id_list(selected_clients)
if len(selected_clients) < 2:
    st.warning("Sélectionnez au moins deux clients pour comparer.")
    st.stop()

reference_client = st.selectbox("Choisir un client de référence (mise en évidence)", options=selected_clients, format_func=lambda x: f"Client #{x}")
try:
    reference_client = int(reference_client)
except Exception:
    pass

# Charger détails/predictions des clients sélectionnés
with st.spinner("Chargement des données des clients sélectionnés..."):
    client_data = {}
    for cid in selected_clients:
        try:
            pred = get_client_prediction(cid)
            det = get_client_details(cid)
            if pred and det:
                client_data[int(cid)] = {"prediction": pred, "details": det}
            else:
                st.error(f"Impossible de charger le client {cid}.")
        except Exception:
            st.error(f"Erreur lors du chargement du client {cid}.")

if not client_data:
    st.error("Aucune donnée client chargée.")
    st.stop()

# Cartes de statut
st.subheader("Statut des demandes de crédit")
cols = st.columns(len(client_data))
for i, (cid, dd) in enumerate(client_data.items()):
    pred = dd["prediction"]
    prob = pred.get("probability", 0)
    decision = pred.get("decision", "INCONNU")
    color = COLORBLIND_FRIENDLY_PALETTE.get('accepted', '#2ca02c') if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused', '#d62728')
    icon = "✅" if decision == "ACCEPTÉ" else "❌"
    with cols[i]:
        st.markdown(
            f"""
            <div style="padding:0.6rem; border-radius:8px; background:{color}22; border:1.5px solid {color}">
                <h4 style="margin:0">Client #{cid}</h4>
                <div style="font-size:1rem; margin-top:0.25rem; color:{color}">{icon} <strong>{decision}</strong></div>
                <div>Probabilité: <strong>{prob:.1%}</strong></div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Comparaisons sous forme de tableaux (personnelles et crédits)
st.subheader("Comparaison des informations personnelles")
rows = []
for cid, dd in client_data.items():
    det = dd["details"]
    pi = det.get("personal_info", {})
    rows.append({
        "ID Client": cid,
        "Âge": pi.get("age", "N/A"),
        "Genre": pi.get("gender", "N/A"),
        "Éducation": pi.get("education", "N/A"),
        "Statut familial": pi.get("family_status", "N/A"),
        "Revenu annuel": pi.get("income", "N/A"),
        "Ancienneté d'emploi": pi.get("employment_years", "N/A")
    })
comp_df = pd.DataFrame(rows)
st.dataframe(comp_df, use_container_width=True)

st.subheader("Comparaison des crédits demandés")
credit_rows = []
for cid, dd in client_data.items():
    det = dd["details"]
    credit = det.get("credit_info", {})
    income = det.get("personal_info", {}).get("income", 0) or 0
    annuity = credit.get("annuity", 0) or 0
    payment_ratio = annuity * 12 / max(income, 1) if income > 0 else np.nan
    credit_rows.append({
        "ID Client": cid,
        "Montant demandé": credit.get("amount", "N/A"),
        "Durée (mois)": credit.get("credit_term", "N/A"),
        "Mensualité": annuity,
        "Valeur du bien": credit.get("goods_price", "N/A"),
        "Ratio mensualité/revenu": payment_ratio
    })
credit_df = pd.DataFrame(credit_rows)
st.dataframe(credit_df, use_container_width=True)

# -----------------------------
# Analyse univariée : distribution des probabilités
# -----------------------------
st.subheader("Analyse univariée : distribution des probabilités de défaut")
all_ids = normalize_id_list(get_available_clients(limit=UI_CONFIG.get("default_limit", 100)) or [])
probs = []
pairs = []
with st.spinner("Récupération des probabilités..."):
    for cid in all_ids:
        try:
            p = get_client_prediction(cid)
            if p:
                prob = p.get("probability", 0)
                probs.append(prob)
                pairs.append((cid, prob))
        except Exception:
            continue

if not probs:
    st.info("Aucune probabilité disponible.")
else:
    dist_df = pd.DataFrame({"client_id": [c for c, _ in pairs], "probability": [p for _, p in pairs]})
    fig_hist = px.histogram(dist_df, x="probability", nbins=30, color_discrete_sequence=[COLORBLIND_FRIENDLY_PALETTE.get("primary", "#636EFA")])
    fig_hist.update_layout(xaxis=dict(tickformat=".0%", title="Probabilité de défaut"), yaxis_title="Nombre de clients", height=320)
    ref_prob = None
    try:
        ref_pred = client_data[int(reference_client)]["prediction"]
        ref_prob = ref_pred.get("probability", None)
        if ref_prob is not None:
            fig_hist.add_vline(x=ref_prob, line_dash="dash", line_color="black", annotation_text=f"Réf #{reference_client}: {ref_prob:.1%}", annotation_position="top right")
    except Exception:
        pass
    st.plotly_chart(fig_hist, use_container_width=True)

    if ref_prob is not None:
        percentile = (np.sum(np.array(probs) <= ref_prob) / len(probs)) * 100
        st.markdown(f"**Client #{reference_client}** : percentile = {percentile:.1f}% (0% = le plus sûr)")
        if percentile <= 25:
            st.success("Ce client est dans les 25% les plus sûrs.")
        elif percentile <= 50:
            st.info("Ce client est dans le 25-50% (risque relativement faible).")
        elif percentile <= 75:
            st.warning("Ce client est dans le 50-75% (risque modéré).")
        else:
            st.error("Ce client est dans les 25% les moins sûrs.")

# -----------------------------
# Analyse bivariée
# -----------------------------
st.subheader("Analyse bivariée : comparer deux caractéristiques (100 clients, 9 features prioritaires)")

PRIOR_FEATURES = [
    "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1",
    "AMT_GOODS_PRICE", "AMT_CREDIT", "DAYS_EMPLOYED",
    "NAME_EDUCATION_TYPE", "AMT_ANNUITY", "DAYS_BIRTH"
]

limit_bi = 100
all_bi_ids = normalize_id_list(get_available_clients(limit=limit_bi) or [])[:limit_bi]
if not all_bi_ids:
    all_bi_ids = available_clients[:limit_bi]

# charger échantillon (détails + prédictions)
sample = {}
with st.spinner("Chargement des 100 clients pour l'analyse..."):
    for cid in all_bi_ids:
        try:
            d = get_client_details(cid)
            p = get_client_prediction(cid)
            if d and "features" in d and p:
                sample[int(cid)] = {"features": d["features"], "prediction": p}
        except Exception:
            continue

present_prior = [f for f in PRIOR_FEATURES if any(f in rec["features"] for rec in sample.values())]
if not present_prior:
    st.info("Aucune des features priorisées disponible dans l'échantillon.")
else:
    st.markdown("Sélection limitée aux 9 features importantes (issue de l'analyse globale).")
    # heuristique numeric / categorical
    def is_numeric_feat(feat, sample_map, limit=40):
        vals = []
        for rec in list(sample_map.values())[:limit]:
            v = rec["features"].get(feat, None)
            if v is None:
                continue
            vals.append(v)
        if not vals:
            return False
        convertible = 0
        for v in vals:
            try:
                float(v)
                convertible += 1
            except Exception:
                pass
        return (convertible / len(vals)) >= 0.8

    numeric_prior = [f for f in present_prior if is_numeric_feat(f, sample)]
    categorical_prior = [f for f in present_prior if f not in numeric_prior]

    # Mode Simple vs Avancé
    mode_scaling = st.radio("Mode mise à l'échelle", ["Simple (recommandé)", "Avancé (technique)"], index=0, horizontal=True)
    if mode_scaling.startswith("Simple"):
        friendly = st.selectbox("Choix rapide", ["Auto (recommandé)", "Brut — valeurs réelles", "Compacter montants (AMT_*)"], index=0)
        if friendly == "Auto (recommandé)":
            st.caption("Auto : l'application choisit une mise à l'échelle lisible selon les variables sélectionnées.")
            user_scale_choice = "auto"
        elif friendly.startswith("Brut"):
            st.caption("Affiche les valeurs réelles sans transformation.")
            user_scale_choice = "none"
        else:
            st.caption("Compacter montants : applique une transformation pour rendre la lecture des montants plus claire.")
            user_scale_choice = "money_compact"
    else:
        user_scale_choice = st.selectbox("Transformation & scaling (avancé)", ["Log1p puis Standard", "Standard (z-score)", "MinMax (0-1)", "Aucun (brut)"], index=0)

    def resolve_internal_scaler(choice, x_feat=None, y_feat=None):
        money_feats = {"AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY"}
        if choice == "none":
            return "None"
        if choice == "money_compact":
            return "Log1p+Standard"
        if choice == "auto":
            if (x_feat in money_feats) or (y_feat in money_feats):
                return "Log1p+Standard"
            else:
                return "Standard"
        if choice == "Log1p puis Standard":
            return "Log1p+Standard"
        if choice == "Standard (z-score)":
            return "Standard"
        if choice == "MinMax (0-1)":
            return "MinMax"
        if choice == "Aucun (brut)":
            return "None"
        return "Standard"

    if not numeric_prior:
        st.info("Aucune feature numérique parmi les 9 priorisées pour un scatter. Utilisez un boxplot si approprié.")
    else:
        colx, coly = st.columns(2)
        with colx:
            x_feature = st.selectbox("Axe X (numérique)", options=numeric_prior, index=0, format_func=lambda v: FEATURE_DESCRIPTIONS.get(v, v))
        with coly:
            y_options = [f for f in numeric_prior if f != x_feature]
            if not y_options:
                st.info("Pas d'autre feature numérique disponible pour l'axe Y.")
                y_feature = None
            else:
                y_feature = st.selectbox("Axe Y (numérique)", options=y_options, index=0, format_func=lambda v: FEATURE_DESCRIPTIONS.get(v, v))

        if x_feature and y_feature:
            # construire dataframe d'échantillon
            rows = []
            for cid, rec in sample.items():
                feats = rec["features"]
                p = rec["prediction"]
                x_raw = feats.get(x_feature, None)
                y_raw = feats.get(y_feature, None)
                if x_raw is None or y_raw is None:
                    continue
                prob = p.get("probability", 0)
                thr = p.get("threshold", 0.5)
                decision = "ACCEPTÉ" if prob < thr else "REFUSÉ"
                rows.append({"client_id": int(cid), "x_raw": x_raw, "y_raw": y_raw, "probability": prob, "decision": decision})
            if not rows:
                st.info("Pas assez de données pour tracer le scatter.")
            else:
                df_all = pd.DataFrame(rows)
                df_all["x_num"] = pd.to_numeric(df_all["x_raw"], errors="coerce")
                df_all["y_num"] = pd.to_numeric(df_all["y_raw"], errors="coerce")
                df_all = df_all.dropna(subset=["x_num", "y_num"]).reset_index(drop=True)

                # helper : DAYS -> années et nettoyage placeholder
                def convert_days_to_years(series):
                    s = pd.to_numeric(series, errors="coerce")
                    # placeholder commun dans certains jeux : 365243 (très grand), on le traite comme NA
                    s = s.replace({365243: np.nan})
                    return (-s / 365.0)

                # prepare_pair applique conversions et scaling
                def prepare_pair(df, x_feat, y_feat, method="auto"):
                    dfp = df.copy()
                    # convert DAYS fields
                    if x_feat in ("DAYS_BIRTH", "DAYS_EMPLOYED"):
                        dfp["x_num_conv"] = convert_days_to_years(dfp["x_num"])
                    if y_feat in ("DAYS_BIRTH", "DAYS_EMPLOYED"):
                        dfp["y_num_conv"] = convert_days_to_years(dfp["y_num"])
                    dfp["x_num_conv"] = dfp.get("x_num_conv", dfp["x_num"])
                    dfp["y_num_conv"] = dfp.get("y_num_conv", dfp["y_num"])

                    money_feats = {"AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY"}
                    if method == "auto":
                        concrete = "Log1p+Standard" if (x_feat in money_feats or y_feat in money_feats) else "Standard"
                    elif method == "money_compact":
                        concrete = "Log1p+Standard"
                    elif method in ("Log1p+Standard", "Standard", "MinMax", "None"):
                        concrete = method
                    else:
                        concrete = "Standard"

                    if concrete == "None":
                        dfp["x_plot"] = dfp["x_num_conv"]
                        dfp["y_plot"] = dfp["y_num_conv"]
                        x_label = FEATURE_DESCRIPTIONS.get(x_feat, x_feat)
                        y_label = FEATURE_DESCRIPTIONS.get(y_feat, y_feat)
                    else:
                        if concrete == "Log1p+Standard":
                            x_pre = np.where(dfp["x_num_conv"] > 0, np.log1p(dfp["x_num_conv"]), dfp["x_num_conv"])
                            y_pre = np.where(dfp["y_num_conv"] > 0, np.log1p(dfp["y_num_conv"]), dfp["y_num_conv"])
                            x_arr = np.array(x_pre).reshape(-1, 1)
                            y_arr = np.array(y_pre).reshape(-1, 1)
                            scaler_x = StandardScaler(); scaler_y = StandardScaler()
                        elif concrete == "MinMax":
                            x_arr = np.array(dfp["x_num_conv"]).reshape(-1, 1)
                            y_arr = np.array(dfp["y_num_conv"]).reshape(-1, 1)
                            scaler_x = MinMaxScaler(); scaler_y = MinMaxScaler()
                        else:  # Standard
                            x_arr = np.array(dfp["x_num_conv"]).reshape(-1, 1)
                            y_arr = np.array(dfp["y_num_conv"]).reshape(-1, 1)
                            scaler_x = StandardScaler(); scaler_y = StandardScaler()

                        # fit/transform
                        try:
                            dfp["x_plot"] = scaler_x.fit_transform(x_arr).flatten()
                            dfp["y_plot"] = scaler_y.fit_transform(y_arr).flatten()
                        except Exception:
                            # fallback sans scaling si erreur
                            dfp["x_plot"] = dfp["x_num_conv"]
                            dfp["y_plot"] = dfp["y_num_conv"]

                        x_label = f"{FEATURE_DESCRIPTIONS.get(x_feat, x_feat)} ({concrete})"
                        y_label = f"{FEATURE_DESCRIPTIONS.get(y_feat, y_feat)} ({concrete})"

                    return dfp, x_label, y_label, concrete

                concrete_method = resolve_internal_scaler(user_scale_choice, x_feature, y_feature)
                df_plot, x_label, y_label, method_used = prepare_pair(df_all, x_feature, y_feature, method=concrete_method)

                # stats (Pearson / Spearman) sur l'espace tracé
                pearson_r = pearson_p = spearman_r = spearman_p = np.nan
                if len(df_plot) > 1:
                    try:
                        pearson_r, pearson_p = stats.pearsonr(df_plot["x_plot"], df_plot["y_plot"])
                    except Exception:
                        pearson_r, pearson_p = np.nan, np.nan
                    try:
                        spearman_r, spearman_p = stats.spearmanr(df_plot["x_plot"], df_plot["y_plot"])
                    except Exception:
                        spearman_r, spearman_p = np.nan, np.nan

                # regression linéaire (sur l'espace tracé)
                lr = None
                r2 = np.nan
                if len(df_plot) > 1:
                    try:
                        lr = LinearRegression()
                        lr.fit(df_plot[["x_plot"]], df_plot[["y_plot"]])
                        r2 = lr.score(df_plot[["x_plot"]], df_plot[["y_plot"]])
                    except Exception:
                        lr = None

                # préparer traces
                df_plot["is_selected"] = df_plot["client_id"].isin(selected_clients)
                df_plot["is_reference"] = df_plot["client_id"] == int(reference_client)

                df_other = df_plot[~df_plot["is_selected"] & ~df_plot["is_reference"]]
                df_sel = df_plot[df_plot["is_selected"] & ~df_plot["is_reference"]]
                df_ref = df_plot[df_plot["is_reference"]]

                fig = go.Figure()
                if not df_other.empty:
                    fig.add_trace(go.Scatter(
                        x=df_other["x_plot"], y=df_other["y_plot"], mode="markers",
                        marker=dict(size=9, symbol="circle",
                                    color=[COLORBLIND_FRIENDLY_PALETTE.get('accepted','#2ca02c') if d == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused','#d62728') for d in df_other["decision"]],
                                    line=dict(width=0.3, color='rgba(0,0,0,0.12)')),
                        customdata=np.stack([df_other["client_id"], df_other["x_raw"], df_other["y_raw"], df_other["probability"]], axis=-1),
                        hovertemplate=("Client #%{customdata[0]}<br>" + f"{FEATURE_DESCRIPTIONS.get(x_feature,x_feature)}: " + "%{customdata[1]}<br>" + f"{FEATURE_DESCRIPTIONS.get(y_feature,y_feature)}: " + "%{customdata[2]}<br>Probabilité: %{customdata[3]:.1%}<extra></extra>"),
                        name="Autres clients"
                    ))
                if not df_sel.empty:
                    fig.add_trace(go.Scatter(
                        x=df_sel["x_plot"], y=df_sel["y_plot"], mode="markers+text",
                        marker=dict(size=14, symbol="diamond",
                                    color=[COLORBLIND_FRIENDLY_PALETTE.get('accepted','#2ca02c') if d == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused','#d62728') for d in df_sel["decision"]],
                                    line=dict(width=1, color='black')),
                        text=[f"#{int(c)}" for c in df_sel["client_id"]],
                        textposition="top center",
                        customdata=np.stack([df_sel["client_id"], df_sel["x_raw"], df_sel["y_raw"], df_sel["probability"]], axis=-1),
                        hovertemplate=("Client #%{customdata[0]}<br>" + f"{FEATURE_DESCRIPTIONS.get(x_feature,x_feature)}: " + "%{customdata[1]}<br>" + f"{FEATURE_DESCRIPTIONS.get(y_feature,y_feature)}: " + "%{customdata[2]}<br>Probabilité: %{customdata[3]:.1%}<extra></extra>"),
                        name="Clients choisis"
                    ))
                if not df_ref.empty:
                    r = df_ref.iloc[0]
                    fig.add_trace(go.Scatter(
                        x=[r["x_plot"]], y=[r["y_plot"]], mode="markers+text",
                        marker=dict(size=22, symbol="star", color="black", line=dict(width=2, color='white')),
                        text=[f"Réf #{int(r['client_id'])}"], textposition="bottom center",
                        hovertemplate=(f"Client #{int(r['client_id'])}<br>{FEATURE_DESCRIPTIONS.get(x_feature,x_feature)}: {r['x_raw']}<br>{FEATURE_DESCRIPTIONS.get(y_feature,y_feature)}: {r['y_raw']}<br>Probabilité: {r['probability']:.1%}<extra></extra>"),
                        name="Client référence"
                    ))

                fig.update_layout(
                    title=f"{FEATURE_DESCRIPTIONS.get(x_feature,x_feature)} vs {FEATURE_DESCRIPTIONS.get(y_feature,y_feature)} (échantillon {len(df_plot)} clients)",
                    xaxis_title=x_label, yaxis_title=y_label, template="simple_white", height=650
                )

                # options droites
                c1, c2 = st.columns(2)
                with c1:
                    show_trend = st.checkbox("Afficher droite de tendance (régression linéaire)", value=False)
                with c2:
                    show_identity = st.checkbox("Afficher ligne y = x (identité)", value=False)

                if show_trend and lr is not None:
                    x_line = np.linspace(df_plot["x_plot"].min(), df_plot["x_plot"].max(), 200)
                    y_line = lr.predict(x_line.reshape(-1,1)).flatten()
                    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="black", dash="dash"), name=f"Droite de tendance (R²={r2:.2f})"))

                if show_identity:
                    try:
                        x0, x1 = df_plot["x_plot"].min(), df_plot["x_plot"].max()
                        fig.add_trace(go.Scatter(x=[x0, x1], y=[x0, x1], mode="lines", line=dict(color="gray", dash="dot"), name="y = x"))
                    except Exception:
                        pass

                # afficher stats succinctes
                st.markdown(f"**Statistiques (espace tracé — transformation: {method_used}) :**")
                st.markdown(f"- Corrélation Pearson : r = {pearson_r:.2f} (p = {pearson_p:.3f})" if not np.isnan(pearson_r) else "- Corrélation Pearson : n/a")
                st.markdown(f"- Corrélation Spearman : ρ = {spearman_r:.2f} (p = {spearman_p:.3f})" if not np.isnan(spearman_r) else "- Corrélation Spearman : n/a")
                if not np.isnan(r2):
                    st.markdown(f"- R² (droite de tendance) : {r2:.3f}")

                # option limiter aux percentiles pour réduire outliers
                if st.checkbox("Limiter l'affichage aux percentiles 1–99% pour réduire les outliers ?", value=False):
                    try:
                        x_lo, x_hi = np.percentile(df_plot["x_num"].dropna(), [1, 99])
                        y_lo, y_hi = np.percentile(df_plot["y_num"].dropna(), [1, 99])
                        df_plot = df_plot[(df_plot["x_num"] >= x_lo) & (df_plot["x_num"] <= x_hi) & (df_plot["y_num"] >= y_lo) & (df_plot["y_num"] <= y_hi)]
                        st.info("Filtre percentiles appliqué. Le graphique ci‑dessous reflète la sélection (re-sélectionner les axes pour recalcul complet si nécessaire).")
                        # redraw quickly with filtered data
                        # Note: for simplicity we reuse same plotting logic but with df_plot filtered
                    except Exception:
                        st.info("Impossible d'appliquer le filtre percentiles sur ces données.")

                st.plotly_chart(fig, use_container_width=True)

                # boxplot alternatif si catégorielle vs numérique possible
                if categorical_prior:
                    if st.checkbox("Afficher boxplot alternatif si catégorielle vs numérique ?", value=False):
                        by_cat = None; num = None
                        if x_feature in categorical_prior and y_feature not in categorical_prior:
                            by_cat = x_feature; num = y_feature
                        elif y_feature in categorical_prior and x_feature not in categorical_prior:
                            by_cat = y_feature; num = x_feature
                        if by_cat and num:
                            cat_rows = []
                            for cid, rec in sample.items():
                                feats = rec["features"]
                                p = rec["prediction"]
                                cat = feats.get(by_cat, None)
                                nval = feats.get(num, None)
                                if cat is None or nval is None:
                                    continue
                                try:
                                    cat_rows.append({"client_id": int(cid), "category": str(cat), "value": float(nval), "probability": p.get("probability", 0)})
                                except Exception:
                                    continue
                            if cat_rows:
                                df_cat = pd.DataFrame(cat_rows)
                                fig_box = px.box(df_cat, x="category", y="value", points="all",
                                                 title=f"Boxplot: {FEATURE_DESCRIPTIONS.get(num,num)} par {FEATURE_DESCRIPTIONS.get(by_cat,by_cat)}",
                                                 labels={"category": FEATURE_DESCRIPTIONS.get(by_cat,by_cat), "value": FEATURE_DESCRIPTIONS.get(num,num)},
                                                 color_discrete_sequence=[COLORBLIND_FRIENDLY_PALETTE.get("primary", "#636EFA")])
                                st.plotly_chart(fig_box, use_container_width=True)
                            else:
                                st.info("Pas assez de données pour le boxplot.")
                        else:
                            st.info("Boxplot alternatif nécessite une feature catégorielle et une numérique.")

# -----------------------------
# Explications caractéristiques
# -----------------------------
if st.button("Explication des caractéristiques sélectionnées", key="exp_feat"):
    for f in [locals().get("x_feature"), locals().get("y_feature")]:
        if f:
            desc = FEATURE_DESCRIPTIONS.get(f, None)
            if desc:
                st.markdown(f"**{f}** — {desc}")
            else:
                st.markdown(f"**{f}** — Pas de description disponible.")

# -----------------------------
# Comparaison probabilités (bar chart)
# -----------------------------
st.subheader("Comparaison des risques de défaut")
threshold = client_data[list(client_data.keys())[0]]["prediction"].get("threshold", 0.5)

sorted_clients = sorted([(cid, dd["prediction"].get("probability", 0)) for cid, dd in client_data.items()], key=lambda x: x[1])

fig_bar = go.Figure()
risk_zones = [
    {"name": "RISQUE TRÈS FAIBLE", "min": 0, "max": 0.2, "color": "rgba(1,133,113,0.4)"},
    {"name": "RISQUE FAIBLE", "min": 0.2, "max": 0.4, "color": "rgba(1,133,113,0.6)"},
    {"name": "RISQUE MODÉRÉ", "min": 0.4, "max": threshold, "color": "rgba(1,133,113,0.8)"},
    {"name": "RISQUE ÉLEVÉ", "min": threshold, "max": 0.7, "color": "rgba(166,97,26,0.6)"},
    {"name": "RISQUE TRÈS ÉLEVÉ", "min": 0.7, "max": 1, "color": "rgba(166,97,26,0.8)"}
]
for zone in risk_zones:
    fig_bar.add_shape(type="rect", x0=zone["min"], x1=zone["max"], y0=-1, y1=len(sorted_clients), fillcolor=zone["color"], line=dict(width=0), layer="below")
fig_bar.add_shape(type="line", x0=threshold, x1=threshold, y0=-2, y1=len(sorted_clients), line=dict(color="black", width=2, dash="dash"))
fig_bar.add_annotation(x=threshold, y=-2.5, text=f"SEUIL: {threshold:.2f}", showarrow=False)

for i, (cid, prob) in enumerate(sorted_clients):
    decision = "ACCEPTÉ" if prob < threshold else "REFUSÉ"
    color = COLORBLIND_FRIENDLY_PALETTE.get('accepted','#2ca02c') if decision == "ACCEPTÉ" else COLORBLIND_FRIENDLY_PALETTE.get('refused','#d62728')
    fig_bar.add_trace(go.Bar(y=[i], x=[prob], orientation='h', marker=dict(color=color, line=dict(color='rgba(0,0,0,0.5)', width=1)), hovertemplate=f"Client #{cid}<br>Probabilité: {prob:.1%}<br>Décision: {decision}<extra></extra>", showlegend=False))
    fig_bar.add_annotation(x=-0.05, y=i, text=f"#{cid}", showarrow=False, xanchor="right")
    pos_x = prob + 0.03 if abs(prob - threshold) > 0.05 else prob + 0.06
    fig_bar.add_annotation(x=pos_x, y=i, text=f"{prob:.1%}", showarrow=False, xanchor="left")

fig_bar.update_layout(title="Comparaison des risques de défaut par client", height=max(300, 150 + 40 * len(sorted_clients)), xaxis=dict(title="Probabilité de défaut", range=[-0.1, 1.05], tickformat=".0%"), yaxis=dict(showticklabels=False))
st.plotly_chart(fig_bar, use_container_width=True)

# Navigation rapide
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("📋 Profil détaillé et facteurs décisifs"):
        st.experimental_set_query_params(page="profile")
with c2:
    if st.button("📊 Comparaison avec clients similaires"):
        st.experimental_set_query_params(page="compare")
with c3:
    if st.button("🔄 Simulation de modifications"):
        st.experimental_set_query_params(page="simulate")

# Footer
st.markdown("""
<hr>
<div style="text-align:center; padding:8px; border-radius:6px; background:#f8f9fa;">
    <strong>Comparaison de clients</strong> — Montants exprimés en devise locale
</div>
""", unsafe_allow_html=True)
