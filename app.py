"""Application Streamlit — Support de présentation orale DVF.

Lancement : streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Prédiction DVF — ML",
    page_icon="🏠",
    layout="wide",
)

# ── Navigation ──────────────────────────────────────────────────────────────
PAGES = [
    "🏠 Accueil & Contexte",
    "📊 Données & Nettoyage",
    "📈 Régression — Prix",
    "🏷️ Classification — Type de bien",
    "🔮 Démo Régression",
    "🔍 Démo Classification",
]
page = st.sidebar.radio("Navigation", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Support oral — DVF 2024-2025")


# ── Loaders ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des données…")
def load_data():
    data_dir = Path("datasets")
    csv_files = sorted(data_dir.glob("full_202[45].csv"))
    if not csv_files:
        return None
    COLS = [
        "id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere",
        "code_departement", "type_local", "surface_reelle_bati",
        "nombre_pieces_principales", "nombre_lots", "lot1_surface_carrez",
        "longitude", "latitude",
    ]
    frames = [pd.read_csv(f, usecols=COLS, low_memory=False) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["nature_mutation"] == "Vente") & (df["type_local"] == "Appartement")].copy()
    df = df.dropna(subset=["valeur_fonciere"])
    df = df[(df["valeur_fonciere"] >= 10_000) & (df["valeur_fonciere"] <= 10_000_000)]
    df = df[(df["surface_reelle_bati"] > 0) & (df["surface_reelle_bati"] <= 500)]
    df = df.sort_values("surface_reelle_bati", ascending=False).drop_duplicates("id_mutation", keep="first")
    df = df.dropna(subset=["longitude", "latitude"])
    df["nombre_pieces_principales"] = df["nombre_pieces_principales"].fillna(
        df["nombre_pieces_principales"].median()
    )
    df["date_mutation"] = pd.to_datetime(df["date_mutation"])
    df["annee"]     = df["date_mutation"].dt.year
    df["mois"]      = df["date_mutation"].dt.month
    df["trimestre"] = df["date_mutation"].dt.quarter
    df["prix_m2"]   = df["valeur_fonciere"] / df["surface_reelle_bati"]
    df["surface_carrez"]    = df["lot1_surface_carrez"].fillna(0)
    df["surface_par_piece"] = df["surface_reelle_bati"] / df["nombre_pieces_principales"].clip(lower=1)
    dept_mean = df.groupby("code_departement")["valeur_fonciere"].mean()
    df["dept_prix_moyen"] = df["code_departement"].map(dept_mean)
    return df


@st.cache_resource(show_spinner="Chargement des modèles…")
def load_artifacts():
    reg_files = ["model.joblib", "model_linreg.joblib", "dept_encoding.joblib",
                 "fallback.joblib", "features.joblib", "dept_coords.joblib", "reg_metrics.joblib"]
    clf_files = ["model_logreg.joblib", "model_rf_clf.joblib",
                 "features_clf.joblib", "clf_metrics.joblib"]
    missing_reg = [f for f in reg_files if not Path(f).exists()]
    missing_clf = [f for f in clf_files if not Path(f).exists()]
    result = {}
    if not missing_reg:
        result["hgb"]          = joblib.load("model.joblib")
        result["lin_reg"]      = joblib.load("model_linreg.joblib")
        result["encoding"]     = joblib.load("dept_encoding.joblib")
        result["fallback"]     = joblib.load("fallback.joblib")
        result["features_reg"] = joblib.load("features.joblib")
        result["coords"]       = joblib.load("dept_coords.joblib")
        result["reg_metrics"]  = joblib.load("reg_metrics.joblib")
    if not missing_clf:
        result["log_reg"]      = joblib.load("model_logreg.joblib")
        result["rf_clf"]       = joblib.load("model_rf_clf.joblib")
        result["features_clf"] = joblib.load("features_clf.joblib")
        result["clf_metrics"]  = joblib.load("clf_metrics.joblib")
    return result if result else None


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Accueil & Contexte
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil & Contexte":
    st.title("Prédiction du prix des appartements — DVF")
    st.markdown("### Projet Machine Learning — ESAIP 2024-2025")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ## Problématique
        > *Peut-on prédire le prix d'un appartement en France à partir de ses
        > caractéristiques et de sa localisation ?*

        Ce type de modèle est utilisé par :
        - Les **agents immobiliers** pour estimer un bien rapidement
        - Les plateformes comme **MeilleursAgents / SeLoger**
        - Les **banques** pour évaluer un bien en garantie de prêt

        ## Deux objectifs ML
        | Tâche | Modèle | Ce qu'on prédit |
        |---|---|---|
        | **Régression** | Régression linéaire + HistGradientBoosting | Prix de vente en € |
        | **Classification** | Régression logistique + Random Forest | Type de bien (Appartement / Maison) |
        """)

    with col2:
        st.markdown("""
        ## Source des données — DVF

        - **DVF** = Demandes de Valeurs Foncières
        - Publié par la **DGFiP** (Direction Générale des Finances Publiques)
        - Toutes les transactions immobilières notariées en France
        - Open data sur **data.gouv.fr**

        ## Volumétrie
        """)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Fichiers", "2 (2024 + 2025)")
        col_b.metric("Lignes brutes", "~7,2 M")
        col_c.metric("Après filtrage", "~726 k")

        st.markdown("""
        ## Colonnes utiles (sur 39)
        `valeur_fonciere` · `surface_reelle_bati` · `nombre_pieces_principales`
        · `code_departement` · `type_local` · `longitude` · `latitude`
        · `date_mutation` · `lot1_surface_carrez` · `nombre_lots`
        """)

    st.markdown("---")
    st.info("💡 **Comment lire cette app** : navigue avec le menu à gauche. "
            "Chaque page contient les explications à l'oral + les résultats.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Données & Nettoyage
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Données & Nettoyage":
    st.title("Données & Nettoyage")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ## Pipeline de nettoyage

        **Étape 1 — Filtrage**
        - Garder uniquement `nature_mutation == "Vente"` et `type_local == "Appartement"`
        - Résultat : 2,7 M → ~950 k lignes

        **Étape 2 — Valeurs aberrantes**
        - `valeur_fonciere` entre **10 000 €** et **10 000 000 €**
        - Raison : en dessous = vente familiale symbolique, au dessus = vente d'immeuble entier

        **Étape 3 — Surface incohérente**
        - `surface_reelle_bati` entre **0** et **500 m²**
        - Un appartement à 5000 m² = erreur de saisie

        **Étape 4 — Déduplication**
        - Une mutation peut apparaître plusieurs fois (plusieurs lots)
        - On garde **le lot à plus grande surface** par `id_mutation`

        **Étape 5 — GPS manquant**
        - Suppression des lignes sans `longitude` / `latitude`

        **Étape 6 — Imputation**
        - `nombre_pieces_principales` manquant → remplacé par la **médiane**
        """)

    with col2:
        st.markdown("""
        ## Feature Engineering

        | Feature créée | Formule / Source | Pourquoi ? |
        |---|---|---|
        | `annee`, `mois`, `trimestre` | extrait de `date_mutation` | Capturer l'évolution du marché |
        | `surface_carrez` | `lot1_surface_carrez` (ou 0) | Surface réglementée en copropriété |
        | `surface_par_piece` | surface / pièces | Proxy du confort (un T2 de 60m² > un T4 de 60m²) |
        | `prix_m2` | prix / surface | **EDA seulement** — contient la cible → pas dans le modèle |
        | `dept_prix_moyen` | moyenne des prix par département | **Target encoding** — transforme un code catégoriel en valeur numérique |

        ## ⚠️ Anti-fuite de données (Data Leakage)

        Le `dept_prix_moyen` est calculé **uniquement sur le jeu d'entraînement**,
        puis appliqué au test.

        **Pourquoi ?** Si on utilise tout le dataset, le modèle "voit" indirectement
        le prix des appartements du test pendant l'entraînement → scores gonflés.

        C'est une erreur classique et grave en ML.
        """)

    st.markdown("---")
    df = load_data()
    if df is not None:
        st.subheader("Aperçu des données nettoyées")
        col1, col2, col3 = st.columns(3)
        col1.metric("Lignes finales", f"{len(df):,}")
        col2.metric("Features modèle", "11")
        col3.metric("Valeurs manquantes", "0")
        st.dataframe(
            df[["date_mutation", "code_departement", "surface_reelle_bati",
                "nombre_pieces_principales", "valeur_fonciere", "prix_m2"]].head(8),
            use_container_width=True,
        )

        st.subheader("Distribution des prix (avant/après log)")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df["valeur_fonciere"] / 1000, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
        axes[0].set_title("Prix brut (en milliers €)")
        axes[0].set_xlabel("Prix (k€)")
        axes[1].hist(np.log1p(df["valeur_fonciere"]), bins=80, color="orange", edgecolor="none", alpha=0.8)
        axes[1].set_title("log(1 + prix) — distribution symétrique")
        axes[1].set_xlabel("log(prix)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("La distribution brute est très asymétrique (beaucoup de biens à 100-300k, "
                   "quelques-uns à plusieurs millions). Le log la rend quasi-normale → "
                   "meilleurs résultats pour les modèles linéaires.")
    else:
        st.warning("Lance `python download_data.py` pour télécharger les données.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Régression
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Régression — Prix":
    st.title("Régression — Prédiction du prix")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ## Objectif
        Prédire `valeur_fonciere` (prix de vente en €) à partir des caractéristiques du bien.

        ## Modèles testés

        **Régression Linéaire** (baseline)
        - Formule : `prix = a₁×surface + a₂×pièces + ... + b`
        - Simple et interprétable, mais suppose des relations linéaires
        - Utilisé avec `TransformedTargetRegressor` : entraîné sur `log(prix)`,
          prédit transformé en euros avec `exp`

        **HistGradientBoosting** (meilleur modèle)
        - Algorithme de boosting par gradient sur histogrammes (inspiré de LightGBM)
        - Construit des arbres successivement, chacun corrigeant les erreurs du précédent
        - Capture les **relations non-linéaires** (ex : grande surface ET Paris = prix
          beaucoup plus élevé qu'attendu par simple addition)
        - Robuste aux valeurs aberrantes, rapide à entraîner
        """)

    with col2:
        st.markdown("""
        ## Métriques d'évaluation

        | Métrique | Ce qu'elle mesure | Exemple |
        |---|---|---|
        | **MAE** | Erreur moyenne en € | "On se trompe de 70 000 € en moyenne" |
        | **RMSE** | Erreur quadratique — pénalise les grosses erreurs | Plus sensible aux outliers |
        | **MAPE** | Erreur en % du prix réel | "On se trompe de 45% en moyenne" |
        | **R²** | Part de la variance expliquée (0 à 1) | R²=0.57 → on explique 57% de la variance |

        ## Pourquoi le log de la cible ?
        La distribution des prix est très asymétrique.
        La régression minimise l'erreur au carré : elle est "tirée" par les prix élevés
        et prédit mal les biens standards.
        En travaillant sur `log(prix)`, tous les prix ont le même poids.
        """)

    st.markdown("---")
    arts = load_artifacts()
    if arts and "reg_metrics" in arts:
        m = arts["reg_metrics"]
        st.subheader("Résultats sur le jeu de test (20% des données)")
        df_res = pd.DataFrame([
            {"Modèle": "Régression Linéaire",  "R²": m["lin_reg"]["R2"],
             "MAE (€)": m["lin_reg"]["MAE"],   "MAPE (%)": m["lin_reg"]["MAPE"]},
            {"Modèle": "HistGradientBoosting", "R²": m["hgb"]["R2"],
             "MAE (€)": m["hgb"]["MAE"],       "MAPE (%)": m["hgb"]["MAPE"]},
        ])
        df_display = df_res.copy()
        df_display["R²"]       = df_display["R²"].map(lambda x: f"{x:.3f}")
        df_display["MAE (€)"]  = df_display["MAE (€)"].map(lambda x: f"{x:,.0f}")
        df_display["MAPE (%)"] = df_display["MAPE (%)"].map(lambda x: f"{x:.1f}%")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Meilleur R²",   f"{m['hgb']['R2']:.3f}",  help="HistGradientBoosting")
        col2.metric("Meilleur MAE",  f"{m['hgb']['MAE']:,.0f} €")
        col3.metric("Meilleur MAPE", f"{m['hgb']['MAPE']:.1f}%")
    else:
        st.warning("Lance `python save_model.py` pour générer les métriques.")

    st.markdown("---")
    df = load_data()
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 15 départements — prix médian au m²")
            top = df.groupby("code_departement")["prix_m2"].median().sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(7, 4))
            top.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_ylabel("€/m²")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            st.subheader("Prix médian par nombre de pièces")
            fig, ax = plt.subplots(figsize=(7, 4))
            df.groupby("nombre_pieces_principales")["valeur_fonciere"].median().loc[1:7].plot(
                kind="bar", ax=ax, color="coral"
            )
            ax.set_ylabel("Prix médian (€)")
            ax.set_xlabel("Pièces")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("""
    ## Limites du modèle (à mentionner à l'oral)
    - Pas d'info sur l'**étage**, l'**année de construction**, l'**ascenseur**, le **balcon** → très prédictifs
    - Le target encoding par **département** est grossier (le 7ème et le 20ème arr. de Paris ont des prix très différents)
    - Le marché évolue : un modèle 2024-2025 peut mal prédire en 2027

    ## Pistes d'amélioration
    - Target encoding par **commune** avec lissage bayésien
    - Données INSEE (revenus par commune), proximité transports
    - Split temporel : train 2024, test 2025
    """)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Classification
# ════════════════════════════════════════════════════════════════════════════
elif page == "🏷️ Classification — Type de bien":
    st.title("Classification — Identifier le type de bien")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ## Objectif
        Prédire si un bien est un **Appartement** (1) ou une **Maison** (0)
        à partir de ses caractéristiques physiques et géographiques.

        ## Modèles

        **Régression Logistique**
        - Modèle linéaire pour la classification
        - Calcule la probabilité d'appartenir à une classe : `P(Appartement) = σ(w·x + b)`
        - `σ` = fonction sigmoïde → sortie entre 0 et 1
        - Simple, rapide, interprétable
        - Nécessite un **StandardScaler** (les features doivent avoir la même échelle)

        **Random Forest**
        - Ensemble de nombreux arbres de décision entraînés sur des sous-échantillons
        - Vote majoritaire des arbres pour la prédiction finale
        - **Robuste** et performant sans normalisation
        - Donne l'importance des features

        ## Features utilisées
        `surface_reelle_bati` · `nombre_pieces_principales` · `nombre_lots`
        · `surface_carrez` · `longitude` · `latitude`
        """)

    with col2:
        st.markdown("""
        ## Métriques de classification

        | Métrique | Définition |
        |---|---|
        | **Accuracy** | % de bonnes prédictions au total |
        | **Précision** | Parmi les "Appartements" prédits, combien sont vrais ? |
        | **Rappel** | Parmi les vrais Appartements, combien détectés ? |
        | **F1-score** | Moyenne harmonique précision/rappel |

        ## Pourquoi Appartement vs Maison ?
        Ces deux types sont très différents en surface, nombre de pièces et localisation
        (appartements = grandes villes, maisons = zones péri-urbaines/rurales).
        Le modèle peut apprendre ces patterns géographiques.

        ## Intuition
        Un bien avec :
        - Surface < 80 m², 3 pièces, Paris → probablement **Appartement**
        - Surface > 100 m², 5 pièces, campagne → probablement **Maison**
        """)

    st.markdown("---")
    arts = load_artifacts()
    if arts and "clf_metrics" in arts:
        m = arts["clf_metrics"]
        st.subheader("Résultats sur le jeu de test (20% des données)")
        df_res = pd.DataFrame([
            {"Modèle": "Régression Logistique", "Accuracy": m["log_reg"]["accuracy"]},
            {"Modèle": "Random Forest",          "Accuracy": m["rf"]["accuracy"]},
        ])
        df_res["Accuracy"] = df_res["Accuracy"].map(lambda x: f"{x:.3f} ({x*100:.1f}%)")
        st.dataframe(df_res, use_container_width=True, hide_index=True)
    else:
        st.warning("Lance `python save_model.py` pour générer les métriques.")

    st.markdown("""
    ## Limites & pistes d'amélioration
    - La frontière Appartement/Maison est parfois floue dans les données
    - On pourrait ajouter **Dépendance** et **Local commercial** (multi-classe)
    - Features supplémentaires : `surface_terrain` (les maisons ont souvent un terrain)
    """)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Démo Régression
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Démo Régression":
    st.title("Démo — Estimer le prix d'un appartement")

    arts = load_artifacts()
    if not arts or "hgb" not in arts:
        st.error("Lance `python save_model.py` puis recharge la page.")
        st.stop()

    encoding     = arts["encoding"]
    coords       = arts["coords"]
    fallback_val = arts["fallback"]
    features_reg = arts["features_reg"]

    dept_options = sorted(encoding.index.tolist())

    col1, col2 = st.columns(2)
    with col1:
        surface = st.slider("Surface réelle bâtie (m²)", 10, 300, 65)
        pieces  = st.slider("Nombre de pièces principales", 1, 8, 3)
        dept_code = st.selectbox(
            "Département",
            dept_options,
            index=dept_options.index("75") if "75" in dept_options else 0,
        )
        nombre_lots = st.slider("Nombre de lots", 1, 5, 1)
    with col2:
        surface_carrez = st.slider("Surface Carrez (m², 0 si inconnue)", 0, 300, int(surface * 0.9))
        annee = st.selectbox("Année de vente", [2025, 2024])
        mois  = st.slider("Mois", 1, 12, 6)

    trimestre        = (mois - 1) // 3 + 1
    surface_par_piece = surface / max(pieces, 1)
    lat = float(coords.loc[dept_code, "latitude"])  if dept_code in coords.index else 46.6
    lon = float(coords.loc[dept_code, "longitude"]) if dept_code in coords.index else 2.3
    dept_prix_moyen  = float(encoding.get(dept_code, fallback_val))

    st.markdown("---")
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        predict_btn = st.button("Estimer le prix", type="primary", use_container_width=True)

    if predict_btn:
        X_in = pd.DataFrame([{
            "surface_reelle_bati":       surface,
            "nombre_pieces_principales": pieces,
            "nombre_lots":               nombre_lots,
            "surface_carrez":            surface_carrez,
            "surface_par_piece":         surface_par_piece,
            "longitude":                 lon,
            "latitude":                  lat,
            "annee":                     annee,
            "mois":                      mois,
            "trimestre":                 trimestre,
            "dept_prix_moyen":           dept_prix_moyen,
        }])[features_reg]

        pred_hgb = max(float(arts["hgb"].predict(X_in)[0]), 0)
        pred_lr  = max(float(arts["lin_reg"].predict(X_in)[0]), 0)

        st.success("### Résultats de prédiction")
        c1, c2, c3 = st.columns(3)
        c1.metric("HistGradientBoosting", f"{pred_hgb:,.0f} €",
                  help="Meilleur modèle — non-linéaire")
        c2.metric("Régression Linéaire",  f"{pred_lr:,.0f} €",
                  help="Baseline — linéaire")
        c3.metric("Prix au m² (HGB)",     f"{pred_hgb/surface:,.0f} €/m²")

        st.info(f"Prix moyen de référence pour le département **{dept_code}** : "
                f"{dept_prix_moyen:,.0f} €")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Démo Classification
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Démo Classification":
    st.title("Démo — Identifier le type de bien")

    arts = load_artifacts()
    if not arts or "log_reg" not in arts:
        st.error("Lance `python save_model.py` puis recharge la page.")
        st.stop()

    features_clf = arts["features_clf"]

    col1, col2 = st.columns(2)
    with col1:
        surface     = st.slider("Surface réelle bâtie (m²)", 10, 500, 80)
        pieces      = st.slider("Nombre de pièces principales", 1, 10, 4)
        nombre_lots = st.slider("Nombre de lots", 0, 5, 1)
    with col2:
        surface_carrez = st.slider("Surface Carrez (m², 0 si inconnue)", 0, 400, 70)
        latitude  = st.slider("Latitude",  42.0, 51.0, 46.5, step=0.1)
        longitude = st.slider("Longitude", -5.0, 9.0,   2.5, step=0.1)

    st.markdown("---")
    if st.button("Identifier le bien", type="primary"):
        X_in = pd.DataFrame([{
            "surface_reelle_bati":       surface,
            "nombre_pieces_principales": pieces,
            "nombre_lots":               nombre_lots,
            "surface_carrez":            surface_carrez,
            "longitude":                 longitude,
            "latitude":                  latitude,
        }])[features_clf]

        pred_lr = arts["log_reg"].predict(X_in)[0]
        prob_lr = arts["log_reg"].predict_proba(X_in)[0]
        pred_rf = arts["rf_clf"].predict(X_in)[0]
        prob_rf = arts["rf_clf"].predict_proba(X_in)[0]

        label_lr = "🏢 Appartement" if pred_lr == 1 else "🏡 Maison"
        label_rf = "🏢 Appartement" if pred_rf == 1 else "🏡 Maison"

        st.success("### Résultats de classification")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Régression Logistique", label_lr)
            st.progress(float(prob_lr[pred_lr]), text=f"Confiance : {prob_lr[pred_lr]*100:.0f}%")
        with c2:
            st.metric("Random Forest", label_rf)
            st.progress(float(prob_rf[pred_rf]), text=f"Confiance : {prob_rf[pred_rf]*100:.0f}%")

        if pred_lr == pred_rf:
            st.info("Les deux modèles sont d'accord.")
        else:
            st.warning("Les modèles ne sont pas d'accord — cas ambigu.")
