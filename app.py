"""Application Streamlit — Présentation du projet ML DVF.

Lancement : streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Config page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Prédiction prix appartements — DVF",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Navigation latérale
# ---------------------------------------------------------------------------
st.sidebar.title("🏠 Projet ML — DVF")
page = st.sidebar.radio(
    "Navigation",
    [
        "A. Contexte & Données",
        "B. Feature Engineering & EDA",
        "C. Modélisation & Résultats",
        "D. Démo interactive",
    ],
)

# ---------------------------------------------------------------------------
# Fonctions de chargement (mises en cache)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement des données DVF…")
def load_data():
    data_dir = Path("datasets")
    csv_files = sorted(data_dir.glob("full_202[45].csv"))
    if not csv_files:
        return None

    COLS = [
        "id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere",
        "code_postal", "code_departement", "type_local",
        "surface_reelle_bati", "nombre_pieces_principales",
        "nombre_lots", "lot1_surface_carrez",
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
    df["annee"] = df["date_mutation"].dt.year
    df["mois"] = df["date_mutation"].dt.month
    df["trimestre"] = df["date_mutation"].dt.quarter
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    df["surface_carrez"] = df["lot1_surface_carrez"].fillna(0)
    df["surface_par_piece"] = df["surface_reelle_bati"] / df["nombre_pieces_principales"].clip(lower=1)
    dept_mean = df.groupby("code_departement")["valeur_fonciere"].mean()
    df["dept_prix_moyen"] = df["code_departement"].map(dept_mean)
    return df


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model_artifacts():
    needed = ["model.joblib", "dept_encoding.joblib", "fallback.joblib",
              "features.joblib", "dept_coords.joblib"]
    if not all(Path(f).exists() for f in needed):
        return None
    return {
        "model":    joblib.load("model.joblib"),
        "encoding": joblib.load("dept_encoding.joblib"),
        "fallback": joblib.load("fallback.joblib"),
        "features": joblib.load("features.joblib"),
        "coords":   joblib.load("dept_coords.joblib"),
    }


# ---------------------------------------------------------------------------
# Page A — Contexte & Données
# ---------------------------------------------------------------------------
if page == "A. Contexte & Données":
    st.title("A. Contexte & Données")

    st.markdown("""
    ### Problématique
    > *Peut-on prédire le prix d'un appartement en France à partir de ses caractéristiques et de sa localisation ?*

    Ce type de modèle intéresse les **agents immobiliers**, les plateformes comme MeilleursAgents,
    et les **banques** pour évaluer un bien en garantie.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Années couvertes", "2024 – 2025")
    col2.metric("Source", "DVF (DGFiP)")
    col3.metric("Format", "CSV open data")

    st.markdown("---")
    st.subheader("Pipeline de nettoyage")
    st.markdown("""
    1. Filtrer `Vente` + `Appartement`
    2. Valeur foncière entre **10 000 €** et **10 000 000 €**
    3. Surface entre **0** et **500 m²**
    4. **Déduplication** par `id_mutation` (on garde le lot à plus grande surface)
    5. Suppression des lignes sans coordonnées GPS
    6. Imputation des pièces manquantes → médiane
    """)

    df = load_data()
    if df is not None:
        col1, col2 = st.columns(2)
        col1.metric("Lignes après nettoyage", f"{len(df):,}")
        col2.metric("Colonnes features", "11")
        st.subheader("Aperçu des données")
        st.dataframe(df[["date_mutation", "code_departement", "surface_reelle_bati",
                          "nombre_pieces_principales", "valeur_fonciere"]].head(10))
    else:
        st.warning("Données introuvables dans `datasets/`. Lance `python download_data.py` d'abord.")


# ---------------------------------------------------------------------------
# Page B — Feature Engineering & EDA
# ---------------------------------------------------------------------------
elif page == "B. Feature Engineering & EDA":
    st.title("B. Feature Engineering & EDA")

    st.markdown("""
    | Feature | Description |
    |---|---|
    | `annee`, `mois`, `trimestre` | Évolution du marché dans le temps |
    | `surface_carrez` | Surface officielle du lot (réglementée) |
    | `surface_par_piece` | Proxy du confort du logement |
    | `dept_prix_moyen` | **Target encoding** du département → valeur numérique |
    | `prix_m2` | Indicateur EDA uniquement — **pas utilisé par le modèle** |
    """)

    st.info("**Point clé — Anti-fuite de données :** le `dept_prix_moyen` est calculé "
            "uniquement sur le train, puis appliqué au test. Sans cette précaution, "
            "le modèle verrait indirectement sa cible pendant l'entraînement.")

    df = load_data()
    if df is None:
        st.warning("Données non disponibles.")
        st.stop()

    st.markdown("---")
    st.subheader("Distribution des prix")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["valeur_fonciere"], bins=80, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_title("Prix brut (EUR)")
    axes[1].hist(np.log1p(df["valeur_fonciere"]), bins=80, color="orange", edgecolor="black", alpha=0.7)
    axes[1].set_title("log(1 + prix)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Top 15 départements — prix médian au m²")
    top_dept = df.groupby("code_departement")["prix_m2"].median().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 4))
    top_dept.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("Prix médian au m² (EUR)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prix par nombre de pièces")
        fig, ax = plt.subplots(figsize=(6, 4))
        df.groupby("nombre_pieces_principales")["valeur_fonciere"].median().loc[1:7].plot(
            kind="bar", ax=ax, color="coral")
        ax.set_xlabel("Pièces")
        ax.set_ylabel("Prix médian (EUR)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Évolution trimestrielle")
        prix_trim = df.groupby(["annee", "trimestre"])["valeur_fonciere"].median().reset_index()
        prix_trim["periode"] = prix_trim["annee"].astype(str) + "-T" + prix_trim["trimestre"].astype(str)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(prix_trim["periode"], prix_trim["valeur_fonciere"], marker="o")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Matrice de corrélation")
    num_cols = ["valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales",
                "surface_carrez", "longitude", "latitude", "dept_prix_moyen"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ---------------------------------------------------------------------------
# Page C — Modélisation & Résultats
# ---------------------------------------------------------------------------
elif page == "C. Modélisation & Résultats":
    st.title("C. Modélisation & Résultats")

    st.markdown("""
    ### Modèles entraînés
    | Famille | Modèle | Notes |
    |---|---|---|
    | Linéaire | Régression Linéaire | Baseline — avec log(y) |
    | Linéaire | Ridge | Régularisation L2 |
    | Linéaire | Lasso | Régularisation L1 — sélection de variables |
    | Arbres | Random Forest | Robuste, non-linéaire |
    | Arbres | HistGradientBoosting | Équivalent LightGBM, très rapide |
    """)

    st.markdown("---")
    st.subheader("Résultats comparatifs")
    results = pd.DataFrame([
        {"Modèle": "Régression Linéaire", "R²": "—", "MAE (€)": "—", "MAPE (%)": "—"},
        {"Modèle": "Ridge",               "R²": "—", "MAE (€)": "—", "MAPE (%)": "—"},
        {"Modèle": "Lasso",               "R²": "—", "MAE (€)": "—", "MAPE (%)": "—"},
        {"Modèle": "Random Forest",        "R²": "—", "MAE (€)": "—", "MAPE (%)": "—"},
        {"Modèle": "HistGradientBoosting", "R²": "—", "MAE (€)": "—", "MAPE (%)": "—"},
    ])
    st.dataframe(results, use_container_width=True)
    st.caption("Résultats à mettre à jour après exécution complète du notebook.")

    st.markdown("---")
    st.subheader("Points clés ML")
    col1, col2 = st.columns(2)
    col1.markdown("""
    **Transformation log de la cible**
    - Distribution des prix très asymétrique
    - `log(1+prix)` → distribution quasi-normale
    - Re-transformation en euros avec `exp`
    - Géré par `TransformedTargetRegressor`
    """)
    col2.markdown("""
    **Validation croisée 5-fold**
    - 5 splits différents du train
    - R² moyen ± écart-type
    - Vérifie la stabilité du modèle
    """)

    st.subheader("Importance des features")
    st.info("À compléter après exécution du notebook — graphique permutation importance.")


# ---------------------------------------------------------------------------
# Page D — Démo interactive
# ---------------------------------------------------------------------------
elif page == "D. Démo interactive":
    st.title("D. Démo — Estimer un prix d'appartement")

    arts = load_model_artifacts()
    if arts is None:
        st.error(
            "Modèle non trouvé. Exécute la dernière cellule du notebook "
            "(celle avec `joblib.dump`) puis recharge cette page."
        )
        st.stop()

    encoding: pd.Series = arts["encoding"]
    coords: pd.DataFrame = arts["coords"]
    model = arts["model"]
    fallback_price: float = arts["fallback"]

    dept_options = sorted(encoding.index.tolist())

    st.markdown("Renseigne les caractéristiques d'un appartement pour obtenir une estimation de prix.")

    col1, col2 = st.columns(2)
    with col1:
        surface = st.slider("Surface réelle bâtie (m²)", 10, 300, 65)
        pieces = st.slider("Nombre de pièces principales", 1, 8, 3)
        dept_code = st.selectbox(
            "Département",
            dept_options,
            index=dept_options.index("75") if "75" in dept_options else 0,
        )
        nombre_lots = st.slider("Nombre de lots", 1, 5, 1)
    with col2:
        surface_carrez = st.slider("Surface Carrez (m², 0 si non applicable)", 0, 300, int(surface * 0.9))
        annee = st.selectbox("Année de vente", [2025, 2024])
        mois = st.slider("Mois", 1, 12, 6)

    trimestre = (mois - 1) // 3 + 1
    surface_par_piece = surface / max(pieces, 1)
    lat = float(coords.loc[dept_code, "latitude"]) if dept_code in coords.index else 46.6
    lon = float(coords.loc[dept_code, "longitude"]) if dept_code in coords.index else 2.3
    dept_prix_moyen = float(encoding.get(dept_code, fallback_price))

    st.markdown("---")
    if st.button("Estimer le prix", type="primary"):
        X_input = pd.DataFrame([{
            "surface_reelle_bati":        surface,
            "nombre_pieces_principales":  pieces,
            "nombre_lots":                nombre_lots,
            "surface_carrez":             surface_carrez,
            "surface_par_piece":          surface_par_piece,
            "longitude":                  lon,
            "latitude":                   lat,
            "annee":                      annee,
            "mois":                       mois,
            "trimestre":                  trimestre,
            "dept_prix_moyen":            dept_prix_moyen,
        }])[arts["features"]]

        pred = float(model.predict(X_input)[0])
        pred = max(pred, 0)

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Prix estimé", f"{pred:,.0f} €")
        col_r2.metric("Prix au m²", f"{pred / surface:,.0f} €/m²")
        col_r3.metric("Département", dept_code)
        st.success(
            f"Estimation basée sur un prix moyen de {dept_prix_moyen:,.0f} € "
            f"dans le département {dept_code}."
        )
