"""Application Streamlit — Support de presentation orale DVF.

Lancement : streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Prediction DVF — ML",
    page_icon="🏠",
    layout="wide",
)

# ── Navigation ──────────────────────────────────────────────────────────────
PAGES = [
    "🏠 Accueil & Contexte",
    "📊 Donnees & Nettoyage",
    "🔬 Analyses Exploratoires",
    "📈 Regression — Prix",
    "🏷️ Classification — Type de bien",
    "🔮 Demo Regression",
    "🔍 Demo Classification",
]
page = st.sidebar.radio("Navigation", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Support oral — DVF 2024-2025")


# ── Loaders ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des donnees…")
def load_data():
    """Charge et nettoie le dataset DVF (Maison + Appartement) — meme logique
    que projet.ipynb, pour que l'EDA de l'app reflete ce qu'on a fait dans le
    notebook."""
    data_dir = Path("datasets")
    csv_files = sorted(data_dir.glob("full_202[45].csv"))
    if not csv_files:
        return None
    cols = [
        "id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere",
        "code_departement", "type_local", "surface_reelle_bati",
        "nombre_pieces_principales", "surface_terrain", "nombre_lots",
        "lot1_surface_carrez", "longitude", "latitude",
    ]
    frames = [pd.read_csv(f, usecols=cols, low_memory=False) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)

    df = df[
        (df["nature_mutation"] == "Vente")
        & (df["type_local"].isin(["Maison", "Appartement"]))
    ].copy()
    df = df.dropna(subset=["valeur_fonciere"])
    df = df[(df["valeur_fonciere"] >= 10_000) & (df["valeur_fonciere"] <= 10_000_000)]
    df = df[(df["surface_reelle_bati"] > 0) & (df["surface_reelle_bati"] <= 500)]
    df = df.sort_values("surface_reelle_bati", ascending=False).drop_duplicates(
        "id_mutation", keep="first"
    )
    df = df.dropna(subset=["longitude", "latitude"])
    df["nombre_pieces_principales"] = df["nombre_pieces_principales"].fillna(
        df["nombre_pieces_principales"].median()
    )
    df["surface_terrain"] = df["surface_terrain"].fillna(0)

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


@st.cache_resource(show_spinner="Chargement des modeles…")
def load_artifacts():
    reg_files = ["model.joblib", "model_linreg.joblib", "dept_encoding.joblib",
                 "fallback.joblib", "features.joblib", "dept_coords.joblib",
                 "reg_metrics.joblib"]
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
        if Path("reg_preview.joblib").exists():
            result["reg_preview"] = joblib.load("reg_preview.joblib")
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
    st.title("Prediction du prix immobilier — DVF")
    st.markdown("### Projet Machine Learning — DVF 2024-2025")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ## Problematique
        > *Peut-on, a partir des transactions immobilieres publiques, predire
        > le prix d'un bien et identifier automatiquement son type ?*

        Ce type de modele est utilise par :
        - Les **agents immobiliers** pour estimer un bien rapidement
        - Les plateformes comme **MeilleursAgents / SeLoger**
        - Les **banques** pour evaluer un bien en garantie de pret

        ## Deux objectifs ML
        | Tache | Modeles | Cible |
        |---|---|---|
        | **Regression** | Regression Lineaire (log) + HistGradientBoosting | Prix de vente en EUR |
        | **Classification** | Regression Logistique + Random Forest | Maison vs Appartement |
        """)

    with col2:
        st.markdown("""
        ## Source des donnees — DVF

        - **DVF** = Demandes de Valeurs Foncieres
        - Publie par la **DGFiP** (Direction Generale des Finances Publiques)
        - Toutes les transactions immobilieres notariees en France
        - Open data sur **data.gouv.fr** (geo-DVF)

        ## Volumetrie
        """)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Fichiers", "2 (2024 + 2025)")
        col_b.metric("Lignes brutes", "~7,2 M")
        col_c.metric("Apres nettoyage", "~1,1 M")

        st.markdown("""
        ## Colonnes utiles (sur 39)
        `valeur_fonciere` · `surface_reelle_bati` · `nombre_pieces_principales`
        · `surface_terrain` · `code_departement` · `type_local` · `longitude`
        · `latitude` · `date_mutation` · `lot1_surface_carrez` · `nombre_lots`
        """)

    st.markdown("---")
    st.info("💡 **Comment lire cette app** : navigue avec le menu a gauche. "
            "Chaque page contient les explications a l'oral + les resultats.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Donnees & Nettoyage
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Donnees & Nettoyage":
    st.title("Donnees & Nettoyage")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ## Pipeline de nettoyage

        **Etape 1 — Filtrage**
        - Garder `nature_mutation == "Vente"` et `type_local in {Maison, Appartement}`

        **Etape 2 — Valeurs aberrantes**
        - `valeur_fonciere` entre **10 000 EUR** et **10 000 000 EUR**
        - Raison : en dessous = vente familiale symbolique, au dessus = vente d'immeuble

        **Etape 3 — Surface incoherente**
        - `surface_reelle_bati` entre **0** et **500 m²**

        **Etape 4 — Deduplication**
        - Une mutation peut apparaitre plusieurs fois (multi-lots)
        - On garde **le lot a plus grande surface** par `id_mutation`

        **Etape 5 — GPS manquant**
        - Suppression des lignes sans `longitude` / `latitude`

        **Etape 6 — Imputation**
        - `nombre_pieces_principales` manquant → **mediane**
        - `surface_terrain` manquante → **0** (typique des appartements)
        """)

    with col2:
        st.markdown("""
        ## Feature Engineering

        | Feature creee | Formule | Pourquoi ? |
        |---|---|---|
        | `annee`, `mois`, `trimestre` | extrait de `date_mutation` | Capturer l'evolution du marche |
        | `surface_carrez` | `lot1_surface_carrez` (ou 0) | Surface reglementaire en copro |
        | `surface_par_piece` | surface / pieces | Proxy du confort |
        | `prix_m2` | prix / surface | **EDA seulement** — contient la cible |
        | `is_maison` | type_local == Maison | Feature binaire pour la regression |
        | `dept_prix_moyen` | moyenne des prix par departement | **Target encoding** |

        ## ⚠️ Anti-fuite de donnees (Data Leakage)

        Le `dept_prix_moyen` est calcule **uniquement sur le jeu d'entrainement**,
        puis applique au test.

        **Pourquoi ?** Si on utilise tout le dataset, le modele "voit" indirectement
        le prix des biens du test pendant l'entrainement → scores gonfles.

        C'est une erreur classique et grave en ML.
        """)

    st.markdown("---")
    df = load_data()
    if df is not None:
        st.subheader("Apercu des donnees nettoyees")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes finales", f"{len(df):,}")
        c2.metric("Appartements", f"{(df['type_local']=='Appartement').sum():,}")
        c3.metric("Maisons", f"{(df['type_local']=='Maison').sum():,}")
        c4.metric("Departements", f"{df['code_departement'].nunique()}")

        st.dataframe(
            df[["date_mutation", "type_local", "code_departement", "surface_reelle_bati",
                "nombre_pieces_principales", "surface_terrain",
                "valeur_fonciere", "prix_m2"]].head(10),
            use_container_width=True,
        )
    else:
        st.warning("Lance `python download_data.py` pour telecharger les donnees.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Analyses Exploratoires (NOUVELLE)
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Analyses Exploratoires":
    st.title("Analyses Exploratoires (EDA)")
    st.caption("Visualisations cles a presenter pendant la soutenance.")

    df = load_data()
    if df is None:
        st.warning("Donnees indisponibles.")
        st.stop()

    # ─── Distribution de la cible ──────────────────────────────────────
    st.subheader("1. Distribution de la variable cible")
    st.markdown("La distribution brute est tres asymetrique. La transformation "
                "**log(1 + prix)** la rend quasi-normale → meilleurs resultats "
                "pour les modeles lineaires.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(df["valeur_fonciere"] / 1000, bins=80, color="steelblue",
                 edgecolor="none", alpha=0.85)
    axes[0].set_title("Prix brut (k EUR)")
    axes[0].set_xlabel("Prix (k EUR)")
    axes[0].set_ylabel("Frequence")
    axes[1].hist(np.log1p(df["valeur_fonciere"]), bins=80, color="orange",
                 edgecolor="none", alpha=0.85)
    axes[1].axvline(np.log1p(df["valeur_fonciere"]).mean(), color="red",
                    linestyle="--", linewidth=1, label="Moyenne")
    axes[1].set_title("log(1 + prix) — distribution symetrique")
    axes[1].set_xlabel("log(prix)")
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ─── Top departements ──────────────────────────────────────────────
    st.subheader("2. Prix median au m² — Top 15 departements")
    st.markdown("La localisation est de loin la variable la plus predictive du "
                "prix. Paris et la petite couronne ecrasent le classement.")
    top = (df.groupby("code_departement")["prix_m2"].median()
             .sort_values(ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(12, 4))
    top.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xlabel("Departement")
    ax.set_ylabel("Prix median (EUR/m²)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ─── Pieces ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("3. Prix median par nombre de pieces")
        fig, ax = plt.subplots(figsize=(7, 4))
        df.groupby("nombre_pieces_principales")["valeur_fonciere"].median().loc[1:7].plot(
            kind="bar", ax=ax, color="coral"
        )
        ax.set_ylabel("Prix median (EUR)")
        ax.set_xlabel("Nombre de pieces")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Le prix croit avec le nombre de pieces, mais pas lineairement "
                   "— les T6+ sont sur-representes en zones rurales (effet localisation).")

    with col2:
        st.subheader("4. Repartition Maison / Appartement")
        fig, ax = plt.subplots(figsize=(7, 4))
        df["type_local"].value_counts().plot(
            kind="bar", ax=ax, color=["steelblue", "coral"]
        )
        ax.set_ylabel("Nombre de ventes")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        rep = df["type_local"].value_counts(normalize=True) * 100
        st.caption(f"Repartition : Maison {rep.get('Maison', 0):.1f}% / "
                   f"Appartement {rep.get('Appartement', 0):.1f}%.")

    st.markdown("---")

    # ─── Surface terrain par type (le KO de la classification) ─────────
    st.subheader("5. Surface terrain par type — feature ultra-discriminante")
    st.markdown("`surface_terrain` est quasi-nulle pour les appartements et "
                "non-nulle pour les maisons. C'est *la* raison pour laquelle "
                "la classification atteint ~99% d'accuracy.")
    sample_box = df[df["surface_terrain"] <= 2000].sample(
        min(100_000, len(df)), random_state=42
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(
        data=sample_box, x="type_local", y="surface_terrain",
        ax=ax, hue="type_local", palette=["steelblue", "coral"], legend=False,
    )
    ax.set_ylabel("Surface terrain (m²)")
    ax.set_xlabel("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ─── Evolution temporelle ─────────────────────────────────────────
    st.subheader("6. Evolution trimestrielle des prix")
    prix_trim = (df.groupby(["annee", "trimestre"])["valeur_fonciere"]
                   .median().reset_index())
    prix_trim["periode"] = (prix_trim["annee"].astype(str) + "-T" +
                            prix_trim["trimestre"].astype(str))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(prix_trim["periode"], prix_trim["valeur_fonciere"], marker="o", linewidth=2)
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Prix median (EUR)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ─── Scatter surface ↔ prix ───────────────────────────────────────
    st.subheader("7. Surface bati vs prix")
    sample = df.sample(min(30_000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 5))
    for typ, color in [("Appartement", "steelblue"), ("Maison", "coral")]:
        sub = sample[sample["type_local"] == typ]
        ax.scatter(sub["surface_reelle_bati"], sub["valeur_fonciere"],
                   alpha=0.15, s=5, color=color, label=typ)
    ax.set_xlabel("Surface bati (m²)")
    ax.set_ylabel("Prix (EUR)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("La relation surface↔prix est globalement lineaire mais tres "
               "dispersee — la dispersion est portee par la localisation.")

    st.markdown("---")

    # ─── Correlation ──────────────────────────────────────────────────
    st.subheader("8. Matrice de correlation")
    num_cols = [
        "valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales",
        "surface_terrain", "nombre_lots", "surface_carrez",
        "longitude", "latitude", "dept_prix_moyen",
    ]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("`dept_prix_moyen` et `surface_reelle_bati` sont les plus correlees "
               "au prix. `surface_carrez` est correlee a la surface bati (logique).")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Regression
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Regression — Prix":
    st.title("Regression — Prediction du prix")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ## Objectif
        Predire `valeur_fonciere` (prix de vente en EUR) a partir des
        caracteristiques du bien.

        ## Modeles testes

        **Regression Lineaire (log-target)** — modele du notebook
        - Formule : `log(1 + prix) = a₁·surface + a₂·pieces + ... + b`
        - Simple et interpretable, suppose des relations lineaires
        - Encapsule dans un `TransformedTargetRegressor` qui applique `log` a
          l'entrainement et `exp` a la prediction.

        **HistGradientBoosting** — meilleur modele
        - Boosting par gradient sur histogrammes (inspire de LightGBM)
        - Construit des arbres successivement, chacun corrigeant les erreurs
          du precedent
        - Capture les **interactions non-lineaires** (ex : grande surface ET
          Paris = explosion du prix)
        - Robuste aux valeurs aberrantes, rapide a entrainer
        """)

    with col2:
        st.markdown("""
        ## Metriques d'evaluation

        | Metrique | Ce qu'elle mesure |
        |---|---|
        | **MAE** | Erreur moyenne en EUR |
        | **RMSE** | Erreur quadratique — penalise les grosses erreurs |
        | **MAPE** | Erreur en % du prix reel |
        | **R²** | Part de la variance expliquee (0 a 1) |

        ## Pourquoi le log de la cible ?
        La distribution des prix est tres asymetrique. La regression
        minimise l'erreur au carre : sans correction, elle est "tiree" par
        les prix eleves et predit mal les biens standards. En travaillant
        sur `log(prix)`, tous les ordres de grandeur ont le meme poids.

        ## Anti-fuite
        Le target encoding `dept_prix_moyen` est recalcule sur le **train
        uniquement**, puis applique au test. Sans ce soin, le R² mesure
        serait artificiellement meilleur que la realite operationnelle.
        """)

    st.markdown("---")
    arts = load_artifacts()
    if arts and "reg_metrics" in arts:
        m = arts["reg_metrics"]
        st.subheader("Resultats sur le jeu de test (20% des donnees)")
        df_res = pd.DataFrame([
            {"Modele": "Regression Lineaire (log)",
             "R²":      m["lin_reg"]["R2"],
             "MAE (EUR)": m["lin_reg"]["MAE"],
             "RMSE (EUR)": m["lin_reg"].get("RMSE", float("nan")),
             "MAPE (%)": m["lin_reg"]["MAPE"]},
            {"Modele": "HistGradientBoosting",
             "R²":      m["hgb"]["R2"],
             "MAE (EUR)": m["hgb"]["MAE"],
             "RMSE (EUR)": m["hgb"].get("RMSE", float("nan")),
             "MAPE (%)": m["hgb"]["MAPE"]},
        ])
        df_display = df_res.copy()
        df_display["R²"]         = df_display["R²"].map(lambda x: f"{x:.3f}")
        df_display["MAE (EUR)"]  = df_display["MAE (EUR)"].map(lambda x: f"{x:,.0f}")
        df_display["RMSE (EUR)"] = df_display["RMSE (EUR)"].map(lambda x: f"{x:,.0f}")
        df_display["MAPE (%)"]   = df_display["MAPE (%)"].map(lambda x: f"{x:.1f}%")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Meilleur R²",   f"{m['hgb']['R2']:.3f}",  help="HistGradientBoosting")
        c2.metric("Meilleur MAE",  f"{m['hgb']['MAE']:,.0f} EUR")
        c3.metric("Meilleur MAPE", f"{m['hgb']['MAPE']:.1f}%")

        # Scatter predictions vs realite
        if "reg_preview" in arts:
            st.markdown("---")
            st.subheader("Predictions vs realite (echantillon du test)")
            preview = arts["reg_preview"]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, col, title, color in [
                (axes[0], "y_pred_lr",  "Regression Lineaire",  "steelblue"),
                (axes[1], "y_pred_hgb", "HistGradientBoosting", "seagreen"),
            ]:
                ax.scatter(preview["y_true"], preview[col], alpha=0.2, s=6, color=color)
                lim = float(preview["y_true"].quantile(0.99))
                ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Prediction parfaite")
                ax.set_xlim(0, lim); ax.set_ylim(0, lim)
                ax.set_xlabel("Valeur reelle (EUR)")
                ax.set_ylabel("Valeur predite (EUR)")
                ax.set_title(title)
                ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("Plus les points collent a la diagonale rouge, mieux le modele "
                       "predit. HGB est visiblement plus aligne, surtout pour les prix "
                       "moyens et eleves.")
    else:
        st.warning("Lance `python save_model.py` pour generer les metriques.")

    st.markdown("""
    ---
    ## Limites du modele (a mentionner a l'oral)
    - Pas d'info sur l'**etage**, l'**annee de construction**, l'**ascenseur**,
      le **balcon** → toutes tres predictives
    - Le target encoding par **departement** est grossier (le 7eme et le 20eme
      arr. de Paris ont des prix tres differents)
    - Le marche evolue : un modele 2024-2025 peut mal predire en 2027

    ## Pistes d'amelioration
    - Target encoding par **commune** avec lissage bayesien
    - Donnees externes : revenus INSEE par commune, proximite transports
    - Split temporel : train 2024 / test 2025 (plus realiste)
    - Modeles plus puissants : XGBoost / LightGBM, voire reseaux de neurones
      tabulaires (TabNet, FT-Transformer)
    """)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Classification
# ════════════════════════════════════════════════════════════════════════════
elif page == "🏷️ Classification — Type de bien":
    st.title("Classification — Identifier le type de bien")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ## Objectif
        Predire si un bien est un **Appartement** (1) ou une **Maison** (0)
        a partir de ses caracteristiques physiques et geographiques.

        ## Modeles

        **Regression Logistique**
        - Modele lineaire pour la classification
        - Calcule la probabilite : `P(Appartement) = σ(w·x + b)`
        - `σ` = fonction sigmoide → sortie entre 0 et 1
        - Necessite un **StandardScaler** (echelle homogene)

        **Random Forest**
        - Ensemble d'arbres de decision sur sous-echantillons
        - Vote majoritaire pour la prediction finale
        - Robuste, capture les interactions non-lineaires
        - Donne l'importance des features

        ## Features utilisees
        `surface_reelle_bati` · `nombre_pieces_principales` · `nombre_lots` ·
        `surface_carrez` · `surface_par_piece` · **`surface_terrain`** ·
        `valeur_fonciere` · `longitude` · `latitude`
        """)

    with col2:
        st.markdown("""
        ## Metriques de classification

        | Metrique | Definition |
        |---|---|
        | **Accuracy** | % de bonnes predictions |
        | **Precision** | Parmi les "Appartements" predits, combien vrais ? |
        | **Rappel** | Parmi les vrais Appartements, combien detectes ? |
        | **F1-score** | Moyenne harmonique precision/rappel |

        ## Pourquoi cette tache est facile ?
        `surface_terrain` est quasi-nulle pour les appartements et tres
        elevee pour les maisons. Cette seule feature suffit deja a
        depasser 95% d'accuracy.

        ## Intuition
        - Surface < 80 m², 3 pieces, terrain ≈ 0 m², Paris → **Appartement**
        - Surface > 100 m², 5 pieces, 600 m² de terrain → **Maison**
        """)

    st.markdown("---")
    arts = load_artifacts()
    if arts and "clf_metrics" in arts:
        m = arts["clf_metrics"]
        st.subheader("Resultats sur le jeu de test (20% des donnees)")
        df_res = pd.DataFrame([
            {"Modele": "Regression Logistique", "Accuracy": m["log_reg"]["accuracy"]},
            {"Modele": "Random Forest",          "Accuracy": m["rf"]["accuracy"]},
        ])
        df_res["Accuracy"] = df_res["Accuracy"].map(lambda x: f"{x:.3f} ({x*100:.1f}%)")
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        # Matrices de confusion
        st.subheader("Matrices de confusion")
        labels = ["Maison", "Appartement"]
        col_a, col_b = st.columns(2)
        for col_ax, key, name, cmap in [
            (col_a, "log_reg", "Regression Logistique", "Blues"),
            (col_b, "rf",      "Random Forest",         "Greens"),
        ]:
            cm = np.array(m[key].get("confusion_matrix", [[0, 0], [0, 0]]))
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Prediction"); ax.set_ylabel("Realite")
            ax.set_title(f"{name} — {m[key]['accuracy']:.3f}")
            plt.tight_layout()
            with col_ax:
                st.pyplot(fig); plt.close()

        # Classification report
        st.subheader("Classification report (Random Forest)")
        report_rf = m["rf"].get("report", {})
        if report_rf:
            df_rep = pd.DataFrame(report_rf).T
            df_rep = df_rep.round(3)
            st.dataframe(df_rep, use_container_width=True)

        # Importances
        if "rf_feature_importances" in m:
            st.subheader("Importance des features — Random Forest")
            imp = pd.Series(m["rf_feature_importances"]).sort_values()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(imp.index, imp.values, color="seagreen")
            ax.set_xlabel("Importance (diminution moyenne d'impurete)")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.caption("Sans surprise : `surface_terrain` domine, suivie de "
                       "`surface_carrez` (presente surtout en copropriete) et "
                       "`nombre_lots`.")
    else:
        st.warning("Lance `python save_model.py` pour generer les metriques.")

    st.markdown("""
    ---
    ## Limites & pistes d'amelioration
    - La frontiere Appartement/Maison est parfois floue dans les donnees
      (ex : duplex avec terrain)
    - On pourrait ajouter **Dependance** et **Local commercial** (multi-classe)
    - Tester **XGBoost** / **GradientBoosting** + **GridSearchCV** pour gratter
      les derniers points
    """)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Demo Regression
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Demo Regression":
    st.title("Demo — Estimer le prix d'un bien")

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
        type_local = st.radio(
            "Type de bien",
            ["Appartement", "Maison"],
            horizontal=True,
            help="Determine la feature is_maison.",
        )
        surface = st.slider("Surface reelle batie (m²)", 10, 500, 65)
        pieces  = st.slider("Nombre de pieces principales", 1, 10, 3)
        dept_code = st.selectbox(
            "Departement",
            dept_options,
            index=dept_options.index("75") if "75" in dept_options else 0,
        )
        nombre_lots = st.slider("Nombre de lots", 1, 5, 1)

    with col2:
        surface_carrez = st.slider(
            "Surface Carrez (m², 0 si inconnue)",
            0, 400, int(surface * 0.9 if type_local == "Appartement" else 0),
        )
        surface_terrain = st.slider(
            "Surface terrain (m², 0 pour un appartement)",
            0, 5000, 0 if type_local == "Appartement" else 400, step=10,
        )
        annee = st.selectbox("Annee de vente", [2025, 2024])
        mois  = st.slider("Mois", 1, 12, 6)

    trimestre         = (mois - 1) // 3 + 1
    surface_par_piece = surface / max(pieces, 1)
    is_maison         = 1 if type_local == "Maison" else 0
    lat = float(coords.loc[dept_code, "latitude"])  if dept_code in coords.index else 46.6
    lon = float(coords.loc[dept_code, "longitude"]) if dept_code in coords.index else 2.3
    dept_prix_moyen   = float(encoding.get(dept_code, fallback_val))

    st.markdown("---")
    if st.button("Estimer le prix", type="primary", use_container_width=True):
        X_in = pd.DataFrame([{
            "surface_reelle_bati":       surface,
            "nombre_pieces_principales": pieces,
            "nombre_lots":               nombre_lots,
            "surface_carrez":            surface_carrez,
            "surface_par_piece":         surface_par_piece,
            "surface_terrain":           surface_terrain,
            "longitude":                 lon,
            "latitude":                  lat,
            "annee":                     annee,
            "mois":                      mois,
            "trimestre":                 trimestre,
            "dept_prix_moyen":           dept_prix_moyen,
            "is_maison":                 is_maison,
        }])[features_reg]

        pred_hgb = max(float(arts["hgb"].predict(X_in)[0]), 0)
        pred_lr  = max(float(arts["lin_reg"].predict(X_in)[0]), 0)

        st.success("### Resultats de prediction")
        c1, c2, c3 = st.columns(3)
        c1.metric("HistGradientBoosting", f"{pred_hgb:,.0f} EUR",
                  help="Meilleur modele — non-lineaire")
        c2.metric("Regression Lineaire (log)",  f"{pred_lr:,.0f} EUR",
                  help="Baseline — lineaire sur log(prix)")
        c3.metric("Prix au m² (HGB)",     f"{pred_hgb/max(surface, 1):,.0f} EUR/m²")

        st.info(f"Prix moyen de reference pour le departement **{dept_code}** : "
                f"{dept_prix_moyen:,.0f} EUR")
        st.caption(f"Type : **{type_local}** · {surface} m² · {pieces} pieces · "
                   f"terrain {surface_terrain} m² · dep. {dept_code}.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Demo Classification
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Demo Classification":
    st.title("Demo — Identifier le type de bien")

    arts = load_artifacts()
    if not arts or "log_reg" not in arts:
        st.error("Lance `python save_model.py` puis recharge la page.")
        st.stop()

    features_clf = arts["features_clf"]

    col1, col2 = st.columns(2)
    with col1:
        surface         = st.slider("Surface reelle batie (m²)", 10, 500, 80)
        pieces          = st.slider("Nombre de pieces principales", 1, 10, 4)
        nombre_lots     = st.slider("Nombre de lots", 0, 5, 1)
        valeur_fonciere = st.number_input(
            "Prix de vente (EUR)", min_value=10_000, max_value=10_000_000,
            value=250_000, step=10_000,
        )

    with col2:
        surface_carrez  = st.slider("Surface Carrez (m², 0 si inconnue)", 0, 400, 70)
        surface_terrain = st.slider(
            "Surface terrain (m², 0 = pas de terrain)",
            0, 5000, 0, step=10,
            help="Feature la plus discriminante : ≈ 0 pour appartement, > 0 pour maison.",
        )
        latitude  = st.slider("Latitude",  42.0, 51.0, 46.5, step=0.1)
        longitude = st.slider("Longitude", -5.0, 9.0,   2.5, step=0.1)

    surface_par_piece = surface / max(pieces, 1)

    st.markdown("---")
    if st.button("Identifier le bien", type="primary"):
        X_in = pd.DataFrame([{
            "surface_reelle_bati":       surface,
            "nombre_pieces_principales": pieces,
            "nombre_lots":               nombre_lots,
            "surface_carrez":            surface_carrez,
            "surface_par_piece":         surface_par_piece,
            "surface_terrain":           surface_terrain,
            "valeur_fonciere":           valeur_fonciere,
            "longitude":                 longitude,
            "latitude":                  latitude,
        }])[features_clf]

        pred_lr = int(arts["log_reg"].predict(X_in)[0])
        prob_lr = arts["log_reg"].predict_proba(X_in)[0]
        pred_rf = int(arts["rf_clf"].predict(X_in)[0])
        prob_rf = arts["rf_clf"].predict_proba(X_in)[0]

        label_lr = "🏢 Appartement" if pred_lr == 1 else "🏡 Maison"
        label_rf = "🏢 Appartement" if pred_rf == 1 else "🏡 Maison"

        st.success("### Resultats de classification")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Regression Logistique", label_lr)
            st.progress(float(prob_lr[pred_lr]),
                        text=f"Confiance : {prob_lr[pred_lr]*100:.1f}%")
        with c2:
            st.metric("Random Forest", label_rf)
            st.progress(float(prob_rf[pred_rf]),
                        text=f"Confiance : {prob_rf[pred_rf]*100:.1f}%")

        if pred_lr == pred_rf:
            st.info("Les deux modeles sont d'accord.")
        else:
            st.warning("Les modeles ne sont pas d'accord — cas ambigu.")
