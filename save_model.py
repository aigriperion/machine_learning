"""
Entraine et sauvegarde les artefacts ML pour l'application Streamlit.

Aligne strictement avec projet.ipynb :
  - meme nettoyage v4 (8 filtres anti-outliers)
  - memes features (15 pour la regression, 9 pour la classification)
  - meme pipeline (StandardScaler + LinearRegression encapsules dans un
    TransformedTargetRegressor log/exp)
  - meme target encoding sans fuite (dept + commune avec lissage bayesien)

  - Regression  : LinearRegression (Scaler + log-target) + HistGradientBoosting
                  --> predit valeur_fonciere pour Maison ET Appartement
  - Classification : LogisticRegression (Scaler) + RandomForest
                  --> identifie Appartement vs Maison

Usage : python save_model.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error,
    accuracy_score, classification_report, confusion_matrix,
)

DATA_DIR = Path("datasets")
COLS = [
    "id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere",
    "code_departement", "code_commune", "type_local",
    "surface_reelle_bati", "nombre_pieces_principales",
    "surface_terrain", "nombre_lots", "lot1_surface_carrez",
    "longitude", "latitude",
]

# ── 1. Chargement ──────────────────────────────────────────────────────────
print("Chargement des donnees...")
csv_files = sorted(DATA_DIR.glob("full_202[45].csv"))
df_raw = pd.concat(
    [pd.read_csv(f, usecols=COLS, low_memory=False) for f in csv_files],
    ignore_index=True,
)
print(f"  {len(df_raw):,} lignes brutes")

# ── 2. Nettoyage de base (Vente + Maison|Appartement) ──────────────────────
df = df_raw[
    (df_raw["nature_mutation"] == "Vente")
    & (df_raw["type_local"].isin(["Maison", "Appartement"]))
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
df = df[(df["nombre_pieces_principales"] >= 1) & (df["nombre_pieces_principales"] <= 10)]
df = df[df["latitude"].between(41.0, 51.5) & df["longitude"].between(-5.5, 10.0)]

# ── 2bis. Nettoyage agressif v4 — 8 filtres anti-aberrants ─────────────────
n_init = len(df)
print(f"  {n_init:,} lignes apres nettoyage de base")

# [1] Coherence surface / pieces
df["_spp"] = df["surface_reelle_bati"] / df["nombre_pieces_principales"]
df = df[(df["_spp"] >= 8) & (df["_spp"] <= 60)].drop(columns=["_spp"])

# [2] Prix/m² entre 1 000 et 10 000 EUR/m²
df["_pm2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
df = df[(df["_pm2"] >= 1_000) & (df["_pm2"] <= 10_000)]

# [3] Bornes absolues par type
appart = (df["type_local"] == "Appartement") & df["valeur_fonciere"].between(30_000, 1_500_000)
maison = (df["type_local"] == "Maison")      & df["valeur_fonciere"].between(40_000, 2_000_000)
df = df[appart | maison]

# [4] & [5] IQR x 1.0 par DEPARTEMENT sur valeur_fonciere puis prix/m²
def iqr_filter_by_group(d, grp_col, target_col, k=1.0, min_count=30):
    grp = d.groupby(grp_col)[target_col]
    Q1 = grp.transform("quantile", 0.25)
    Q3 = grp.transform("quantile", 0.75)
    IQR = Q3 - Q1
    counts = grp.transform("count")
    keep = (counts < min_count) | (
        (d[target_col] >= Q1 - k * IQR) & (d[target_col] <= Q3 + k * IQR)
    )
    return d[keep]

df = iqr_filter_by_group(df, "code_departement", "valeur_fonciere", k=1.0)
df = iqr_filter_by_group(df, "code_departement", "_pm2",            k=1.0)
df = df.drop(columns=["_pm2"])

# [6] IQR x 1.5 par TYPE sur surface_reelle_bati
for type_bien in ["Maison", "Appartement"]:
    mask = df["type_local"] == type_bien
    Q1 = df.loc[mask, "surface_reelle_bati"].quantile(0.25)
    Q3 = df.loc[mask, "surface_reelle_bati"].quantile(0.75)
    IQR = Q3 - Q1
    bi, bs = max(Q1 - 1.5 * IQR, 5), Q3 + 1.5 * IQR
    df = df[~(mask & ((df["surface_reelle_bati"] < bi) | (df["surface_reelle_bati"] > bs)))]

# [7] P97 sur surface_terrain des Maisons
mask_mai = df["type_local"] == "Maison"
p97 = df.loc[mask_mai, "surface_terrain"].quantile(0.97)
df = df[~(mask_mai & (df["surface_terrain"] > p97))]

# [8] Z-score > 2.5 sur log(valeur_fonciere) par type
for type_bien in ["Maison", "Appartement"]:
    mask = df["type_local"] == type_bien
    log_p = np.log1p(df.loc[mask, "valeur_fonciere"])
    z = ((log_p - log_p.mean()) / log_p.std()).abs()
    df = df.drop(log_p[z > 2.5].index)

n_final = len(df)
print(f"  {n_final:,} lignes apres nettoyage v4 ({n_final/n_init*100:.1f}% conserve)")

# ── 3. Feature engineering (identique au notebook) ─────────────────────────
df["date_mutation"] = pd.to_datetime(df["date_mutation"])
df["annee"]     = df["date_mutation"].dt.year
df["mois"]      = df["date_mutation"].dt.month
df["trimestre"] = df["date_mutation"].dt.quarter

df["surface_carrez"]      = df["lot1_surface_carrez"].fillna(0)
df["surface_par_piece"]   = df["surface_reelle_bati"] / df["nombre_pieces_principales"].clip(lower=1)
df["log_surface_bati"]    = np.log1p(df["surface_reelle_bati"])
df["log_surface_terrain"] = np.log1p(df["surface_terrain"])
df["is_maison"]           = (df["type_local"] == "Maison").astype(int)

print(f"  Maisons : {(df['is_maison']==1).sum():,}  |  "
      f"Appartements : {(df['is_maison']==0).sum():,}")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE A — REGRESSION (prediction du prix)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── REGRESSION ──")

# 15 features alignees avec le notebook (annee retiree : variance quasi-nulle)
FEATURES_REG = [
    "surface_reelle_bati", "log_surface_bati",
    "nombre_pieces_principales", "nombre_lots",
    "surface_carrez", "surface_par_piece",
    "surface_terrain", "log_surface_terrain",
    "longitude", "latitude",
    "mois", "trimestre",
    "dept_prix_moyen", "commune_prix_moyen",
    "is_maison",
]

# On prepare X sans les target encodings (calcules apres split, sans fuite)
cols_no_te = [c for c in FEATURES_REG if c not in ("dept_prix_moyen", "commune_prix_moyen")]
# On garde code_departement et code_commune dans X pour pouvoir mapper apres split
X = df[cols_no_te + ["code_departement", "code_commune"]].copy()
y = df["valeur_fonciere"].copy()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Target encoding sans fuite ────────────────────────────────────────────
# (1) dept_prix_moyen : moyenne par departement sur le train
dept_mean_train = y_tr.groupby(X_tr["code_departement"]).mean()
fallback_global = float(y_tr.mean())

X_tr["dept_prix_moyen"] = X_tr["code_departement"].map(dept_mean_train)
X_te["dept_prix_moyen"] = X_te["code_departement"].map(dept_mean_train).fillna(fallback_global)

# (2) commune_prix_moyen : lissage bayesien (alpha = 20) sur le train
ALPHA = 20
commune_stats = y_tr.groupby(X_tr["code_commune"]).agg(["mean", "count"])
commune_stats.columns = ["mean", "count"]

# Pour chaque commune du train, retrouver son departement
commune_to_dept = (
    X_tr[["code_commune", "code_departement"]]
    .drop_duplicates(subset="code_commune")
    .set_index("code_commune")["code_departement"]
)
commune_stats["dept"]      = commune_stats.index.map(commune_to_dept)
commune_stats["dept_mean"] = commune_stats["dept"].map(dept_mean_train).fillna(fallback_global)

n = commune_stats["count"]
commune_stats["smoothed"] = (
    n * commune_stats["mean"] + ALPHA * commune_stats["dept_mean"]
) / (n + ALPHA)
commune_smoothed_map = commune_stats["smoothed"]

# Application au train et test (cascade : commune -> dept -> global)
X_tr["commune_prix_moyen"] = X_tr["code_commune"].map(commune_smoothed_map)
X_te["commune_prix_moyen"] = (
    X_te["code_commune"].map(commune_smoothed_map)
    .fillna(X_te["code_departement"].map(dept_mean_train))
    .fillna(fallback_global)
)

# On selectionne les 15 features dans le bon ordre
X_tr = X_tr[FEATURES_REG]
X_te = X_te[FEATURES_REG]

print(f"  Train : {len(X_tr):,} lignes  |  Test : {len(X_te):,} lignes")
print(f"  Manquants train : {X_tr.isnull().sum().sum()}  "
      f"|  test : {X_te.isnull().sum().sum()}")

# ── Modele 1 : Regression Lineaire (StandardScaler + log-target) ──────────
# Pipeline = StandardScaler -> LinearRegression, encapsule dans un
# TransformedTargetRegressor qui applique log1p a y (et expm1 en sortie).
pipeline_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("reg",    LinearRegression()),
])
lin_reg = TransformedTargetRegressor(
    regressor=pipeline_reg,
    func=np.log1p, inverse_func=np.expm1,
)
lin_reg.fit(X_tr, y_tr)
y_pred_lr = np.clip(lin_reg.predict(X_te), 0, 10_000_000)

r2_lr   = r2_score(y_te, y_pred_lr)
mae_lr  = mean_absolute_error(y_te, y_pred_lr)
rmse_lr = float(np.sqrt(mean_squared_error(y_te, y_pred_lr)))
mape_lr = mean_absolute_percentage_error(y_te, y_pred_lr) * 100
print(f"  Regression Lineaire (Scaler + log)  — "
      f"R2={r2_lr:.3f}  MAE={mae_lr:,.0f}EUR  MAPE={mape_lr:.1f}%")

# ── Modele 2 : HistGradientBoosting (non-lineaire, meilleur) ──────────────
hgb = HistGradientBoostingRegressor(
    max_iter=500, max_depth=10, learning_rate=0.1,
    min_samples_leaf=20, random_state=42,
)
hgb.fit(X_tr, y_tr)
y_pred_hgb = np.clip(hgb.predict(X_te), 0, 10_000_000)
r2_hgb   = r2_score(y_te, y_pred_hgb)
mae_hgb  = mean_absolute_error(y_te, y_pred_hgb)
rmse_hgb = float(np.sqrt(mean_squared_error(y_te, y_pred_hgb)))
mape_hgb = mean_absolute_percentage_error(y_te, y_pred_hgb) * 100
print(f"  HistGradientBoosting                — "
      f"R2={r2_hgb:.3f}  MAE={mae_hgb:,.0f}EUR  MAPE={mape_hgb:.1f}%")

# ── Coordonnees medianes par departement (auto-remplit lat/lon dans la demo)
dept_coords = df.groupby("code_departement")[["latitude", "longitude"]].median()

# ── Echantillon predictions vs realite (pour le scatter dans l'app) ───────
sample_idx = np.random.RandomState(42).choice(
    len(y_te), size=min(5000, len(y_te)), replace=False
)
reg_preview = pd.DataFrame({
    "y_true":     y_te.values[sample_idx],
    "y_pred_lr":  y_pred_lr[sample_idx],
    "y_pred_hgb": y_pred_hgb[sample_idx],
})

reg_metrics = {
    "lin_reg": {"R2": r2_lr,  "MAE": mae_lr,  "RMSE": rmse_lr,  "MAPE": mape_lr},
    "hgb":     {"R2": r2_hgb, "MAE": mae_hgb, "RMSE": rmse_hgb, "MAPE": mape_hgb},
    "n_train": int(len(X_tr)),
    "n_test":  int(len(X_te)),
}

joblib.dump(hgb,                  "model.joblib")
joblib.dump(lin_reg,              "model_linreg.joblib")
joblib.dump(dept_mean_train,      "dept_encoding.joblib")
joblib.dump(commune_smoothed_map, "commune_encoding.joblib")  # pour ref / debug
joblib.dump(fallback_global,      "fallback.joblib")
joblib.dump(FEATURES_REG,         "features.joblib")
joblib.dump(dept_coords,          "dept_coords.joblib")
joblib.dump(reg_metrics,          "reg_metrics.joblib")
joblib.dump(reg_preview,          "reg_preview.joblib")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE B — CLASSIFICATION (Appartement vs Maison)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── CLASSIFICATION ──")

FEATURES_CLF = [
    "surface_reelle_bati", "nombre_pieces_principales", "nombre_lots",
    "surface_carrez", "surface_par_piece", "surface_terrain",
    "valeur_fonciere", "longitude", "latitude",
]

# 1 = Appartement, 0 = Maison (cohere avec l'app)
df["label"] = (df["type_local"] == "Appartement").astype(int)
print(f"  Appartements : {(df['label']==1).sum():,}  |  "
      f"Maisons : {(df['label']==0).sum():,}")

X_clf = df[FEATURES_CLF].copy()
y_clf = df["label"].copy()

X_c_tr, X_c_te, y_c_tr, y_c_te = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf,
)

# Regression Logistique avec StandardScaler (Pipeline)
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
])
log_reg.fit(X_c_tr, y_c_tr)
y_pred_lr_clf = log_reg.predict(X_c_te)
acc_lr  = accuracy_score(y_c_te, y_pred_lr_clf)
cm_lr   = confusion_matrix(y_c_te, y_pred_lr_clf, labels=[0, 1])
report_lr = classification_report(
    y_c_te, y_pred_lr_clf, target_names=["Maison", "Appartement"],
    output_dict=True, digits=3,
)
print(f"  Regression Logistique — Accuracy={acc_lr:.3f}")

# Random Forest (insensible a l'echelle, pas de scaler)
rf_clf = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=42,
)
rf_clf.fit(X_c_tr, y_c_tr)
y_pred_rf = rf_clf.predict(X_c_te)
acc_rf  = accuracy_score(y_c_te, y_pred_rf)
cm_rf   = confusion_matrix(y_c_te, y_pred_rf, labels=[0, 1])
report_rf = classification_report(
    y_c_te, y_pred_rf, target_names=["Maison", "Appartement"],
    output_dict=True, digits=3,
)
print(f"  Random Forest         — Accuracy={acc_rf:.3f}")

clf_metrics = {
    "log_reg": {"accuracy": acc_lr, "confusion_matrix": cm_lr.tolist(), "report": report_lr},
    "rf":      {"accuracy": acc_rf, "confusion_matrix": cm_rf.tolist(), "report": report_rf},
    "n_train": int(len(X_c_tr)),
    "n_test":  int(len(X_c_te)),
    "rf_feature_importances": dict(zip(FEATURES_CLF, rf_clf.feature_importances_.tolist())),
}

joblib.dump(log_reg,       "model_logreg.joblib")
joblib.dump(rf_clf,        "model_rf_clf.joblib")
joblib.dump(FEATURES_CLF,  "features_clf.joblib")
joblib.dump(clf_metrics,   "clf_metrics.joblib")

print("\nOK — tous les artefacts sauvegardes.")