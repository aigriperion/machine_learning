"""
Entraîne et sauvegarde :
  - Régression  : HistGradientBoosting pour prédire le prix (valeur_fonciere)
  - Classification : LogisticRegression + RandomForest pour identifier Appartement vs Maison

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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_absolute_percentage_error,
    accuracy_score, classification_report,
)

DATA_DIR = Path("datasets")
COLS = [
    "id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere",
    "code_departement", "type_local",
    "surface_reelle_bati", "nombre_pieces_principales",
    "nombre_lots", "lot1_surface_carrez", "longitude", "latitude",
]

# ── 1. Chargement ──────────────────────────────────────────────────────────
print("Chargement des données...")
csv_files = sorted(DATA_DIR.glob("full_202[45].csv"))
df_raw = pd.concat(
    [pd.read_csv(f, usecols=COLS, low_memory=False) for f in csv_files],
    ignore_index=True,
)
print(f"  {len(df_raw):,} lignes brutes")

# ── 2. Préparation commune ─────────────────────────────────────────────────
df_raw["date_mutation"] = pd.to_datetime(df_raw["date_mutation"])
df_raw["annee"]     = df_raw["date_mutation"].dt.year
df_raw["mois"]      = df_raw["date_mutation"].dt.month
df_raw["trimestre"] = df_raw["date_mutation"].dt.quarter
df_raw["surface_carrez"]    = df_raw["lot1_surface_carrez"].fillna(0)
df_raw["nombre_pieces_principales"] = df_raw["nombre_pieces_principales"].fillna(
    df_raw["nombre_pieces_principales"].median()
)
df_raw = df_raw.dropna(subset=["longitude", "latitude", "surface_reelle_bati"])
df_raw = df_raw[(df_raw["surface_reelle_bati"] > 0) & (df_raw["surface_reelle_bati"] <= 500)]

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE A — RÉGRESSION (prédiction du prix des appartements)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── RÉGRESSION ──")

FEATURES_REG = [
    "surface_reelle_bati", "nombre_pieces_principales", "nombre_lots",
    "surface_carrez", "surface_par_piece", "longitude", "latitude",
    "annee", "mois", "trimestre", "dept_prix_moyen",
]

df_reg = df_raw[
    (df_raw["nature_mutation"] == "Vente") &
    (df_raw["type_local"] == "Appartement")
].copy()
df_reg = df_reg.dropna(subset=["valeur_fonciere"])
df_reg = df_reg[(df_reg["valeur_fonciere"] >= 10_000) & (df_reg["valeur_fonciere"] <= 10_000_000)]
df_reg = df_reg.sort_values("surface_reelle_bati", ascending=False).drop_duplicates("id_mutation", keep="first")

df_reg["surface_par_piece"] = df_reg["surface_reelle_bati"] / df_reg["nombre_pieces_principales"].clip(lower=1)
dept_mean_eda = df_reg.groupby("code_departement")["valeur_fonciere"].mean()
df_reg["dept_prix_moyen"] = df_reg["code_departement"].map(dept_mean_eda)

X_reg = df_reg[FEATURES_REG].copy()
y_reg = df_reg["valeur_fonciere"].copy()

X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Target encoding sans fuite
dept_code_train = df_reg.loc[X_tr.index, "code_departement"]
dept_mean_train = y_tr.groupby(dept_code_train).mean()
fallback = float(y_tr.mean())
X_tr["dept_prix_moyen"] = df_reg.loc[X_tr.index, "code_departement"].map(dept_mean_train)
X_te["dept_prix_moyen"] = df_reg.loc[X_te.index, "code_departement"].map(dept_mean_train).fillna(fallback)

# Régression linéaire (baseline)
lr = LinearRegression()
lr.fit(X_tr, y_tr)
y_pred_lr = np.clip(lr.predict(X_te), 0, None)
r2_lr   = r2_score(y_te, y_pred_lr)
mae_lr  = mean_absolute_error(y_te, y_pred_lr)
mape_lr = mean_absolute_percentage_error(y_te, y_pred_lr) * 100
print(f"  Régression Linéaire  — R²={r2_lr:.3f}  MAE={mae_lr:,.0f}€  MAPE={mape_lr:.1f}%")

# HistGradientBoosting (meilleur modèle)
hgb = HistGradientBoostingRegressor(
    max_iter=500, max_depth=10, learning_rate=0.1, min_samples_leaf=20, random_state=42
)
hgb.fit(X_tr, y_tr)
y_pred_hgb = hgb.predict(X_te)
r2_hgb   = r2_score(y_te, y_pred_hgb)
mae_hgb  = mean_absolute_error(y_te, y_pred_hgb)
mape_hgb = mean_absolute_percentage_error(y_te, y_pred_hgb) * 100
print(f"  HistGradientBoosting — R²={r2_hgb:.3f}  MAE={mae_hgb:,.0f}€  MAPE={mape_hgb:.1f}%")

dept_coords = df_reg.groupby("code_departement")[["latitude", "longitude"]].median()

reg_metrics = {
    "lin_reg":  {"R2": r2_lr,  "MAE": mae_lr,  "MAPE": mape_lr},
    "hgb":      {"R2": r2_hgb, "MAE": mae_hgb, "MAPE": mape_hgb},
}

# Sauvegarde régression
joblib.dump(hgb,            "model.joblib")
joblib.dump(lr,             "model_linreg.joblib")
joblib.dump(dept_mean_train,"dept_encoding.joblib")
joblib.dump(fallback,       "fallback.joblib")
joblib.dump(FEATURES_REG,   "features.joblib")
joblib.dump(dept_coords,    "dept_coords.joblib")
joblib.dump(reg_metrics,    "reg_metrics.joblib")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE B — CLASSIFICATION (Appartement vs Maison)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── CLASSIFICATION ──")

FEATURES_CLF = [
    "surface_reelle_bati", "nombre_pieces_principales", "nombre_lots",
    "surface_carrez", "longitude", "latitude",
]

df_clf = df_raw[
    (df_raw["nature_mutation"] == "Vente") &
    (df_raw["type_local"].isin(["Appartement", "Maison"]))
].copy()
df_clf = df_clf.sort_values("surface_reelle_bati", ascending=False).drop_duplicates("id_mutation", keep="first")
df_clf["label"] = (df_clf["type_local"] == "Appartement").astype(int)  # 1=Appart, 0=Maison

print(f"  Appartements : {(df_clf['label']==1).sum():,}  |  Maisons : {(df_clf['label']==0).sum():,}")

X_clf = df_clf[FEATURES_CLF].copy()
y_clf = df_clf["label"].copy()

X_c_tr, X_c_te, y_c_tr, y_c_te = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# Régression logistique (avec StandardScaler)
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=500, random_state=42)),
])
log_reg.fit(X_c_tr, y_c_tr)
acc_lr = accuracy_score(y_c_te, log_reg.predict(X_c_te))
print(f"  Régression Logistique — Accuracy={acc_lr:.3f}")
print(classification_report(y_c_te, log_reg.predict(X_c_te),
                             target_names=["Maison", "Appartement"], digits=3))

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf_clf.fit(X_c_tr, y_c_tr)
acc_rf = accuracy_score(y_c_te, rf_clf.predict(X_c_te))
print(f"  Random Forest         — Accuracy={acc_rf:.3f}")
print(classification_report(y_c_te, rf_clf.predict(X_c_te),
                             target_names=["Maison", "Appartement"], digits=3))

clf_metrics = {
    "log_reg": {"accuracy": acc_lr},
    "rf":      {"accuracy": acc_rf},
}

# Sauvegarde classification
joblib.dump(log_reg,       "model_logreg.joblib")
joblib.dump(rf_clf,        "model_rf_clf.joblib")
joblib.dump(FEATURES_CLF,  "features_clf.joblib")
joblib.dump(clf_metrics,   "clf_metrics.joblib")

print("\n✓ Tous les artefacts sauvegardés.")
