"""
Entraine et sauvegarde les artefacts ML pour l'application Streamlit.

  - Regression  : LinearRegression (log-target) + HistGradientBoosting
                  --> predit valeur_fonciere pour Maison ET Appartement
  - Classification : LogisticRegression + RandomForest
                  --> identifie Appartement vs Maison

Aligne avec projet.ipynb (memes features, meme split, meme target encoding sans fuite).

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
    "code_departement", "type_local",
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

# ── 2. Filtrage / nettoyage commun (Vente + Maison|Appartement) ────────────
df = df_raw[
    (df_raw["nature_mutation"] == "Vente")
    & (df_raw["type_local"].isin(["Maison", "Appartement"]))
].copy()
df = df.dropna(subset=["valeur_fonciere"])
df = df[(df["valeur_fonciere"] >= 10_000) & (df["valeur_fonciere"] <= 10_000_000)]
df = df[(df["surface_reelle_bati"] > 0) & (df["surface_reelle_bati"] <= 500)]
df = df.sort_values("surface_reelle_bati", ascending=False).drop_duplicates("id_mutation", keep="first")
df = df.dropna(subset=["longitude", "latitude"])
df["nombre_pieces_principales"] = df["nombre_pieces_principales"].fillna(
    df["nombre_pieces_principales"].median()
)
df["surface_terrain"] = df["surface_terrain"].fillna(0)

# Feature engineering (identique au notebook)
df["date_mutation"] = pd.to_datetime(df["date_mutation"])
df["annee"]     = df["date_mutation"].dt.year
df["mois"]      = df["date_mutation"].dt.month
df["trimestre"] = df["date_mutation"].dt.quarter
df["surface_carrez"]    = df["lot1_surface_carrez"].fillna(0)
df["surface_par_piece"] = df["surface_reelle_bati"] / df["nombre_pieces_principales"].clip(lower=1)
df["is_maison"]         = (df["type_local"] == "Maison").astype(int)

print(f"  {len(df):,} lignes apres nettoyage  "
      f"(Maisons : {(df['is_maison']==1).sum():,}  |  "
      f"Appartements : {(df['is_maison']==0).sum():,})")

# ═══════════════════════════════════════════════════════════════════════════
# PARTIE A — REGRESSION (prediction du prix)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── REGRESSION ──")

FEATURES_REG = [
    "surface_reelle_bati", "nombre_pieces_principales", "nombre_lots",
    "surface_carrez", "surface_par_piece", "surface_terrain",
    "longitude", "latitude",
    "annee", "mois", "trimestre",
    "dept_prix_moyen", "is_maison",
]

cols_reg_no_dept = [c for c in FEATURES_REG if c != "dept_prix_moyen"]
X = df[cols_reg_no_dept + ["code_departement"]].copy()
y = df["valeur_fonciere"].copy()

# Placeholder dept_prix_moyen (sera recalcule sur le train apres split)
X["dept_prix_moyen"] = np.nan

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Target encoding sans fuite : recalcul sur le train uniquement
dept_mean_train = y_tr.groupby(X_tr["code_departement"]).mean()
fallback = float(y_tr.mean())
X_tr["dept_prix_moyen"] = X_tr["code_departement"].map(dept_mean_train)
X_te["dept_prix_moyen"] = X_te["code_departement"].map(dept_mean_train).fillna(fallback)

X_tr = X_tr[FEATURES_REG]
X_te = X_te[FEATURES_REG]

# Regression Lineaire avec log-target (modele du notebook)
lin_reg = TransformedTargetRegressor(
    regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1,
)
lin_reg.fit(X_tr, y_tr)
y_pred_lr = np.clip(lin_reg.predict(X_te), 0, 10_000_000)  # plafond du domaine
r2_lr   = r2_score(y_te, y_pred_lr)
mae_lr  = mean_absolute_error(y_te, y_pred_lr)
rmse_lr = float(np.sqrt(mean_squared_error(y_te, y_pred_lr)))
mape_lr = mean_absolute_percentage_error(y_te, y_pred_lr) * 100
print(f"  Regression Lineaire (log)  — R2={r2_lr:.3f}  MAE={mae_lr:,.0f}EUR  MAPE={mape_lr:.1f}%")

# HistGradientBoosting (modele non-lineaire, meilleur)
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
print(f"  HistGradientBoosting       — R2={r2_hgb:.3f}  MAE={mae_hgb:,.0f}EUR  MAPE={mape_hgb:.1f}%")

# Coordonnees medianes par departement (pour l'app : auto-remplit lat/lon depuis le code dep)
dept_coords = df.groupby("code_departement")[["latitude", "longitude"]].median()

# Sauvegarde echantillon (predictions vs realite) pour le scatter dans l'app
sample_idx = np.random.RandomState(42).choice(len(y_te), size=min(5000, len(y_te)), replace=False)
reg_preview = pd.DataFrame({
    "y_true":    y_te.values[sample_idx],
    "y_pred_lr": y_pred_lr[sample_idx],
    "y_pred_hgb": y_pred_hgb[sample_idx],
})

reg_metrics = {
    "lin_reg":  {"R2": r2_lr,  "MAE": mae_lr,  "RMSE": rmse_lr,  "MAPE": mape_lr},
    "hgb":      {"R2": r2_hgb, "MAE": mae_hgb, "RMSE": rmse_hgb, "MAPE": mape_hgb},
    "n_train":  int(len(X_tr)),
    "n_test":   int(len(X_te)),
}

joblib.dump(hgb,             "model.joblib")
joblib.dump(lin_reg,         "model_linreg.joblib")
joblib.dump(dept_mean_train, "dept_encoding.joblib")
joblib.dump(fallback,        "fallback.joblib")
joblib.dump(FEATURES_REG,    "features.joblib")
joblib.dump(dept_coords,     "dept_coords.joblib")
joblib.dump(reg_metrics,     "reg_metrics.joblib")
joblib.dump(reg_preview,     "reg_preview.joblib")

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

# Regression logistique (avec StandardScaler)
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

# Random Forest
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
