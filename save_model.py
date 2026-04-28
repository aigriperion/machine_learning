"""
Charge les données 2024-2025, entraîne le meilleur modèle et sauvegarde les artefacts.
Usage : python save_model.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path("datasets")
COLS = [
    "id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere",
    "code_postal", "code_departement", "type_local",
    "surface_reelle_bati", "nombre_pieces_principales",
    "nombre_lots", "lot1_surface_carrez", "longitude", "latitude",
]
FEATURES = [
    "surface_reelle_bati", "nombre_pieces_principales", "nombre_lots",
    "surface_carrez", "surface_par_piece", "longitude", "latitude",
    "annee", "mois", "trimestre", "dept_prix_moyen",
]

print("Chargement des données...")
csv_files = sorted(DATA_DIR.glob("full_202[45].csv"))
df = pd.concat([pd.read_csv(f, usecols=COLS, low_memory=False) for f in csv_files], ignore_index=True)
print(f"  {len(df):,} lignes brutes")

print("Nettoyage...")
df = df[(df["nature_mutation"] == "Vente") & (df["type_local"] == "Appartement")].copy()
df = df.dropna(subset=["valeur_fonciere"])
df = df[(df["valeur_fonciere"] >= 10_000) & (df["valeur_fonciere"] <= 10_000_000)]
df = df[(df["surface_reelle_bati"] > 0) & (df["surface_reelle_bati"] <= 500)]
df = df.sort_values("surface_reelle_bati", ascending=False).drop_duplicates("id_mutation", keep="first")
df = df.dropna(subset=["longitude", "latitude"])
df["nombre_pieces_principales"] = df["nombre_pieces_principales"].fillna(df["nombre_pieces_principales"].median())

print("Feature engineering...")
df["date_mutation"] = pd.to_datetime(df["date_mutation"])
df["annee"]     = df["date_mutation"].dt.year
df["mois"]      = df["date_mutation"].dt.month
df["trimestre"] = df["date_mutation"].dt.quarter
df["surface_carrez"]    = df["lot1_surface_carrez"].fillna(0)
df["surface_par_piece"] = df["surface_reelle_bati"] / df["nombre_pieces_principales"].clip(lower=1)

dept_mean_eda = df.groupby("code_departement")["valeur_fonciere"].mean()
df["dept_prix_moyen"] = df["code_departement"].map(dept_mean_eda)
print(f"  {len(df):,} lignes propres")

print("Split train/test...")
X = df[FEATURES].copy()
y = df["valeur_fonciere"].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Target encoding sans fuite
dept_code_train = df.loc[X_train.index, "code_departement"]
dept_mean_train = y_train.groupby(dept_code_train).mean()
fallback = float(y_train.mean())
X_train["dept_prix_moyen"] = df.loc[X_train.index, "code_departement"].map(dept_mean_train)
X_test["dept_prix_moyen"]  = df.loc[X_test.index,  "code_departement"].map(dept_mean_train).fillna(fallback)

print("Entraînement HistGradientBoosting...")
model = HistGradientBoostingRegressor(
    max_iter=500, max_depth=10, learning_rate=0.1, min_samples_leaf=20, random_state=42
)
model.fit(X_train, y_train)

from sklearn.metrics import r2_score, mean_absolute_percentage_error
y_pred = model.predict(X_test)
r2   = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"  R² = {r2:.4f}  |  MAPE = {mape:.1f}%")

print("Sauvegarde des artefacts...")
dept_coords = df.groupby("code_departement")[["latitude", "longitude"]].median()
joblib.dump(model,          "model.joblib")
joblib.dump(dept_mean_train,"dept_encoding.joblib")
joblib.dump(fallback,       "fallback.joblib")
joblib.dump(FEATURES,       "features.joblib")
joblib.dump(dept_coords,    "dept_coords.joblib")
print("OK — model.joblib, dept_encoding.joblib, dept_coords.joblib, features.joblib, fallback.joblib")
