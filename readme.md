# 🏠 Prédiction du prix des appartements — DVF

## Description du projet

Ce projet de Machine Learning a pour objectif de **prédire le prix de vente des appartements** en France à partir du jeu de données ouvert **DVF (Demandes de Valeurs Foncières)**, publié par la Direction Générale des Finances Publiques (DGFiP).

Les données recensent l'ensemble des transactions immobilières réalisées sur le territoire métropolitain et les DOM-TOM (hors Alsace-Moselle et Mayotte), issues des actes notariés et du cadastre.

## Source des données

- **Jeu de données** : DVF géolocalisées (Etalab)
- **URL** : https://files.data.gouv.fr/geo-dvf/latest/csv/
- **Format** : CSV (séparateur virgule, encodage UTF-8)
- **Granularité** : par département et par commune, organisé par année

## Structure du dataset

Le fichier CSV contient **39 colonnes**. Les principales variables utiles pour la prédiction sont :

| Variable | Description | Type |
|---|---|---|
| `id_mutation` | Identifiant unique de la transaction | string |
| `date_mutation` | Date de la vente (format ISO-8601) | date |
| `nature_mutation` | Type de mutation (Vente, VEFA, Échange…) | catégorielle |
| `valeur_fonciere` | **Prix de vente (€)** — variable cible 🎯 | numérique |
| `code_postal` | Code postal du bien | string |
| `code_commune` | Code INSEE de la commune | string |
| `nom_commune` | Nom de la commune | string |
| `code_departement` | Code du département | string |
| `type_local` | Type de bien (Appartement, Maison, Dépendance, Local) | catégorielle |
| `surface_reelle_bati` | Surface du bâti en m² | numérique |
| `nombre_pieces_principales` | Nombre de pièces principales | numérique |
| `surface_terrain` | Surface du terrain en m² | numérique |
| `longitude` | Longitude (WGS-84) | numérique |
| `latitude` | Latitude (WGS-84) | numérique |
| `lot1_surface_carrez` à `lot5_surface_carrez` | Surface Carrez des lots | numérique |
| `nombre_lots` | Nombre de lots dans la transaction | numérique |

## Pipeline du projet

### 1. Chargement et exploration des données

```python
import pandas as pd

df = pd.read_csv("full.csv", low_memory=False)
print(df.shape)
df.info()
df.describe()
```

### 2. Filtrage et nettoyage

- Filtrer uniquement les **ventes** (`nature_mutation == "Vente"`) et les **appartements** (`type_local == "Appartement"`)
- Supprimer les lignes avec `valeur_fonciere` manquante ou aberrante (ex : < 10 000 € ou > 10 000 000 €)
- Supprimer les doublons liés aux mutations multi-lots (agréger par `id_mutation`)
- Filtrer les surfaces incohérentes (`surface_reelle_bati > 0` et < 500 m²)
- Gérer les valeurs manquantes (latitude, longitude, nombre de pièces)

### 3. Feature engineering

- **Prix au m²** : `valeur_fonciere / surface_reelle_bati` (indicateur utile pour l'analyse)
- **Extraction temporelle** : année, mois, trimestre à partir de `date_mutation`
- **Géolocalisation** : utiliser `latitude` et `longitude` comme features, ou créer des clusters géographiques
- **Encodage** : encoder `code_departement` ou `code_commune` (label encoding, target encoding…)

### 4. Modélisation

Modèles à explorer :

- **Régression linéaire** : baseline
- **Random Forest** : robuste aux outliers
- **Gradient Boosting** (XGBoost / LightGBM) : souvent le plus performant sur ce type de données tabulaires
- **Ridge / Lasso** : si multicolinéarité

### 5. Évaluation

Métriques utilisées :

- **MAE** (Mean Absolute Error) : erreur moyenne en euros
- **RMSE** (Root Mean Squared Error) : pénalise davantage les grosses erreurs
- **R²** (coefficient de détermination) : proportion de la variance expliquée

Validation croisée (k-fold) pour estimer la robustesse du modèle.

## Prérequis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm
```

## Structure du projet

```
├── README.md
├── data/
│   └── full.csv              # Données DVF (non versionnées)
├── notebooks/
│   ├── 01_exploration.ipynb   # EDA et visualisations
│   ├── 02_preprocessing.ipynb # Nettoyage et feature engineering
│   └── 03_modelisation.ipynb  # Entraînement et évaluation
├── src/
│   ├── preprocessing.py       # Fonctions de nettoyage
│   └── models.py              # Entraînement des modèles
└── requirements.txt
```

## Auteur

Projet réalisé dans le cadre d'un TP de Machine Learning — ESAIP.

## Licence

Les données DVF sont en **open data** sous [Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence).

> ⚠️ L'utilisation des données DVF ne doit pas permettre la ré-identification des personnes concernées, ni faire l'objet d'une indexation par les moteurs de recherche.