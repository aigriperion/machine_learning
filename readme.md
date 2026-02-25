# Prediction du prix des appartements — DVF

## Description du projet

Ce projet de Machine Learning a pour objectif de **predire le prix de vente des appartements** en France a partir du jeu de donnees ouvert **DVF (Demandes de Valeurs Foncieres)**, publie par la Direction Generale des Finances Publiques (DGFiP).

Les donnees recensent l'ensemble des transactions immobilieres realisees sur le territoire metropolitain et les DOM-TOM (hors Alsace-Moselle et Mayotte), issues des actes notaries et du cadastre.

## Source des donnees

- **Jeu de donnees** : DVF geolocalisees (Etalab)
- **URL** : https://files.data.gouv.fr/geo-dvf/latest/csv/
- **Format** : CSV (separateur virgule, encodage UTF-8)
- **Granularite** : par departement et par commune, organise par annee
- **Annees** : 2020 a 2025

## Structure du dataset

Le fichier CSV contient **39 colonnes**. Les principales variables utiles pour la prediction sont :

| Variable | Description | Type |
|---|---|---|
| `id_mutation` | Identifiant unique de la transaction | string |
| `date_mutation` | Date de la vente (format ISO-8601) | date |
| `nature_mutation` | Type de mutation (Vente, VEFA, Echange…) | categorielle |
| `valeur_fonciere` | **Prix de vente** — variable cible | numerique |
| `code_postal` | Code postal du bien | string |
| `code_commune` | Code INSEE de la commune | string |
| `nom_commune` | Nom de la commune | string |
| `code_departement` | Code du departement | string |
| `type_local` | Type de bien (Appartement, Maison, Dependance, Local) | categorielle |
| `surface_reelle_bati` | Surface du bati en m2 | numerique |
| `nombre_pieces_principales` | Nombre de pieces principales | numerique |
| `surface_terrain` | Surface du terrain en m2 | numerique |
| `longitude` | Longitude (WGS-84) | numerique |
| `latitude` | Latitude (WGS-84) | numerique |
| `lot1_surface_carrez` a `lot5_surface_carrez` | Surface Carrez des lots | numerique |
| `nombre_lots` | Nombre de lots dans la transaction | numerique |

## Pipeline du projet

### 1. Chargement et exploration des donnees
- Chargement des 6 fichiers CSV annuels (2020-2025) depuis `datasets/`
- Selection des colonnes utiles pour limiter l'usage memoire
- Statistiques descriptives et analyse des valeurs manquantes

### 2. Filtrage et nettoyage
- Filtrer uniquement les **ventes** (`nature_mutation == "Vente"`) et les **appartements** (`type_local == "Appartement"`)
- Supprimer les lignes avec `valeur_fonciere` manquante ou aberrante (< 10 000 EUR ou > 10 000 000 EUR)
- Supprimer les doublons lies aux mutations multi-lots (deduplication par `id_mutation`)
- Filtrer les surfaces incoherentes (`surface_reelle_bati > 0` et <= 500 m2)
- Supprimer les lignes sans coordonnees GPS
- Imputer les valeurs manquantes (nombre de pieces par la mediane)

### 3. Feature engineering
- **Extraction temporelle** : annee, mois, trimestre a partir de `date_mutation`
- **Prix au m2** : `valeur_fonciere / surface_reelle_bati` (indicateur exploratoire)
- **Surface Carrez** : surface du lot principal
- **Target encoding** : prix moyen par departement

### 4. Analyse exploratoire (EDA)
- Distribution de la variable cible (brute et log)
- Prix median au m2 par departement (Top 15)
- Prix median par nombre de pieces
- Evolution trimestrielle des prix
- Matrice de correlation
- Scatter surface vs prix

### 5. Modelisation
Modeles entraines :
- **Regression lineaire** : baseline
- **Ridge** : regularisation L2
- **Lasso** : regularisation L1
- **Random Forest** : robuste aux outliers
- **HistGradientBoosting** : equivalent LightGBM integre a scikit-learn

### 6. Evaluation
Metriques utilisees :
- **MAE** (Mean Absolute Error) : erreur moyenne en euros
- **RMSE** (Root Mean Squared Error) : penalise davantage les grosses erreurs
- **R2** (coefficient de determination) : proportion de la variance expliquee
- **Validation croisee** (5-fold) pour estimer la robustesse du modele
- Analyse de l'importance des features (permutation importance)

## Prerequis

```bash
pip install -r requirements.txt
```

## Structure du projet

```
├── readme.md
├── requirements.txt
├── .gitignore
├── projet.ipynb           # Notebook complet (EDA + modelisation + evaluation)
└── datasets/              # Donnees DVF (non versionnees)
    ├── full_2020.csv
    ├── full_2021.csv
    ├── full_2022.csv
    ├── full_2023.csv
    ├── full_2024.csv
    └── full_2025.csv
```
