# Prediction du prix immobilier — DVF

## Description du projet

Ce projet de Machine Learning a deux objectifs sur le jeu de donnees ouvert
**DVF (Demandes de Valeurs Foncieres)**, publie par la Direction Generale des
Finances Publiques (DGFiP) :

1. **Regression** — predire le **prix de vente** (`valeur_fonciere`) d'un bien
   a partir de ses caracteristiques.
2. **Classification** — identifier le **type de bien** (Maison vs Appartement)
   a partir de ses caracteristiques physiques et geographiques.

Le projet est livre avec un notebook complet (EDA + modelisation) et une
**application Streamlit** servant de support de soutenance.

## Source des donnees

- **Jeu de donnees** : DVF geolocalisees (Etalab)
- **URL** : https://files.data.gouv.fr/geo-dvf/latest/csv/
- **Format** : CSV (separateur virgule, encodage UTF-8)
- **Granularite** : par departement et par commune, organise par annee
- **Annees utilisees** : **2024 + 2025** (volume restreint pour entrainement
  rapide, et donnees plus representatives du marche actuel)

## Structure du dataset

Le fichier CSV contient **39 colonnes**. Principales variables utilisees :

| Variable | Description | Type |
|---|---|---|
| `id_mutation` | Identifiant unique de la transaction | string |
| `date_mutation` | Date de la vente (ISO-8601) | date |
| `nature_mutation` | Type de mutation (Vente, VEFA, Echange...) | categorielle |
| `valeur_fonciere` | **Prix de vente** — variable cible (regression) | numerique |
| `code_departement` | Code du departement | string |
| `type_local` | Type de bien — **cible classification** (Maison/Appartement) | categorielle |
| `surface_reelle_bati` | Surface du bati en m² | numerique |
| `nombre_pieces_principales` | Nombre de pieces principales | numerique |
| `surface_terrain` | Surface du terrain en m² (≈ 0 pour appartements) | numerique |
| `lot1_surface_carrez` | Surface Carrez du lot principal | numerique |
| `nombre_lots` | Nombre de lots dans la transaction | numerique |
| `longitude`, `latitude` | Coordonnees GPS (WGS-84) | numerique |

## Pipeline du projet

### 1. Chargement et exploration
Chargement des CSV 2024 + 2025 depuis `datasets/`, statistiques descriptives,
analyse des valeurs manquantes.

### 2. Filtrage et nettoyage
- Garder uniquement les **Ventes** de **Maisons ou Appartements**
- `valeur_fonciere` entre 10 000 EUR et 10 000 000 EUR
- `surface_reelle_bati` entre 0 et 500 m²
- Deduplication par `id_mutation` (lot a plus grande surface)
- Suppression des lignes sans GPS
- Imputation : pieces (mediane), surface_terrain (0)

### 3. Feature engineering
- Variables temporelles (`annee`, `mois`, `trimestre`)
- `surface_carrez` (lot principal), `surface_par_piece`
- `is_maison` (binaire) pour la regression
- **Target encoding** : prix moyen par departement, calcule **uniquement sur
  le train** pour eviter toute fuite de donnees

### 4. Modelisation
- **Regression** :
  - `LinearRegression` avec `TransformedTargetRegressor` (log de la cible)
  - `HistGradientBoostingRegressor` (modele non-lineaire, meilleur)
- **Classification** :
  - `LogisticRegression` (avec `StandardScaler`)
  - `RandomForestClassifier`

### 5. Evaluation
- **Regression** : MAE, RMSE, MAPE, R² + validation croisee 5-fold
- **Classification** : Accuracy, matrice de confusion, classification report,
  importance des features (Random Forest)

## Prerequis & installation

```bash
# 1) Cloner et entrer dans le dossier
git clone <url> machine_learning
cd machine_learning

# 2) Creer un environnement virtuel (recommande)
python -m venv .venv
# Activation :
#   - Windows :  .venv\Scripts\activate
#   - Linux/Mac : source .venv/bin/activate

# 3) Installer les dependances
pip install -r requirements.txt
```

## Telechargement des donnees

Les fichiers CSV ne sont **pas** versionnes (volumineux). Telecharge-les via :

```bash
python download_data.py
```

Cela cree `datasets/full_2024.csv` et `datasets/full_2025.csv` (~1.5 Go au
total apres decompression).

## Utilisation

### Entrainer / sauvegarder les modeles

```bash
python save_model.py
```

Genere les fichiers `.joblib` (modeles + encodages + metriques) consommes par
l'application Streamlit.

### Lancer l'application Streamlit

```bash
streamlit run app.py
```

L'app s'ouvre sur http://localhost:8501. Elle contient 7 pages :

| Page | Contenu |
|---|---|
| 🏠 Accueil & Contexte | Problematique, source des donnees, volumetrie |
| 📊 Donnees & Nettoyage | Pipeline de pre-traitement, feature engineering |
| 🔬 Analyses Exploratoires | 8 visualisations EDA pretes pour la soutenance |
| 📈 Regression — Prix | Resultats des modeles, scatter predictions vs realite |
| 🏷️ Classification — Type | Accuracy, matrices de confusion, importance des features |
| 🔮 Demo Regression | Estimer le prix d'un bien (formulaire interactif) |
| 🔍 Demo Classification | Identifier Maison vs Appartement |

### Explorer le notebook

```bash
jupyter notebook projet.ipynb
```

## Structure du projet

```
.
├── readme.md              # ce fichier
├── requirements.txt       # dependances Python
├── download_data.py       # telecharge les CSV DVF
├── save_model.py          # entraine et sauvegarde les modeles
├── app.py                 # application Streamlit (support de soutenance)
├── projet.ipynb           # notebook complet (EDA + modelisation)
├── soutenance.md          # notes pour la soutenance orale
├── datasets/              # donnees DVF (non versionnees)
│   ├── full_2024.csv
│   └── full_2025.csv
└── *.joblib               # artefacts ML generes par save_model.py
```

## Resultats de reference (test = 20% des donnees)

| Tache | Modele | Score |
|---|---|---|
| Regression | Regression Lineaire (log) | R²=0.27 · MAE≈103k EUR · MAPE≈54% |
| Regression | HistGradientBoosting | R²=0.58 · MAE≈77k EUR · MAPE≈47% |
| Classification | Regression Logistique | Accuracy≈95.2% |
| Classification | Random Forest | Accuracy≈95.9% |

## Limites & pistes d'amelioration

- DVF ne contient ni l'**etage**, ni l'**annee de construction**, ni la
  presence d'**ascenseur / balcon** — toutes tres predictives du prix.
- Target encoding par departement trop grossier (Paris 7e ≠ Paris 20e).
- Pistes : target encoding par commune avec lissage bayesien, donnees INSEE
  (revenus par commune), proximite transports, split temporel
  (train 2024 / test 2025), modeles plus puissants (XGBoost, LightGBM).
