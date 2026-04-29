# Notes de soutenance — Projet ML DVF (2024-2025)


---

## 👤 EMY — Introduction + Données / Nettoyage

**Pages Streamlit : `🏠 Accueil & Contexte` puis `📊 Données & Nettoyage`**

### 1. Ouverture (≈ 30 s)

> « Bonjour, nous présentons un projet de Machine Learning sur les transactions immobilières en France. À partir du jeu de données public DVF, nous avons construit deux modèles : l'un pour **prédire le prix de vente** d'un bien (régression), l'autre pour **identifier son type — Maison ou Appartement** (classification). On vous montre tout via une application Streamlit pour rendre les résultats interactifs. »

### 2. Problématique & cas d'usage (≈ 45 s)

- Question centrale : *peut-on, à partir des transactions immobilières publiques, estimer un prix et identifier un type de bien ?*
- À quoi ça sert dans la vraie vie :
  - **Agents immobiliers** : estimation rapide d'un mandat
  - **Plateformes** type MeilleursAgents / SeLoger → estimation automatique
  - **Banques** : évaluation d'un bien donné en garantie de prêt
- Insister : c'est un problème **réel**, pas un toy dataset.

### 3. La donnée — DVF (≈ 1 min)

- **DVF = Demandes de Valeurs Foncières**, publiée par la **DGFiP**.
- Toutes les **mutations notariées** en France, en open data sur data.gouv.fr (geo-DVF).
- On a pris **2024 + 2025** :
  - Volume restreint (entraînement rapide)
  - Données représentatives du marché actuel
- Volumétrie : **~7,2 M de lignes brutes**, 39 colonnes → **~1,1 M après nettoyage**.
- Variables clés : `valeur_fonciere` (cible régression), `type_local` (cible classification), `surface_reelle_bati`, `nombre_pieces_principales`, `surface_terrain`, `code_departement`, `longitude`/`latitude`.

### 4. Pipeline de nettoyage (≈ 1 min 30) — **le cœur de ta partie**

Dérouler les 6 étapes en justifiant *pourquoi* :

1. **Filtrage métier** : on ne garde que les `Vente` de `Maison` ou `Appartement` (on exclut VEFA, échanges, dépendances, locaux commerciaux).
2. **Bornage du prix** : `valeur_fonciere` ∈ [10 000 € ; 10 000 000 €].
   → *En dessous : ventes familiales symboliques (1 €). Au-dessus : ventes d'immeubles entiers, hors scope.*
3. **Bornage des surfaces** : `surface_reelle_bati` ∈ ]0 ; 500 m²].
   → *Coquilles fréquentes dans les déclarations.*
4. **Déduplication par `id_mutation`** : une mutation peut apparaître plusieurs fois (multi-lots). On garde **le lot avec la plus grande surface**.
5. **Suppression des lignes sans GPS** (longitude/latitude).
6. **Imputation** : `nombre_pieces_principales` → médiane ; `surface_terrain` → 0 (typique des appartements).

### 5. Feature Engineering (≈ 45 s)

- **Variables temporelles** : `annee`, `mois`, `trimestre` extraits de `date_mutation`.
- `surface_carrez` (lot principal, sinon 0)
- `surface_par_piece` = surface / pièces → proxy du confort
- `is_maison` (binaire, pour la régression)
- **`dept_prix_moyen`** : prix moyen par département → c'est du **target encoding**.

### 6. Le point fort à mettre en valeur : l'anti-fuite (≈ 30 s) 🚩

> « Un point méthodologique important : le `dept_prix_moyen` est **calculé uniquement sur le jeu d'entraînement**, puis appliqué au test. Si on calculait cette moyenne sur tout le dataset, on ferait une fuite de données : le modèle "verrait" indirectement le prix des biens du test pendant l'entraînement, et nos scores seraient artificiellement gonflés. »

→ C'est le genre de détail que les jurys adorent. **Insister dessus.**

### 7. Transition vers Matthéo

> « Maintenant que la donnée est propre, Matthéo va vous présenter ce qu'on en a appris avant de modéliser, puis la partie régression. »

### ⚠️ Questions probables du jury pour toi

- **« Pourquoi pas plus d'années ? »** → Volume + temps d'entraînement, et le marché bouge (cf. Pinel, hausse des taux). 2024-2025 = représentatif de la dynamique actuelle.
- **« Pourquoi remplacer les pièces manquantes par la médiane et pas la moyenne ? »** → La distribution est asymétrique (queues longues), la médiane est plus robuste.
- **« Pourquoi mettre `surface_terrain` à 0 et pas la médiane ? »** → C'est sémantique : l'appartement n'a *pas* de terrain, ce n'est pas une donnée manquante.
- **« 1,1 M de lignes restantes, c'est beaucoup ou peu ? »** → Largement suffisant pour des modèles tabulaires classiques. Au-delà, on a des rendements décroissants.

---

## 👤 MATTHÉO — Analyses Exploratoires (1 à 4) + Régression

**Pages Streamlit : `🔬 Analyses Exploratoires` (haut de page) puis `📈 Régression — Prix`**

### 1. Reprise (≈ 10 s)

> « Merci Emy. Avant de modéliser, on a fait des analyses exploratoires pour comprendre la structure du jeu de données. »

### 2. EDA 1 — Distribution de la cible (≈ 1 min) 🚩 **Important**

- À gauche : prix brut → **très asymétrique**, longue queue à droite.
- À droite : `log(1 + prix)` → distribution **quasi-normale, symétrique**.
- **Pourquoi c'est essentiel ?**
  > « Une régression linéaire minimise l'erreur au carré. Sans transformation log, le modèle est tiré par les biens chers (qui pèsent énormément dans la loss) et prédit mal les biens standards. Travailler sur le log met tous les ordres de grandeur sur un pied d'égalité. »
- Cette observation justifie directement le `TransformedTargetRegressor` qu'on verra dans la partie modèle.

### 3. EDA 2 — Top 15 départements (≈ 45 s)

- Prix médian au m² par département, top 15.
- Paris (75) et la petite couronne **écrasent** le classement.
- Conclusion : **la localisation est la variable la plus prédictive**, et de loin.
- Justifie pourquoi on a fait du target encoding sur le département.

### 4. EDA 3 — Prix par nombre de pièces (≈ 30 s)

- Le prix médian croît avec le nombre de pièces, **mais pas linéairement**.
- Anomalie : les T6+ sont parfois **moins chers** que les T5.
  → *Effet localisation* : les très grands logements sont sur-représentés en zones rurales.
- Donne envie de modèles non-linéaires (qui captent ces interactions).

### 5. EDA 4 — Répartition Maison / Appartement (≈ 30 s)

- Un peu plus de Maisons que d'Appartements (≈ 55 / 45).
- Classes **équilibrées** → on peut utiliser l'accuracy comme métrique sans biais.
- C'est important : pas besoin d'oversampling/SMOTE pour la classification.

### 6. Transition vers la modélisation (≈ 15 s)

> « Ces analyses orientent les choix de modèle : transformation log de la cible, target encoding par département, et besoin de modèles non-linéaires pour capter les interactions. On passe à la régression. »

---

### 7. Régression — Objectif (≈ 20 s)

- Cible : `valeur_fonciere` (prix de vente en €).
- 13 features : surfaces, pièces, GPS, temporel, target encoding, `is_maison`.
- Split **80/20**, `random_state=42` pour la reproductibilité.

### 8. Modèle 1 — Régression Linéaire (log) (≈ 1 min)

- Baseline interprétable : `log(1 + prix) = a₁·surface + a₂·pieces + ... + b`.
- Encapsulée dans un **`TransformedTargetRegressor`** : applique `log` à l'entraînement et `exp` à la prédiction → on récupère un prix en € directement.
- **Résultats** : R² ≈ 0,27 · MAE ≈ 103 k€ · MAPE ≈ 54 %.
- **Lecture honnête** : c'est faible. Ça veut dire que la relation prix ↔ features n'est **pas linéaire**, même en log. Ce qui justifie le passage à un modèle plus puissant.

### 9. Modèle 2 — HistGradientBoosting (≈ 1 min 30) 🚩 **Le clou de ta partie**

- Boosting par gradient sur **histogrammes** (l'idée de LightGBM, intégrée à scikit-learn).
- Construit des arbres **successivement**, chacun corrigeant les erreurs du précédent.
- Capture les **interactions non-linéaires** : par exemple « grande surface ET Paris = explosion du prix ».
- Hyperparamètres : `max_iter=500`, `max_depth=10`, `learning_rate=0.1`, `min_samples_leaf=20`.
- **Résultats** : R² ≈ 0,58 · MAE ≈ 77 k€ · MAPE ≈ 47 %.
  → on **double** le R² par rapport à la régression linéaire.

### 10. Visualiser les résultats (≈ 30 s)

- Scatter prédictions vs réalité : la diagonale rouge = prédiction parfaite.
- HGB **colle bien** à la diagonale, surtout pour les prix moyens et élevés.
- La régression linéaire sous-estime systématiquement les prix élevés (effet de l'asymétrie résiduelle).

### 11. Limites & pistes (≈ 30 s) — **À assumer franchement**

- DVF **ne contient pas** : étage, année de construction, ascenseur, balcon, état du bien. Toutes très prédictives — c'est notre plafond de verre.
- Target encoding par **département** trop grossier : Paris 7e ≈ Paris 20e dans nos features alors que les prix vont du simple au triple.
- Pistes : target encoding **par commune** avec lissage bayésien, données INSEE (revenus), proximité transports, **split temporel** (train 2024 / test 2025), modèles plus puissants (XGBoost, LightGBM).

### 12. Transition vers Armand

> « Voilà pour la régression. Armand va vous présenter le reste de l'EDA puis la classification — où on va voir qu'une seule feature change tout. »

### ⚠️ Questions probables du jury pour toi

- **« Pourquoi un R² de 0,58 c'est bien ? »** → Sur des données réelles avec des features manquantes (pas d'étage, pas d'année de construction…), c'est tout à fait honorable. Les modèles industriels (Meilleurs Agents) atteignent ~0,75 mais avec des features bien plus riches.
- **« Pourquoi pas XGBoost ? »** → HistGradientBoosting est l'équivalent natif sklearn, plus simple à intégrer. XGBoost est dans les pistes d'amélioration.
- **« MAPE de 47 %, ça veut dire qu'on se trompe de 47 % en moyenne ? »** → Oui, c'est la moyenne des erreurs relatives. C'est élevé sur les biens à bas prix (10 k€ → la moindre erreur explose le %). En valeur absolue, MAE = 77 k€, ce qui est plus parlant.
- **« Pourquoi le `random_state=42` ? »** → Reproductibilité. Le 42 c'est juste la convention (Hitchhiker's Guide). N'importe quel entier figé fait l'affaire.
- **« Pourquoi pas de validation croisée affichée ? »** → On en a fait dans le notebook (5-fold), les résultats sont stables ; on a montré le test set pour gagner en clarté.

---

## 👤 ARMAND — Analyses Exploratoires (5 à 8) + Classification

**Pages Streamlit : `🔬 Analyses Exploratoires` (bas de page) puis `🏷️ Classification — Type de bien`**

### 1. Reprise (≈ 10 s)

> « Merci Matthéo. Je termine les analyses exploratoires avant d'enchaîner sur la classification. »

### 2. EDA 5 — Surface terrain par type (≈ 1 min) 🚩 **À mettre en avant**

- Boxplot de `surface_terrain` selon `type_local`.
- **Observation forte** : la `surface_terrain` est **quasi-nulle** pour les appartements et **non nulle (souvent > 100 m²)** pour les maisons.
- C'est *la* feature ultra-discriminante.
- **Annonce** : « Vous allez voir, c'est cette feature qui explique pourquoi on dépasse 95 % d'accuracy en classification. »

### 3. EDA 6 — Évolution trimestrielle des prix (≈ 30 s)

- Prix médian par trimestre sur 2024-2025.
- Tendance et saisonnalité visibles : ralentissement marqué fin 2024, reprise en 2025.
- Justifie l'inclusion des features `mois` et `trimestre` dans la régression.

### 4. EDA 7 — Surface bâti vs prix (≈ 45 s)

- Scatter sur 30 000 points, coloré par type.
- La relation surface ↔ prix est **globalement linéaire** mais **très dispersée**.
- Cette dispersion est portée par la **localisation** (deux 60 m² dans deux départements différents = écart x 5).
- Confirme : ni la surface seule, ni la localisation seule, ne suffit. Il faut combiner.

### 5. EDA 8 — Matrice de corrélation (≈ 45 s)

- Variables les plus corrélées au prix :
  - `dept_prix_moyen` (target encoding)
  - `surface_reelle_bati`
  - `surface_carrez`
- À noter : `surface_carrez` est très corrélée à `surface_reelle_bati` (logique → potentielle redondance, mais pas gênant pour les modèles à arbres).
- Pas de corrélation extrême (> 0,95) qui imposerait de virer une feature.

### 6. Transition (≈ 10 s)

> « On passe à la classification : Maison ou Appartement ? »

---

### 7. Classification — Objectif (≈ 20 s)

- Variable cible : 1 = Appartement, 0 = Maison.
- 9 features physiques + GPS (pas de target encoding ici, pas besoin).
- Split **stratifié** 80/20 (`stratify=y_clf`) pour conserver la proportion des deux classes.

### 8. Modèle 1 — Régression Logistique (≈ 1 min)

- Modèle linéaire pour la classification : `P(Appartement) = σ(w·x + b)`, avec σ la sigmoïde → sortie ∈ [0, 1].
- **Pipeline** : `StandardScaler` puis `LogisticRegression` — la mise à l'échelle est indispensable car les features ont des unités très différentes (m², €, degrés GPS…).
- **Résultat** : Accuracy ≈ **95,2 %**.

### 9. Modèle 2 — Random Forest (≈ 1 min 30) 🚩

- Ensemble d'arbres de décision construits sur des **sous-échantillons** des données et des features (bagging).
- Vote majoritaire pour la prédiction finale.
- Hyperparamètres : `n_estimators=200`, `max_depth=20`, `min_samples_leaf=5`.
- **Résultat** : Accuracy ≈ **95,9 %**.
- Marginalement meilleur, mais surtout : on récupère les **importances de features**.

### 10. Importance des features (≈ 1 min) 🚩 **Le moment fort**

- Dans le graph d'importance, `surface_terrain` **domine** très largement, suivie de `surface_carrez` et `nombre_lots`.
- **Lecture intuitive** :
  - `surface_terrain` ≈ 0 → appartement
  - `surface_carrez` > 0 → copropriété → appartement
  - `nombre_lots` > 1 → copropriété → appartement
- C'est cohérent avec le boxplot de l'EDA 5 : on **vérifie** ce qu'on avait soupçonné.

### 11. Matrices de confusion (≈ 30 s)

- Présenter les 2 matrices côte à côte.
- Erreurs typiques : maisons sans terrain renseigné, ou appartements en rez-de-chaussée avec petit jardin → cas **ambigus** que le modèle confond.
- Insister : **95 % d'accuracy est plafonnée par cette ambiguïté intrinsèque** dans les déclarations DVF.

### 12. Limites & pistes (≈ 30 s)

- Frontière Maison/Appartement parfois **floue** (duplex avec terrain, maison de ville).
- On pourrait étendre à du **multi-classes** : Dépendance, Local commercial.
- XGBoost / GridSearchCV pour gratter les derniers points (mais marginal).

### 13. Conclusion finale (≈ 30 s) — **Tu fermes la présentation**

> « Pour conclure : sur la régression, on atteint un R² de 0,58 avec HistGradientBoosting — limité par les features absentes du DVF (étage, année de construction…). Sur la classification, on dépasse 95 % d'accuracy, principalement grâce à `surface_terrain`. Le projet montre surtout l'importance du **pipeline avant le modèle** : nettoyage rigoureux, anti-fuite sur le target encoding, et choix de la transformation log. On vous laisse tester l'app en démo, et on est preneurs de vos questions. »

### ⚠️ Questions probables du jury pour toi

- **« 95 %, c'est presque trop beau ? »** → C'est expliqué par la nature de `surface_terrain` qui est quasi parfaite comme séparateur. Avec uniquement cette feature, on est déjà au-dessus de 90 %. Le modèle complet apporte les 5 derniers points.
- **« Pourquoi stratifier le split en classification mais pas en régression ? »** → La stratification s'applique aux variables catégorielles. En régression, on aurait dû binner `valeur_fonciere` avant — pas indispensable vu la taille du dataset (~1 M).
- **« Pourquoi pas un seul Random Forest pour les deux tâches ? »** → On peut faire un `RandomForestRegressor` mais HistGradientBoosting est plus performant sur les distributions asymétriques.
- **« L'app sait-elle gérer un cas vraiment ambigu ? »** → Oui, on affiche la **probabilité** dans la démo. Si les deux modèles ne sont pas d'accord, l'app le signale explicitement.
- **« Et si la `surface_terrain` est absente ? »** → On l'impute à 0, ce qui biaise la prédiction vers Appartement. C'est une limite assumée.

---

## 🔗 Conseils communs pour l'oral

- **Démo Streamlit à la fin** : prévoir un cas "facile" (Paris 75, 50 m², 2 pièces, terrain 0 → Appartement à ~600 k€) et un cas "piège" (duplex avec petit terrain) pour montrer la fiabilité **et** les limites.
- **Vocabulaire à maîtriser** : *target encoding*, *data leakage*, *TransformedTargetRegressor*, *StandardScaler*, *boosting vs bagging*, *MAE / RMSE / MAPE*.
- **Si vous séchez sur une question** : « C'est une bonne question, on n'a pas testé cette piste — voici comment on s'y prendrait : […] ». Mieux vaut assumer que bluffer.
- **Répartition du temps de Q/R** : Emy répond aux questions méthodo et données ; Matthéo aux questions de régression et de modèles linéaires ; Armand aux questions de classification, importances et conclusion.

Bonne soutenance 🚀
