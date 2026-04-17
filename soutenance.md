# Guide de preparation — Soutenance ML (groupe de 3)

Projet : **Prediction du prix des appartements a partir du dataset DVF**.

Ce guide vous aide a :
1. Repartir le travail entre 3 personnes
2. Suivre un plan de presentation clair
3. Anticiper les questions du jury

---

## 1. Repartition des roles (30-40 minutes de presentation type)

| Partie | Duree | Qui | Contenu |
|---|---|---|---|
| **A. Contexte & Donnees** | 5-8 min | Personne 1 | Enjeu metier, presentation DVF, exploration, filtrage, nettoyage |
| **B. Feature engineering & EDA** | 8-10 min | Personne 2 | Nouvelles variables (target encoding, log-transform), graphiques EDA, correction anti-fuite |
| **C. Modelisation & Resultats** | 10-12 min | Personne 3 | Choix des modeles, metriques, comparaison, optimisation, interpretation |
| **D. Demo + Conclusion + Q/R** | 5-10 min | A 3 | Tour de notebook live, limites, pistes, reponses aux questions |

**Conseil** : chaque personne connait sa partie en profondeur, mais TOUS maitrisent les grandes lignes de tout le projet (pour les questions croisees).

---

## 2. Script de presentation (ce que chaque personne doit dire)

### Partie A — Contexte & Donnees (Personne 1)

**Problematique** :
> "Peut-on predire le prix d'un appartement en France a partir de ses caracteristiques et de sa localisation ? Ce type de modele interesse les agents immobiliers, les plateformes type MeilleursAgents, les banques pour evaluer un bien en garantie."

**Source des donnees** :
- **DVF** (Demandes de Valeurs Foncieres) : dataset **public** publie par la DGFiP
- Toutes les transactions immobilieres en France, issues des actes notaries
- 5 fichiers CSV (un par annee, 2021 a 2025 ; 2020 a ete retire de data.gouv.fr fin 2025), disponibles sur `files.data.gouv.fr/geo-dvf/`
- 39 colonnes au total — nous en avons garde 17 pour limiter la memoire (20 Go de RAM evites !)

**Volumetrie** :
- **~15 millions** de lignes brutes (5 annees)
- **~1,5 a 2 millions** apres filtrage (ventes d'appartements uniquement)

**Pipeline de nettoyage** (insister sur les choix) :
1. Filtrer `nature_mutation == "Vente"` et `type_local == "Appartement"` → 2,7 M lignes
2. Valeur fonciere entre 10 000 EUR et 10 000 000 EUR (elimine les ventes a 1 EUR entre familles et les ventes de portefeuille)
3. Surface reelle bati entre 0 et 500 m2 (un appartement de 5000 m2 est une erreur de saisie)
4. **Deduplication par `id_mutation`** : une meme vente peut apparaitre plusieurs fois si elle comporte plusieurs lots → on garde le lot a plus grande surface
5. Suppression des lignes sans GPS
6. Imputation : nombre de pieces manquant → remplace par la mediane

**Resultat** : **2,04 millions** de lignes propres pretes pour la suite.

---

### Partie B — Feature engineering & EDA (Personne 2)

**Nouvelles variables creees** :
- `annee`, `mois`, `trimestre` : extrait de la date → permet de capturer l'evolution du marche
- `surface_carrez` : surface "officielle" du lot 1 (la surface reelle bati peut inclure des zones non habitables ; la surface Carrez est reglementee)
- `surface_par_piece` : surface / nombre de pieces → indique le "confort" (un 60 m2 en 2 pieces est plus spacieux qu'un 60 m2 en 4 pieces)
- `prix_m2` : **attention, sert uniquement a l'EDA**, pas au modele (il contient la cible, ce serait de la triche)
- `dept_prix_moyen` (target encoding) : moyenne de `valeur_fonciere` par departement → transforme une variable categorielle (code dept) en variable numerique

**Point fort a mettre en avant — la fuite de donnees** :
> "Le target encoding peut **faire tricher le modele**. Si on calcule le prix moyen d'un departement en incluant les lignes du test, le modele 'voit' indirectement sa cible pendant l'entrainement. On a corrige ca en recalculant le prix moyen **uniquement sur le train** apres le split, puis en l'appliquant aux deux jeux."

C'est un detail technique **apprecie par les jurys** : ca montre la rigueur ML.

**EDA — points saillants a commenter** :
- **Distribution du prix** : tres asymetrique (skewed right), longue queue jusqu'a plusieurs millions. D'ou l'interet de la transformation log.
- **Top departements** : Paris (75), Hauts-de-Seine (92), Alpes-Maritimes (06) dominent. Normal : centres economiques et tourisme cote d'Azur.
- **Prix vs nombre de pieces** : le prix augmente avec le nombre de pieces, mais pas lineairement (plafond autour de 5-6 pieces).
- **Evolution trimestrielle** : hausse reguliere 2020-2022, inflexion 2023-2024 (crise de l'immobilier, hausse des taux).
- **Matrice de correlation** : `dept_prix_moyen` et `surface_reelle_bati` sont les plus correlees a `valeur_fonciere`.

---

### Partie C — Modelisation & Resultats (Personne 3)

**Choix des modeles (presentation structuree par famille)** :

**Famille 1 — Modeles lineaires (baseline)** :
- **Regression Lineaire** : modele le plus simple, y = a1*x1 + a2*x2 + ... + b
- **Ridge** : regression lineaire avec regularisation L2 (penalise les gros coefficients → evite l'overfit)
- **Lasso** : regularisation L1 (peut mettre des coefficients a zero → selection de variables automatique)

**Famille 2 — Modeles d'arbres (non-lineaires)** :
- **Random Forest** : moyenne de plein d'arbres entraines sur des sous-echantillons → robuste, bon par defaut
- **HistGradientBoosting** : equivalent LightGBM integre a sklearn, entraine les arbres successivement en corrigeant les erreurs du precedent → tres puissant pour les donnees tabulaires

**Transformation log de la cible** :
> "La distribution des prix est tres asymetrique. Un modele lineaire minimise l'erreur quadratique : il est 'tire' par les gros prix et predit mal les petits. En entrainant sur `log(1+prix)`, la distribution devient presque normale. On re-transforme avec `exp` pour retrouver des euros. `TransformedTargetRegressor` de sklearn fait ca automatiquement."

**Metriques choisies — savoir les defendre** :
- **MAE** : erreur moyenne en EUR. **Tres interpretable** : "on se trompe de X euros en moyenne".
- **RMSE** : erreur quadratique moyenne. **Penalise les grosses erreurs** (utile pour eviter de se planter grave sur un bien cher).
- **MAPE** : erreur en pourcentage. **Tres parlant** : "on se trompe de 20% du prix reel" est plus intuitif que "65k EUR" (65k c'est beaucoup pour un 2-pieces a Tours, peu pour un hotel particulier a Paris).
- **R2** : part de la variance expliquee. Nombre entre 0 et 1, **comparable entre datasets**.

**Lecture des resultats** (a adapter apres re-execution) :
- Les **modeles lineaires** explosent leur R2 apres la log-transform (de 0.31 a >0.55 typiquement)
- Le **Random Forest** reste dans le top : il gere naturellement les non-linearites et l'asymetrie
- **HistGradientBoosting** fait jeu egal avec le RF, plus rapide a entrainer
- Apres **optimisation d'hyperparametres**, on gagne 1-3 points de R2 supplementaires

**Validation croisee 5-fold** :
> "On ne se contente pas d'un seul score sur le test : on fait 5 splits differents, on verifie que le R2 reste stable. Si la variance etait enorme, le modele serait instable et inutilisable."

**Importance des features (permutation importance)** :
- Top 1 : `dept_prix_moyen` (la localisation coute cher a Paris, pas a Limoges)
- Top 2 : `surface_reelle_bati` (evidemment)
- Top 3 : `longitude` / `latitude` (affine la localisation a l'interieur d'un dept)
- Features temporelles : importance faible car le marche est globalement stable

---

### Partie D — Demo + Conclusion + Q/R (a 3)

**Demo** (tour rapide du notebook) :
1. Ouvrir le notebook
2. Scroller jusqu'a la distribution avant/apres log → insister sur la difference visuelle
3. Scroller sur le tableau comparatif des modeles → pointer le gain du log-transform
4. Montrer l'importance des features → confirme l'intuition metier

**Limites a avouer avant que le jury les trouve** :
- Pas d'info sur l'etage, l'annee de construction, l'ascenseur, le balcon → ce sont des variables connues pour etre tres predictives
- Marche qui evolue : un modele entraine en 2024 peut mal predire en 2027 (inflation, taux)
- Quelques ventes atypiques (viager, vente familiale) malgre le filtrage
- Target encoding par departement : trop grossier pour Paris (le 20e et le 7e ont des prix tres differents). Une encoding par commune serait mieux.

**Pistes d'amelioration (ne pas en dire TROP, 3-4 suffit)** :
- Donnees externes : INSEE (revenus par commune), proximite transports, ecoles
- Target encoding plus fin : par commune + lissage bayesien
- Split temporel : train sur 2020-2023, test sur 2024-2025 (simulation realiste)
- Optuna / BayesSearchCV pour une recherche d'hyperparametres plus efficace

---

## 3. Questions probables du jury (+ reponses preparees)

### Questions "bases ML"

**Q : C'est quoi le R2 ?**
> R : Le coefficient de determination. Il vaut 1 si le modele predit parfaitement, 0 s'il est equivalent a "toujours predire la moyenne", et peut etre negatif si le modele est pire que la moyenne. En pratique, R2 = 0.6 veut dire qu'on explique 60% de la variance des prix.

**Q : Pourquoi un train/test split et pas juste entrainer sur tout ?**
> R : Pour mesurer la **generalisation**. Un modele peut apprendre par coeur le dataset (overfitting) et avoir un score parfait en entrainement mais etre nul sur de nouvelles donnees. On garde 20% des donnees jamais vues pour mesurer honnetement la performance.

**Q : C'est quoi la difference entre Ridge et Lasso ?**
> R : Les deux ajoutent une penalite aux gros coefficients. Ridge penalise la somme des carres (L2), Lasso la somme des absolus (L1). Lasso peut mettre des coefficients exactement a zero (selection de variables automatique), Ridge les reduit sans les annuler.

**Q : Pourquoi Random Forest ou Gradient Boosting plutot qu'une regression lineaire ?**
> R : Le prix immobilier a des relations non-lineaires entre features (ex : "surface 100m2 ET Paris 16e" coute beaucoup plus qu'attendu par simple somme). Les arbres capturent ces interactions naturellement, pas les lineaires.

### Questions "choix de conception"

**Q : Pourquoi avoir filtre entre 10k et 10M EUR ?**
> R : En dessous de 10k, ce sont souvent des ventes entre familles a prix symbolique. Au-dessus de 10M, ce sont des hotels particuliers atypiques ou des ventes groupees (un immeuble entier). On cible le marche "residentiel classique".

**Q : Pourquoi le target encoding et pas du one-hot ?**
> R : Il y a **96 departements**. Un one-hot creerait 96 colonnes de zeros et de uns, tres sparse. Le target encoding resume la variable categorielle en **une seule colonne numerique** qui capte directement l'info utile : "les appartements dans ce dept coutent en moyenne X EUR".

**Q : La fuite de donnees, qu'est-ce que c'est concretement ici ?**
> R : Si on calcule le prix moyen d'un departement en incluant les lignes qu'on va utiliser au test, le modele "voit" indirectement sa cible. Exemple : si un dept n'a qu'une vente dans le dataset et qu'elle tombe en test, le `dept_prix_moyen` est exactement cette valeur → le modele a la reponse dans sa feature. On corrige en calculant la moyenne **uniquement sur le train**.

**Q : Pourquoi la transformation log ?**
> R : La distribution des prix est tres asymetrique (beaucoup de biens a 100-300k, quelques-uns a 5M+). La regression lineaire minimise l'erreur au carre : elle est dominee par les gros prix et predit mal les petits. En travaillant sur log(prix), la distribution est presque normale, tous les prix pesent pareil. On re-transforme avec exp pour retrouver des euros.

**Q : Pourquoi HistGradientBoosting et pas XGBoost / LightGBM ?**
> R : `HistGradientBoostingRegressor` est l'implementation native de sklearn, elle donne des resultats tres proches de LightGBM (dont elle est inspiree). Ca evite une dependance externe et reste rapide sur plusieurs millions de lignes.

### Questions "resultats et limites"

**Q : Un R2 de 0.6, c'est bon ?**
> R : Pour de l'immobilier sur donnees limitees, oui — on manque d'infos cles (etage, standing, etat). Les leaders du marche (MeilleursAgents, SeLoger) tournent autour de 0.85-0.90 en utilisant des dizaines de features supplementaires et des modeles plus complexes.

**Q : Pourquoi une MAPE de ~20%, c'est beaucoup ?**
> R : Oui, mais c'est normal compte tenu des limites du dataset : pas d'etage, pas de standing, pas d'annee de construction. Sur un bien a 300 000 EUR, 20% d'erreur = 60 000 EUR. Ca veut dire que le modele donne une **fourchette utile** mais ne remplace pas une expertise detaillee.

**Q : Le modele serait-il utilisable en production ?**
> R : Comme **outil d'aide a la decision**, oui : estimer rapidement un ordre de grandeur. Comme **outil de prise de decision automatique** (pret bancaire par exemple), non — l'incertitude est trop forte. Il faudrait ajouter des intervalles de confiance et enrichir les features.

**Q : Quelles seraient les ameliorations prioritaires ?**
> R : D'abord **enrichir les donnees** (INSEE revenus par commune, proximite transports). Ensuite **target encoding par commune** avec lissage. Enfin **split temporel** pour s'assurer que le modele generalise a des annees futures.

### Questions "techniques"

**Q : Vous avez combien de lignes au final ? Combien de features ?**
> R : Environ 1,5 a 2 millions de lignes apres nettoyage (le chiffre exact depend du re-run). 11 features numeriques en entree du modele. Les 5 CSV bruts (2021-2025) faisaient environ 15 millions de lignes avant filtrage.

**Q : Combien de temps pour entrainer les modeles ?**
> R : Quelques secondes pour les lineaires, 2-3 minutes pour le Random Forest, ~1 minute pour HistGradientBoosting. GridSearchCV sur 12 combinaisons + 3-fold CV : environ 5-10 minutes sur un echantillon de 100k lignes.

**Q : Pourquoi un echantillon pour la cross-validation et le GridSearch ?**
> R : Pour des raisons de temps. Random Forest sur 5-fold CV avec 1,6M lignes prendrait des dizaines de minutes. On verifie sur un echantillon representatif (200k pour la CV, 100k pour la grid search) que le modele est stable.

**Q : Comment vous avez gere les valeurs manquantes ?**
> R : Colonne par colonne : `valeur_fonciere` et `surface_reelle_bati` → suppression des lignes (la cible et la principale feature, on veut des donnees propres). `lot1_surface_carrez` → rempli par 0 (absence signifie pas de lot Carrez). `nombre_pieces_principales` → rempli par la mediane.

---

## 4. Checklist avant la presentation

**J-7 (une semaine avant)** :
- [ ] Re-executer TOUT le notebook en entier pour avoir des resultats a jour
- [ ] Verifier que les resultats sont coherents (R2 positif, MAPE < 50%)
- [ ] Capturer les graphiques importants en PNG (pour les slides de secours)
- [ ] Se repartir les parties A, B, C entre les 3 personnes

**J-3** :
- [ ] Rediger les slides (si demandees) : 1-2 slides par personne pour les points saillants
- [ ] Repeter la presentation complete ensemble au moins 2 fois
- [ ] Chronometrer chaque partie

**J-1** :
- [ ] Tester le projecteur / HDMI / rendu du notebook
- [ ] Preparer un fichier de sauvegarde (PDF du notebook) au cas ou Jupyter plante
- [ ] Relire les questions probables une derniere fois

**Jour J** :
- [ ] Arriver 15 minutes en avance
- [ ] Ouvrir le notebook AVANT le debut (pour eviter le temps de chargement)
- [ ] Avoir le README et le notebook ouverts cote a cote

---

## 5. Slides recommandees (si support visuel demande)

Structure minimale pour un support de 10-12 slides :
1. Titre + noms
2. Probleme : predire le prix des appartements (pourquoi c'est utile)
3. Donnees : DVF, volumetrie, pipeline de nettoyage (un schema simple)
4. EDA : distribution des prix (avant / apres log) — graphique marquant
5. EDA : top departements + evolution temporelle
6. Feature engineering : la liste + la correction anti-fuite
7. Methodologie : train/test split + 5 modeles + metriques
8. Resultats : tableau comparatif des 5 modeles (apres log-transform)
9. Optimisation : resultat de GridSearchCV
10. Importance des features : diagramme barh
11. Conclusion : ce qui marche, limites, pistes
12. Questions

---

## 6. Points forts a mettre en avant (ce qui fait pro)

- **Pipeline complet** : de 20M de lignes brutes a un modele entraine et evalue
- **Rigueur ML** : correction anti-fuite du target encoding (les jurys adorent)
- **Transformation log** bien justifiee par l'EDA (asymetrie visible)
- **Comparaison de 5 modeles** + optimisation d'hyperparametres
- **4 metriques complementaires** (MAE, RMSE, MAPE, R2)
- **Validation croisee 5-fold** pour la robustesse
- **Permutation importance** pour interpreter le modele
- **Conclusion honnete** : limites assumees, pistes d'amelioration concretes

---

## 7. Vocabulaire technique a bien maitriser

| Terme | Definition courte |
|---|---|
| Overfitting | Le modele memorise le train et ne generalise pas |
| Underfitting | Le modele est trop simple, il capte mal la realite |
| Data leakage | Le modele a "vu" son test pendant l'entrainement |
| Cross-validation | Diviser le train en K parts, entrainer K fois, moyenner |
| Hyperparametre | Reglage du modele **fixe par l'humain** (pas appris) |
| Feature engineering | Creer de nouvelles variables a partir des existantes |
| Target encoding | Remplacer une categorie par la moyenne de la cible |
| Baseline | Modele de reference simple pour comparer |
| Ensemble | Combinaison de plusieurs modeles (RF, gradient boosting) |
| Regularisation | Penalite ajoutee a la loss pour reduire l'overfit |

---

**Bonne chance !** Vous avez un projet solide, bien documente et bien structure. Le plus important : **parler avec confiance des choix faits** et **assumer les limites**.
