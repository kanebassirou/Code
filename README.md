# Comparaison des cartes ionosphériques TEC — CODE vs JPL (2023)

Code source du mémoire de Master 2 Informatique — Génie Logiciel (Bassirou Kane).

L'objectif est de comparer, sur l'année 2023, les cartes globales du contenu électronique total (TEC — *Total Electron Content*) produites par deux centres d'analyse GNSS :

- **CODE** (Center for Orbit Determination in Europe) — cartes horaires (25 époques/jour)
- **JPL** (Jet Propulsion Laboratory) — cartes bihoraires (13 époques/jour)

au format **IONEX**, afin d'en évaluer les écarts (biais, RMSE, corrélation) dans le temps, l'espace et par bande de latitude.

## Pipeline

Les scripts s'exécutent dans cet ordre :

| Étape | Script | Rôle |
|---|---|---|
| T1 | `telecharger_ionex.py` | Télécharge les fichiers IONEX CODE + JPL 2023 depuis CDDIS (NASA), les décompresse (`.gz` / `.Z`) |
| T1b | `python telecharger_jpl_sideshow.py` | Complète les fichiers JPL manquants via le miroir JPL Sideshow (sans authentification) |
| T2 | `parseur_ionex.py` | Parseur IONEX générique (en-tête, grille lat/lon, blocs TEC) — utilisé par les autres scripts |
| T3 | `construire_base_tec.py` | Lit tous les fichiers IONEX de l'année et constitue une base compressée `.npz` (CODE + JPL) |
| T4 | `calculer_statistiques.py` | Calcule MBE, RMSE, corrélation de Pearson jour par jour + cartes/figures annuelles |
| T5 | `analyse_complementaire.py` | Analyses complémentaires : MBE en %, découpage par bandes de latitude, étude de l'événement géomagnétique du 27-28 février 2023 (multi-centres : CODE, JPL, ESA, CAS, UPC) |

Utilitaire annexe : `Diagnostic ionex.py` (inspection brute du contenu d'un fichier IONEX pour déboguer le format).

## Prérequis

```bash
pip install numpy requests tqdm unlzw3 matplotlib scipy
```

Un compte [NASA Earthdata](https://urs.earthdata.nasa.gov/) est nécessaire pour `telecharger_ionex.py` (accès CDDIS). Les identifiants sont actuellement codés en dur dans `telecharger_ionex.py` et `test_nasa.py` — **à retirer du code et à passer en variables d'environnement avant tout partage/versionnement du dépôt.**

## Structure des données

```
data/
├── ionex/2023/{JJJ}/       # Fichiers IONEX bruts décompressés, par jour julien
├── base_tec/               # Base compressée : code_tec_2023.npz, jpl_tec_2023.npz, jours_*.npy
└── stats/
    ├── stats_journalieres.npz
    ├── stats_complementaires.npz
    └── figures/             # PNG générés par T4 et T5
```

## Utilisation

```bash
# 1. Téléchargement des données brutes (nécessite les identifiants Earthdata)
python telecharger_ionex.py
python "python telecharger_jpl_sideshow.py"     # complète les jours JPL manquants

# 2. Constitution de la base TEC 2023
python construire_base_tec.py

# 3. Statistiques et figures principales (MBE, RMSE, corrélation)
python calculer_statistiques.py

# 4. Analyses complémentaires (MBE %, bandes de latitude, événement de février)
python analyse_complementaire.py
```

## Méthode

- Les cartes CODE (25/jour) sont sous-échantillonnées sur les époques paires pour s'aligner sur les 13 cartes JPL (`code[0::2]`).
- Le MBE (*Mean Bias Error*) et le RMSE sont calculés jour par jour sur les cartes alignées CODE − JPL.
- Le MBE relatif (%) est calculé une seule fois par jour (`mbe_pct`) et cette même série sert de source unique pour toutes les valeurs affichées (figures, tableau mensuel, résumé annuel) afin d'éviter toute incohérence entre indicateurs.

## État d'avancement

Étapes T1 à T5 réalisées. Prochaine étape : analyse de la variabilité / rédaction du mémoire.
