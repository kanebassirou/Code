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
| T1b | `telecharger_jpl_sideshow.py` | Complète les fichiers JPL manquants via le miroir JPL Sideshow (sans authentification) |
| T2 | `parseur_ionex.py` | Parseur IONEX générique (en-tête, grille lat/lon, blocs TEC) — utilisé par les autres scripts |
| T3 | `construire_base_tec.py` | Lit tous les fichiers IONEX de l'année et constitue une base compressée `.npz` (CODE + JPL) |
| T4 | `calculer_statistiques.py` | Calcule MBE, RMSE, corrélation de Pearson jour par jour + cartes/figures annuelles |
| T5 | `analyse_complementaire.py` | Analyses complémentaires : MBE en %, découpage par bandes de latitude, étude de l'événement géomagnétique du 27-28 février 2023 (multi-centres : CODE, JPL, ESA, CAS, UPC) |
| T5-bis | `analyse_longitude.py` | Comparaison par bandes de longitude (demande M. Fleury) : zone Océan Atlantique (15°W-60°W, peu de stations GNSS) vs zone Amérique (75°W-120°W, forte densité de stations GNSS) |
| T6 | `cartographies_saisonnieres.py` | Cartographies saisonnières CODE vs JPL : cartes TEC et carte de différence pour les 4 dates clés de 2023 (équinoxes, solstices) + profils latitudinaux moyens par saison |

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
    ├── stats_zones_longitude.npz
    └── figures/             # PNG générés par T4, T5, T5-bis et T6
```

## Utilisation

```bash
# 1. Téléchargement des données brutes (nécessite les identifiants Earthdata)
python telecharger_ionex.py
python telecharger_jpl_sideshow.py     # complète les jours JPL manquants

# 2. Constitution de la base TEC 2023
python construire_base_tec.py

# 3. Statistiques et figures principales (MBE, RMSE, corrélation)
python calculer_statistiques.py

# 4. Analyses complémentaires (MBE %, bandes de latitude, événement de février)
python analyse_complementaire.py

# 5. Comparaison par bandes de longitude (Atlantique vs Amérique)
python analyse_longitude.py

# 6. Cartographies saisonnières (équinoxes, solstices)
python cartographies_saisonnieres.py
```

## Méthode

- Les cartes CODE (25/jour) sont sous-échantillonnées sur les époques paires pour s'aligner sur les 13 cartes JPL (`code[0::2]`).
- Le MBE (*Mean Bias Error*) et le RMSE sont calculés jour par jour sur les cartes alignées CODE − JPL.
- Le MBE relatif (%) est calculé une seule fois par jour (`mbe_pct`) et cette même série sert de source unique pour toutes les valeurs affichées (figures, tableau mensuel, résumé annuel) afin d'éviter toute incohérence entre indicateurs.
- Les zones de longitude (T5-bis) reprennent le même calcul MBE/RMSE/Pearson que les bandes de latitude (T5), appliqué cette fois à deux tranches de longitude contrastées en densité de stations GNSS, afin de vérifier si cette densité explique une partie des écarts CODE-JPL.
- Les cartographies saisonnières (T6) moyennent les cartes CODE et JPL alignées sur les 4 dates clés de l'année (jours juliens 080, 172, 264, 355 — équinoxes et solstices 2023) et en tirent la carte de différence CODE-JPL ainsi que des profils latitudinaux moyens, afin d'observer l'évolution saisonnière des écarts.

## État d'avancement

Étapes T1 à T6 réalisées. Prochaine étape : rédaction du mémoire.
