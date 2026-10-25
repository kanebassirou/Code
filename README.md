# Comparaison des cartes ionosphériques TEC — CODE vs JPL (2023)

Code source du mémoire de Master 2 Informatique (Génie Logiciel) de **Bassirou Kane**, Université Iba Der Thiam de Thiès.

Ce dépôt compare les cartes mondiales de contenu électronique total (TEC) au format IONEX produites par deux centres d'analyse de l'IGS, **CODE** et **JPL**, sur les 365 jours de 2023. Il mesure leurs écarts (biais, RMSE, corrélation) dans le temps, dans l'espace et par région, et vérifie que ces résultats tiennent quand on change la méthode d'alignement temporel.

---

## Sommaire

1. [Résultats produits](#résultats-produits)
2. [Installation](#installation)
3. [Exécution de la chaîne](#exécution-de-la-chaîne)
4. [Description des modules](#description-des-modules)
5. [Organisation des données](#organisation-des-données)
6. [Méthode](#méthode)
7. [Tests](#tests)
8. [Notes techniques](#notes-techniques)

---

## Résultats produits

| Analyse | Indicateurs | Sorties principales |
|---|---|---|
| Statistiques journalières (T4) | MBE, RMSE, corrélation de Pearson | séries annuelles, boxplot mensuel, carte des différences, heatmap latitude × mois |
| Analyse complémentaire (T5) | MBE relatif (%), bandes de latitude | biais par région, événement géomagnétique du 27-28 février 2023 (5 centres) |
| Bandes de longitude (T5-bis) | MBE, RMSE par zone | Atlantique (peu de stations) vs Amérique (forte densité) |
| Cartographies saisonnières (T6) | différences CODE − JPL | équinoxes et solstices, profils latitudinaux |
| Sensibilité de l'alignement (T9) | MBE, RMSE par méthode | JPL interpolé à 1 h (linéaire, quadratique, spline cubique), validation intrinsèque |
| Distribution des écarts (T10) | histogrammes, fonctions de répartition, centiles | queues de distribution de ΔTEC |
| Diagnostics et annexes | journées extrêmes, en-têtes IONEX | annexes A et B du mémoire |

Toutes les figures sont écrites dans [data/stats/figures/](data/stats/figures/), et les résultats numériques dans des fichiers `.npz` sous [data/stats/](data/stats/).

---

## Installation

**Prérequis :** Python 3.12 (version utilisée pour le mémoire).

```bash
# Créer et activer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Installer les dépendances
pip install numpy scipy matplotlib requests tqdm unlzw3 pytest
```

<details>
<summary>Versions testées</summary>

| Paquet | Version |
|---|---|
| numpy | 2.5.1 |
| scipy | 1.18.0 |
| matplotlib | 3.11.1 |
| requests | 2.34.2 |
| tqdm | 4.69.0 |
| unlzw3 | 0.2.3 |
| pytest | 9.1.1 |

</details>

### Accès aux données NASA CDDIS

Les fichiers IONEX de 2023 et la base TEC sont **déjà présents dans le dépôt**. Vous n'avez besoin de les télécharger que pour reconstruire les données à partir de zéro.

Dans ce cas, il faut un compte [NASA Earthdata](https://urs.earthdata.nasa.gov/). Les identifiants se règlent dans les variables `NASA_LOGIN` et `NASA_PASSWORD` en tête de [telecharger_ionex.py](telecharger_ionex.py).

> **Ne versionnez jamais vos identifiants.** Remettez ces variables à vide avant chaque commit.

---

## Exécution de la chaîne

Lancez toutes les commandes **depuis la racine du projet** : les chemins vers `data/` sont relatifs.

```bash
# 1. Téléchargement des IONEX (facultatif, les données sont fournies)
python telecharger_ionex.py            # CODE + JPL depuis CDDIS (Earthdata requis)
python telecharger_jpl_sideshow.py     # jours JPL manquants (sans authentification)

# 2. Construction de la base TEC
python construire_base_tec.py

# 3. Statistiques de référence (à lancer avant l'étape 4)
python calculer_statistiques.py

# 4. Analyses (indépendantes les unes des autres)
python analyse_complementaire.py       # nécessite stats_journalieres.npz (étape 3)
python analyse_longitude.py
python cartographies_saisonnieres.py
python Interpolation_jpl.py
python analyse_distribution_ecarts.py

# 5. Diagnostics et annexes du mémoire
python diagnostic_journees_extremes.py
python inventaire_entetes_ionex.py
```

### Dépendances entre étapes

```
telecharger_*.py ──► data/ionex/2023/
                          │
                          ▼
              construire_base_tec.py ──► data/base_tec/*.npz
                          │
          ┌───────────────┼──────────────────────────────┐
          ▼               ▼                              ▼
 calculer_statistiques   analyse_longitude          Interpolation_jpl
          │              cartographies_saisonnieres analyse_distribution_ecarts
          ▼              diagnostic_journees_extremes
 analyse_complementaire  inventaire_entetes_ionex
```

---

## Description des modules

### Acquisition et préparation

| Fichier | Tâche | Rôle |
|---|---|---|
| [telecharger_ionex.py](telecharger_ionex.py) | T1 | Télécharge et décompresse les IONEX CODE et JPL depuis CDDIS |
| [telecharger_jpl_sideshow.py](telecharger_jpl_sideshow.py) | T1 | Récupère les jours JPL manquants depuis le miroir Sideshow du JPL |
| [parseur_ionex.py](parseur_ionex.py) | T2 | Parseur IONEX réutilisable, qui gère les coordonnées collées (`87.5-180.0`) |
| [construire_base_tec.py](construire_base_tec.py) | T3 | Assemble les 365 jours en tableaux `.npz` et gère les deux conventions de nommage JPL |

### Analyses

| Fichier | Tâche | Rôle |
|---|---|---|
| [calculer_statistiques.py](calculer_statistiques.py) | T4 | MBE, RMSE, corrélation de Pearson journaliers, et figures principales |
| [analyse_complementaire.py](analyse_complementaire.py) | T5 | MBE en %, bandes de latitude, comparaison multi-centres (CODE, JPL, ESA, CAS, UPC) |
| [analyse_longitude.py](analyse_longitude.py) | T5-bis | Comparaison Atlantique (15°W–60°W) / Amérique (75°W–120°W) |
| [cartographies_saisonnieres.py](cartographies_saisonnieres.py) | T6 | Cartes aux jours 080, 172, 264 et 355 (équinoxes et solstices) |
| [Interpolation_jpl.py](Interpolation_jpl.py) | T9 | Alignement inverse : JPL interpolé à 1 h au lieu de sous-échantillonner CODE |
| [analyse_distribution_ecarts.py](analyse_distribution_ecarts.py) | T10 | Histogrammes et fonctions de répartition de ΔTEC, calculés sans tout charger en mémoire |

### Diagnostics et utilitaires

| Fichier | Rôle |
|---|---|
| [diagnostic_journees_extremes.py](diagnostic_journees_extremes.py) | Date et caractérise les journées extrêmes, teste la cohérence temporelle, situe les orages géomagnétiques |
| [inventaire_entetes_ionex.py](inventaire_entetes_ionex.py) | Stations déclarées dans les en-têtes, valeurs extrêmes et nulles (Annexe A) |
| [outils_figures.py](outils_figures.py) | Sauvegarde des figures résistante aux verrous OneDrive |
| [Diagnostic ionex.py](Diagnostic%20ionex.py) | Affiche le contenu brut d'un fichier IONEX pour le débogage |
| [test_chaine.py](test_chaine.py) | Tests automatisés de la chaîne de traitement (Annexe B) |

---

## Organisation des données

```
data/
├── ionex/2023/{JJJ}/              # IONEX décompressés, un dossier par jour julien (001–365)
│   ├── COD0OPSFIN_2023JJJ0000_01D_01H_GIM.INX   # CODE : 25 cartes/jour (1 h)
│   └── jplgJJJ0.23i                             # JPL  : 13 cartes/jour (2 h)
│                                                # 058-059 : aussi ESA, CAS, UPC, IGS
├── base_tec/
│   ├── code_tec_2023.npz          # TEC CODE  (jours × époques × 71 lat × 73 lon)
│   ├── jpl_tec_2023.npz           # TEC JPL
│   └── jours_{code,jpl,communs}.npy
└── stats/
    ├── stats_journalieres.npz         # T4
    ├── stats_complementaires.npz      # T5
    ├── stats_zones_longitude.npz      # T5-bis
    ├── stats_interpolation_jpl.npz    # T9
    ├── stats_distribution_ecarts.npz  # T10
    └── figures/                       # toutes les figures PNG
```

Grille IONEX : latitudes de 87,5°N à 87,5°S (pas de 2,5°), longitudes de 180°W à 180°E (pas de 5°). Valeurs en TECU.

---

## Méthode

- **Alignement temporel (référence) :** CODE fournit 25 cartes par jour (toutes les heures) et JPL 13 (toutes les 2 h). On ne garde que les époques paires de CODE pour comparer les deux produits aux mêmes instants.
- **Alignement inverse (T9) :** pour vérifier la robustesse du choix précédent, JPL est interpolé à 1 h (linéaire, quadratique, spline cubique) et comparé aux 25 cartes CODE. On valide l'interpolation en retirant les époques impaires de CODE, en les reconstruisant puis en mesurant l'erreur.
- **Indicateurs :** MBE (biais moyen), RMSE, coefficient de corrélation de Pearson et MBE relatif en %. Le MBE % est calculé une fois par jour, et c'est cette série qui sert à toutes les moyennes (figures, tableaux, résumé annuel).
- **Agrégation :** calcul jour par jour sur les cartes alignées, puis moyennes par mois, par saison, par bande de latitude et par zone de longitude.

---

## Tests

```bash
python -m pytest test_chaine.py -v
```

Les 21 tests reprennent le protocole de l'Annexe B :

| Préfixe | Type | Contenu |
|---|---|---|
| `TU` | Tests unitaires | décodage, indicateurs, interpolation, histogrammes, alignement (sur données synthétiques) |
| `TI` | Tests d'intégrité | lecture des IONEX réels, complétude et vraisemblance physique de la base TEC |
| `TR` | Test de reproduction | les valeurs publiées dans le mémoire sont retrouvées |

Si `data/` est absent, les tests `TI` et `TR` sont ignorés. La suite complète prend environ 3 minutes.

---

## Notes techniques

- **OneDrive :** le projet est synchronisé par OneDrive, qui verrouille parfois les fichiers pendant la synchronisation. Les figures passent donc par `sauver_figure()` ([outils_figures.py](outils_figures.py)), qui écrit d'abord dans un fichier temporaire puis remplace l'ancien.
- **Nommage JPL :** deux conventions existent (`jplgJJJ0.23i` et `JPL0OPSFIN_..._GIM.INX`). Les deux sont reconnues.
- **Journaux :** `telechargement.log` et `telechargement_jpl_sideshow.log` tracent les téléchargements.

---

## Auteur

**Bassirou Kane** — Master 2 Informatique, Génie Logiciel
Contact : bassirou.kane@univ-thies.sn
