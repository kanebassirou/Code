# Comparaison des cartes ionosphériques TEC — CODE vs JPL (2023)

Projet du mémoire — comparaison des cartes IONEX de TEC produites par CODE et JPL pour l'année 2023.

Objectif : quantifier et visualiser les écarts (biais, RMSE, corrélation) entre les deux produits en temps, espace et par bandes de latitude/longitude.

**Auteurs :** Bassirou Kane

## Démarrage rapide

### Pré-requis

Installer les dépendances Python minimales :

```bash
pip install numpy requests tqdm unlzw3 matplotlib scipy
```

### Identifiants Earthdata
Le téléchargement depuis CDDIS requiert un compte Earthdata. Définissez les variables d'environnement suivantes avant d'exécuter les scripts de téléchargement :

- `EARTHDATA_USER`
- `EARTHDATA_PASS`

Ne laissez pas d'identifiants en clair dans les fichiers du dépôt. Ils doivent être retirés si présents dans [telecharger_ionex.py](telecharger_ionex.py) ou [test_nasa.py](test_nasa.py).

## Structure du dépôt / données

```
data/
├── ionex/2023/{JJJ}/       # IONEX bruts décompressés, par jour julien
├── base_tec/               # Base compressée : code_tec_2023.npz, jpl_tec_2023.npz, jours_*.npy
└── stats/
     ├── stats_journalieres.npz
     ├── stats_complementaires.npz
     ├── stats_zones_longitude.npz
     └── figures/             # PNG générés par les scripts d'analyse
```

## Pipeline (ordre d'exécution)

1. Téléchargement des IONEX
    - [telecharger_ionex.py](telecharger_ionex.py) — récupère et décompresse les IONEX depuis CDDIS
    - [telecharger_jpl_sideshow.py](telecharger_jpl_sideshow.py) — complète les jours JPL manquants via le miroir Sideshow
2. Parsing
    - [parseur_ionex.py](parseur_ionex.py) — parseur IONEX réutilisable
3. Construction de la base
    - [construire_base_tec.py](construire_base_tec.py) — assemble les fichiers en `.npz`
4. Statistiques et figures
    - [calculer_statistiques.py](calculer_statistiques.py) — MBE, RMSE, corrélation et figures principales
5. Analyses complémentaires
    - [analyse_complementaire.py](analyse_complementaire.py) — MBE en %, bandes de latitude, événement géomagnétique
    - [analyse_longitude.py](analyse_longitude.py) — comparaison par bandes de longitude
6. Cartographies saisonnières
    - [cartographies_saisonnieres.py](cartographies_saisonnieres.py)

Utilitaire : [Diagnostic ionex.py](Diagnostic%20ionex.py) — inspection brute d'un fichier IONEX pour débogage.

## Commandes principales

```bash
# 1. Téléchargement (nécessite EARTHDATA_USER / EARTHDATA_PASS)
python telecharger_ionex.py
python telecharger_jpl_sideshow.py

# 2. Construire la base TEC 2023
python construire_base_tec.py

# 3. Calculer statistiques et générer figures
python calculer_statistiques.py

# 4. Analyses complémentaires
python analyse_complementaire.py
python analyse_longitude.py

# 5. Cartographies saisonnières
python cartographies_saisonnieres.py
```

## Méthode (synthèse)

- Les cartes CODE (25/jour) sont sous-échantillonnées sur les époques paires pour s'aligner sur les 13 cartes JPL.
- Les métriques calculées : MBE (Mean Bias Error), RMSE, coefficient de corrélation de Pearson, et MBE relatif en %.
- Calculs effectués jour par jour sur cartes alignées, puis moyennés pour cartographies saisonnières et analyses par zone.

## État

Pipeline T1 → T6 implémenté et exécuté pour 2023. Prochaine étape : rédaction du mémoire et revue des figures.

---

Si vous souhaitez que j'ajoute une section Licence, des badges CI, ou des instructions pour exécuter des tests unitaires, dites-moi lesquelles ajouter.
