"""
=============================================================================
T4 - Calcul des indicateurs statistiques CODE vs JPL
MBE, RMSE, corrélation de Pearson pour les 365 jours de 2023
=============================================================================
Auteur  : Bassirou Kane
Mémoire : Master 2 Informatique - Génie Logiciel

PRÉREQUIS :
    pip install numpy matplotlib scipy tqdm

UTILISATION :
    python calculer_statistiques.py

RÉSULTATS :
    data/stats/
    ├── stats_journalieres.npz        → MBE, RMSE, r pour chaque jour
    └── figures/
        ├── serie_MBE.png             → biais moyen annuel
        ├── serie_RMSE.png            → écart quadratique annuel
        ├── serie_correlation.png     → corrélation annuelle
        ├── carte_diff_moyenne.png    → carte mondiale différences
        ├── boxplot_mensuel_MBE.png   → variabilité mensuelle
        └── heatmap_mbe_lat_mois.png  → MBE par latitude et mois
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from scipy.stats import pearsonr
from tqdm import tqdm
import os
from outils_figures import sauver_figure
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================

DOSSIER_BASE  = "data/base_tec"
DOSSIER_STATS = "data/stats"
DOSSIER_FIGS  = os.path.join(DOSSIER_STATS, "figures")
os.makedirs(DOSSIER_STATS, exist_ok=True)
os.makedirs(DOSSIER_FIGS,  exist_ok=True)

ANNEE = 2023

# Grille spatiale
LAT = np.linspace(87.5, -87.5, 71)
LON = np.linspace(-180.0, 180.0, 73)

# Style global des figures
plt.rcParams.update({
    'font.family'  : 'DejaVu Sans',
    'font.size'    : 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi'   : 150,
})

# =============================================================================
# UTILITAIRES
# =============================================================================

def date_du_jour(jour):
    """Convertit un numéro de jour julien (1-365) en datetime."""
    return datetime(ANNEE, 1, 1) + timedelta(days=int(jour) - 1)

def mois_du_jour(jour):
    """Retourne le numéro de mois (1-12) pour un jour julien."""
    return date_du_jour(jour).month

NOMS_MOIS = ['Jan','Fév','Mar','Avr','Mai','Juin',
              'Juil','Aoû','Sep','Oct','Nov','Déc']

# =============================================================================
# CHARGEMENT
# =============================================================================

def charger_base():
    print(">>> Chargement de la base de données TEC 2023...")
    tec_code      = np.load(os.path.join(DOSSIER_BASE, "code_tec_2023.npz"))['tec']
    tec_jpl       = np.load(os.path.join(DOSSIER_BASE, "jpl_tec_2023.npz"))['tec']
    jours_communs = np.load(os.path.join(DOSSIER_BASE, "jours_communs.npy"))
    print(f"    CODE : {tec_code.shape}  |  JPL : {tec_jpl.shape}")
    print(f"    Jours communs : {len(jours_communs)}")
    return tec_code, tec_jpl, jours_communs

# =============================================================================
# ALIGNEMENT TEMPOREL
# =============================================================================

def aligner_cartes(tec_code_jour, tec_jpl_jour):
    """
    CODE : 25 cartes à 0h, 1h, ..., 24h
    JPL  : 13 cartes à 0h, 2h, ..., 24h
    On sous-échantillonne CODE sur les époques paires → 13 cartes communes.
    Retourne (code_aligne, jpl) tous deux de shape (13, 71, 73).
    """
    return tec_code_jour[0::2], tec_jpl_jour

# =============================================================================
# CALCUL DES INDICATEURS JOURNALIERS
# =============================================================================

def calculer_stats_journalieres(tec_code, tec_jpl, jours_communs):
    print("\n>>> Calcul MBE / RMSE / Pearson pour les 365 jours...")

    n = len(jours_communs)
    mbe  = np.full(n, np.nan)
    rmse = np.full(n, np.nan)
    r    = np.full(n, np.nan)

    for i, jour in enumerate(tqdm(jours_communs, desc="Stats", unit="jour")):
        idx = int(jour) - 1
        code_a, jpl_a = aligner_cartes(tec_code[idx], tec_jpl[idx])

        diff   = code_a - jpl_a
        masque = ~np.isnan(diff)

        if masque.sum() < 10:
            continue

        mbe[i]  = float(np.nanmean(diff))
        rmse[i] = float(np.sqrt(np.nanmean(diff ** 2)))

        c_flat = code_a[masque].flatten()
        j_flat = jpl_a[masque].flatten()
        if len(c_flat) >= 2:
            r[i], _ = pearsonr(c_flat, j_flat)

    return mbe, rmse, r

# =============================================================================
# CARTE DE DIFFÉRENCE MOYENNE ANNUELLE (71 x 73)
# =============================================================================

def calculer_carte_diff(tec_code, tec_jpl, jours_communs):
    print("\n>>> Calcul de la carte de différence moyenne annuelle...")
    somme = np.zeros((71, 73))
    nb    = np.zeros((71, 73))

    for jour in tqdm(jours_communs, desc="Carte", unit="jour"):
        idx = int(jour) - 1
        code_a, jpl_a = aligner_cartes(tec_code[idx], tec_jpl[idx])
        diff   = code_a - jpl_a
        masque = ~np.isnan(diff)
        somme += np.nansum(diff,  axis=0)
        nb    += masque.sum(axis=0)

    return np.where(nb > 0, somme / nb, np.nan)

# =============================================================================
# MBE MOYEN PAR LATITUDE ET PAR MOIS (71 x 12)
# =============================================================================

def calculer_mbe_lat_mois(tec_code, tec_jpl, jours_communs):
    print("\n>>> Calcul MBE par latitude et par mois...")
    somme = np.zeros((71, 12))
    nb    = np.zeros((71, 12))

    for jour in tqdm(jours_communs, desc="Lat/Mois", unit="jour"):
        m   = mois_du_jour(jour) - 1   # 0-11
        idx = int(jour) - 1
        code_a, jpl_a = aligner_cartes(tec_code[idx], tec_jpl[idx])
        diff = code_a - jpl_a          # (13, 71, 73)
        # Moyenne sur longitudes et époques pour chaque latitude
        diff_lat = np.nanmean(diff, axis=(0, 2))   # (71,)
        masque   = ~np.isnan(diff_lat)
        somme[masque, m] += diff_lat[masque]
        nb[masque, m]    += 1

    return np.where(nb > 0, somme / nb, np.nan)

# =============================================================================
# FIGURES
# =============================================================================

def fig_serie(jours, valeurs, titre, ylabel, couleur, nom_fich, ligne_zero=True):
    """Série temporelle annuelle."""
    dates = [date_du_jour(j) for j in jours]
    moy   = np.nanmean(valeurs)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, valeurs, color=couleur, linewidth=1.0, alpha=0.85)
    ax.fill_between(dates, valeurs, alpha=0.12, color=couleur)

    if ligne_zero:
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)

    ax.axhline(moy, color=couleur, linewidth=1.8, linestyle='-.',
               label=f'Moyenne annuelle : {moy:+.3f}' if ligne_zero else f'Moyenne : {moy:.3f}')

    ax.set_title(titre, fontweight='bold', pad=10)
    ax.set_xlabel("Date (2023)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    chemin = os.path.join(DOSSIER_FIGS, nom_fich)
    sauver_figure(fig, chemin)
    plt.close(fig)
    print(f"    ✓ {nom_fich}")


def fig_carte_diff(carte, nom_fich):
    """Carte mondiale de différence moyenne CODE - JPL."""
    vmax = np.nanpercentile(np.abs(carte), 95)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.pcolormesh(LON, LAT, carte,
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label('TEC CODE − TEC JPL (TECU)', fontsize=11)

    ax.set_title("Carte de différence moyenne annuelle TEC CODE − TEC JPL (2023)",
                 fontweight='bold', pad=10)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.axhline(0,  color='black', linewidth=0.6, linestyle=':')
    ax.axvline(0,  color='black', linewidth=0.6, linestyle=':')
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90,  91, 30))
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    chemin = os.path.join(DOSSIER_FIGS, nom_fich)
    sauver_figure(fig, chemin)
    plt.close(fig)
    print(f"    ✓ {nom_fich}")


def fig_boxplot_mensuel(jours, mbe, nom_fich):
    """Boxplot du MBE par mois."""
    donnees_mois = [[] for _ in range(12)]
    for i, jour in enumerate(jours):
        m = mois_du_jour(jour) - 1
        if not np.isnan(mbe[i]):
            donnees_mois[m].append(mbe[i])

    fig, ax = plt.subplots(figsize=(13, 5))
    import matplotlib; mpl_v = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
    lbl_key = "tick_labels" if mpl_v >= (3, 9) else "labels"
    bp = ax.boxplot(donnees_mois, **{lbl_key: NOMS_MOIS},
                    patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))

    # Colorier les boîtes par saison
    couleurs = ['#4FC3F7','#4FC3F7','#A5D6A7','#A5D6A7','#A5D6A7',
                '#FFCC80','#FFCC80','#FFCC80','#EF9A9A','#EF9A9A',
                '#EF9A9A','#4FC3F7']
    for patch, c in zip(bp['boxes'], couleurs):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.axhline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.5)
    ax.set_title("Variabilité mensuelle du MBE (CODE − JPL) — 2023",
                 fontweight='bold', pad=10)
    ax.set_xlabel("Mois")
    ax.set_ylabel("MBE (TECU)")
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    chemin = os.path.join(DOSSIER_FIGS, nom_fich)
    sauver_figure(fig, chemin)
    plt.close(fig)
    print(f"    ✓ {nom_fich}")


def fig_heatmap_lat_mois(mbe_lat_mois, nom_fich):
    """Heatmap MBE par latitude (axe Y) et par mois (axe X)."""
    vmax = np.nanpercentile(np.abs(mbe_lat_mois), 95)

    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.pcolormesh(np.arange(12), LAT, mbe_lat_mois,
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('MBE CODE − JPL (TECU)', fontsize=11)

    ax.set_xticks(range(12))
    ax.set_xticklabels(NOMS_MOIS)
    ax.set_yticks(range(-90, 91, 15))
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.set_title("MBE moyen par latitude et par mois — CODE vs JPL (2023)",
                 fontweight='bold', pad=10)
    ax.set_xlabel("Mois")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    chemin = os.path.join(DOSSIER_FIGS, nom_fich)
    sauver_figure(fig, chemin)
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

# =============================================================================
# RÉSUMÉ STATISTIQUE ANNUEL
# =============================================================================

def afficher_resume(jours, mbe, rmse, r):
    print("\n" + "=" * 60)
    print("RÉSUMÉ STATISTIQUE ANNUEL (365 jours)")
    print("=" * 60)
    print(f"  MBE moyen annuel     : {np.nanmean(mbe):+.3f} TECU")
    print(f"  MBE médian           : {np.nanmedian(mbe):+.3f} TECU")
    print(f"  MBE min / max        : {np.nanmin(mbe):+.3f} / {np.nanmax(mbe):+.3f} TECU")
    print(f"  RMSE moyen annuel    : {np.nanmean(rmse):.3f} TECU")
    print(f"  RMSE min / max       : {np.nanmin(rmse):.3f} / {np.nanmax(rmse):.3f} TECU")
    print(f"  Corrélation moyenne  : {np.nanmean(r):.4f}")
    print(f"  Corrélation min      : {np.nanmin(r):.4f}")
    print("=" * 60)

    # Résumé mensuel MBE
    print("\n  MBE moyen par mois :")
    mbe_par_mois = {}
    for i, jour in enumerate(jours):
        m = mois_du_jour(jour)
        mbe_par_mois.setdefault(m, []).append(mbe[i])
    for m in range(1, 13):
        vals = [v for v in mbe_par_mois.get(m, []) if not np.isnan(v)]
        if vals:
            print(f"    {NOMS_MOIS[m-1]:>4} : {np.mean(vals):+.3f} TECU  ({len(vals)} jours)")

# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    print("=" * 60)
    print("T4 - Calcul des indicateurs statistiques CODE vs JPL")
    print("     365 jours de l'année 2023")
    print("=" * 60)

    # 1. Chargement
    tec_code, tec_jpl, jours_communs = charger_base()

    # 2. Stats journalières
    mbe, rmse, r = calculer_stats_journalieres(tec_code, tec_jpl, jours_communs)

    # 3. Carte différence
    carte_diff = calculer_carte_diff(tec_code, tec_jpl, jours_communs)

    # 4. MBE par latitude et mois
    mbe_lat_mois = calculer_mbe_lat_mois(tec_code, tec_jpl, jours_communs)

    # 5. Sauvegarde
    np.savez(
        os.path.join(DOSSIER_STATS, "stats_journalieres.npz"),
        jours=jours_communs, mbe=mbe, rmse=rmse, r=r,
        carte_diff=carte_diff, mbe_lat_mois=mbe_lat_mois
    )
    print(f"\n>>> Statistiques sauvegardées : data/stats/stats_journalieres.npz")

    # 6. Figures
    print("\n>>> Génération des figures...")

    fig_serie(jours_communs, mbe,
              "Biais moyen journalier MBE (CODE − JPL) — 2023",
              "MBE (TECU)", "#E74C3C", "serie_MBE.png")

    fig_serie(jours_communs, rmse,
              "Erreur quadratique moyenne RMSE (CODE vs JPL) — 2023",
              "RMSE (TECU)", "#2980B9", "serie_RMSE.png", ligne_zero=False)

    fig_serie(jours_communs, r,
              "Corrélation de Pearson journalière (CODE vs JPL) — 2023",
              "r de Pearson", "#27AE60", "serie_correlation.png", ligne_zero=False)

    fig_carte_diff(carte_diff, "carte_diff_moyenne.png")

    fig_boxplot_mensuel(jours_communs, mbe, "boxplot_mensuel_MBE.png")

    fig_heatmap_lat_mois(mbe_lat_mois, "heatmap_mbe_lat_mois.png")

    # 7. Résumé
    afficher_resume(jours_communs, mbe, rmse, r)

    print(f"\n✅ T4 terminée.")
    print(f"   Figures dans : {os.path.abspath(DOSSIER_FIGS)}/")
    print("   Prochaine étape : T5 - Analyse de la variabilité")

if __name__ == "__main__":
    main()