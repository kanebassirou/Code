"""
=============================================================================
T5-bis - Analyse par bandes de longitude (demande M. Fleury du 12/07/2026)
Comparaison CODE vs JPL sur deux zones contrastées en densité de stations :
  - Océan Atlantique (15°W à 60°W) : peu de stations GNSS
  - Amérique         (75°W à 120°W) : forte densité de stations GNSS
Objectif : vérifier si la densité de stations explique les écarts CODE-JPL
=============================================================================
Auteur  : Bassirou Kane
Mémoire : Master 2 Informatique - Génie Logiciel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.stats import pearsonr

# =============================================================================
# CONFIGURATION
# =============================================================================

DOSSIER_BASE  = "data/base_tec"
DOSSIER_STATS = "data/stats"
DOSSIER_FIGS  = os.path.join(DOSSIER_STATS, "figures")
os.makedirs(DOSSIER_FIGS, exist_ok=True)

ANNEE = 2023
LAT   = np.linspace(87.5, -87.5, 71)
LON   = np.linspace(-180.0, 180.0, 73)

NOMS_MOIS = ['Jan','Fév','Mar','Avr','Mai','Juin',
             'Juil','Aoû','Sep','Oct','Nov','Déc']

# Zones de longitude définies par M. Fleury
ZONES_LON = [
    ("Océan Atlantique", -60, -15, "Peu de stations GNSS — modèles moins contraints"),
    ("Amérique",         -120, -75, "Forte densité de stations GNSS — modèles bien contraints"),
]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.titlesize': 13, 'figure.dpi': 150})

# =============================================================================
# UTILITAIRES
# =============================================================================

def date_du_jour(jour):
    return datetime(ANNEE, 1, 1) + timedelta(days=int(jour) - 1)

def mois_du_jour(jour):
    return date_du_jour(jour).month

def aligner_code_jpl(tec_code_jour, tec_jpl_jour):
    return tec_code_jour[0::2], tec_jpl_jour

def indices_lon(lon_min, lon_max):
    """Retourne les indices du tableau LON pour une zone de longitude."""
    return np.where((LON >= lon_min) & (LON <= lon_max))[0]

# =============================================================================
# CHARGEMENT
# =============================================================================

def charger_base():
    print(">>> Chargement de la base de données...")
    tec_code      = np.load(os.path.join(DOSSIER_BASE, "code_tec_2023.npz"))['tec']
    tec_jpl       = np.load(os.path.join(DOSSIER_BASE, "jpl_tec_2023.npz"))['tec']
    jours_communs = np.load(os.path.join(DOSSIER_BASE, "jours_communs.npy"))
    print(f"    {len(jours_communs)} jours communs chargés.")
    return tec_code, tec_jpl, jours_communs

# =============================================================================
# CALCUL PAR ZONE DE LONGITUDE
# =============================================================================

def calculer_stats_par_zone_lon(tec_code, tec_jpl, jours_communs):
    """
    Calcule MBE, RMSE et corrélation de Pearson pour chaque zone de longitude,
    pour chaque jour de l'année.
    """
    print("\n>>> Calcul MBE/RMSE/Pearson par zone de longitude...")
    resultats = {nom: {'mbe': np.full(len(jours_communs), np.nan),
                       'rmse': np.full(len(jours_communs), np.nan),
                       'r': np.full(len(jours_communs), np.nan)}
                 for nom, _, _, _ in ZONES_LON}

    for i, jour in enumerate(tqdm(jours_communs, desc="Zones lon", unit="jour")):
        idx = int(jour) - 1
        code_a, jpl_a = aligner_code_jpl(tec_code[idx], tec_jpl[idx])

        for nom, lon_min, lon_max, _ in ZONES_LON:
            idx_lon = indices_lon(lon_min, lon_max)
            if len(idx_lon) == 0:
                continue
            code_z = code_a[:, :, idx_lon]
            jpl_z  = jpl_a[:, :, idx_lon]
            diff   = code_z - jpl_z
            masque = ~np.isnan(diff)

            resultats[nom]['mbe'][i]  = float(np.nanmean(diff))
            resultats[nom]['rmse'][i] = float(np.sqrt(np.nanmean(diff ** 2)))

            if masque.sum() >= 2:
                c_flat = code_z[masque].flatten()
                j_flat = jpl_z[masque].flatten()
                r, _ = pearsonr(c_flat, j_flat)
                resultats[nom]['r'][i] = r

    return resultats

def tableau_zones_annuel(resultats):
    """Affiche le tableau annuel MBE/RMSE/r par zone de longitude."""
    print("\n" + "=" * 75)
    print(f"{'Zone':<20} {'Longitude':>15} {'MBE (TECU)':>12} "
          f"{'RMSE (TECU)':>13} {'r Pearson':>10}")
    print("-" * 75)

    resultats_annuel = []
    for nom, lon_min, lon_max, desc in ZONES_LON:
        mbe_m  = float(np.nanmean(resultats[nom]['mbe']))
        rmse_m = float(np.nanmean(resultats[nom]['rmse']))
        r_m    = float(np.nanmean(resultats[nom]['r']))
        lon_str = f"{lon_min}° à {lon_max}°"
        print(f"{nom:<20} {lon_str:>15} {mbe_m:>+12.3f} {rmse_m:>13.3f} {r_m:>10.4f}")
        resultats_annuel.append((nom, lon_min, lon_max, mbe_m, rmse_m, r_m))

    print("=" * 75)
    print("\nInterprétation attendue (hypothèse M. Fleury) :")
    print("  - Océan Atlantique : RMSE et dispersion potentiellement plus élevées")
    print("    (peu de stations -> modèles moins contraints, plus de divergence)")
    print("  - Amérique : RMSE potentiellement plus faible mais MBE peut différer")
    print("    (forte densité de stations -> modèles bien contraints par les mêmes données)")
    return resultats_annuel

def tableau_zones_mensuel(jours_communs, resultats):
    """Tableau mensuel MBE par zone de longitude."""
    print("\n" + "=" * 70)
    print(f"{'Mois':<10}", end="")
    for nom, _, _, _ in ZONES_LON:
        print(f"{nom + ' MBE':>18}", end="")
    print()
    print("-" * 70)

    for m in range(1, 13):
        idx_m = [i for i, j in enumerate(jours_communs) if mois_du_jour(j) == m]
        if not idx_m:
            continue
        print(f"{NOMS_MOIS[m-1]:<10}", end="")
        for nom, _, _, _ in ZONES_LON:
            mbe_m = np.nanmean(resultats[nom]['mbe'][idx_m])
            print(f"{mbe_m:>+18.3f}", end="")
        print()
    print("=" * 70)

# =============================================================================
# FIGURES
# =============================================================================

def fig_serie_zones(jours_communs, resultats, nom_fich):
    """Série temporelle MBE pour les deux zones de longitude."""
    dates    = [date_du_jour(j) for j in jours_communs]
    couleurs = ["#1565C0", "#E65100"]

    fig, ax = plt.subplots(figsize=(14, 5))
    for (nom, lon_min, lon_max, _), couleur in zip(ZONES_LON, couleurs):
        mbe_z = resultats[nom]['mbe']
        label = f"{nom} ({lon_min}\u00b0/{lon_max}\u00b0) moy={np.nanmean(mbe_z):+.3f}"
        ax.plot(dates, mbe_z, color=couleur, linewidth=1.0, alpha=0.85, label=label)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.set_title("MBE (CODE − JPL) par zone de longitude — 2023\n"
                 "Océan Atlantique (peu de stations) vs Amérique (forte densité)",
                 fontweight='bold', pad=10)
    ax.set_xlabel("Date (2023)")
    ax.set_ylabel("MBE (TECU)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(DOSSIER_FIGS, nom_fich), bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

def fig_barres_zones(resultats_annuel, nom_fich):
    """Diagramme en barres MBE/RMSE/r par zone."""
    noms  = [r[0] for r in resultats_annuel]
    mbes  = [r[3] for r in resultats_annuel]
    rmses = [r[4] for r in resultats_annuel]
    rs    = [r[5] for r in resultats_annuel]
    x     = np.arange(len(noms))
    couleurs = ["#1565C0", "#E65100"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.bar(x, mbes, color=couleurs, alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(noms, rotation=15, ha='right')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_ylabel("MBE (TECU)")
    ax1.set_title("MBE moyen annuel", fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.25)
    for i, v in enumerate(mbes):
        ax1.text(i, v - 0.05 if v < 0 else v + 0.05, f"{v:+.3f}",
                 ha='center', va='top' if v < 0 else 'bottom', fontsize=9)

    ax2.bar(x, rmses, color=couleurs, alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(noms, rotation=15, ha='right')
    ax2.set_ylabel("RMSE (TECU)")
    ax2.set_title("RMSE moyenne annuelle", fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.25)
    for i, v in enumerate(rmses):
        ax2.text(i, v + 0.05, f"{v:.3f}", ha='center', fontsize=9)

    ax3.bar(x, rs, color=couleurs, alpha=0.85)
    ax3.set_xticks(x); ax3.set_xticklabels(noms, rotation=15, ha='right')
    ax3.set_ylabel("r de Pearson")
    ax3.set_title("Corrélation moyenne annuelle", fontweight='bold')
    ax3.set_ylim(0.9, 1.0)
    ax3.grid(True, axis='y', alpha=0.25)
    for i, v in enumerate(rs):
        ax3.text(i, v + 0.002, f"{v:.4f}", ha='center', fontsize=9)

    fig.suptitle("Comparaison Oc\u00e9an Atlantique vs Am\u00e9rique — 2023",
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(DOSSIER_FIGS, nom_fich), bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

def fig_carte_zones_annotee(nom_fich):
    """Carte simple montrant les deux zones étudiées sur le globe."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    for nom, lon_min, lon_max, _ in ZONES_LON:
        couleur = "#1565C0" if "Atlantique" in nom else "#E65100"
        ax.axvspan(lon_min, lon_max, alpha=0.25, color=couleur, label=nom)
        ax.text((lon_min + lon_max) / 2, 80, nom, ha='center',
                fontsize=10, fontweight='bold', color=couleur)

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Localisation des zones \u00e9tudi\u00e9es", fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(DOSSIER_FIGS, nom_fich), bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    print("=" * 60)
    print("T5-bis - Analyse par bandes de longitude")
    print("        Demande de M. Fleury (12/07/2026)")
    print("=" * 60)

    tec_code, tec_jpl, jours_communs = charger_base()

    resultats = calculer_stats_par_zone_lon(tec_code, tec_jpl, jours_communs)
    resultats_annuel = tableau_zones_annuel(resultats)
    tableau_zones_mensuel(jours_communs, resultats)

    print("\n>>> Génération des figures...")
    fig_serie_zones(jours_communs, resultats, "serie_zones_longitude.png")
    fig_barres_zones(resultats_annuel, "barres_zones_longitude.png")
    fig_carte_zones_annotee("carte_zones_longitude.png")

    np.savez(os.path.join(DOSSIER_STATS, "stats_zones_longitude.npz"),
             jours=jours_communs,
             atlantique_mbe=resultats["Océan Atlantique"]['mbe'],
             amerique_mbe=resultats["Amérique"]['mbe'])

    print("\n" + "=" * 60)
    print("RÉSUMÉ T5-bis")
    print("=" * 60)
    for nom, lon_min, lon_max, mbe_m, rmse_m, r_m in resultats_annuel:
        print(f"  {nom:<20} MBE={mbe_m:+.3f}  RMSE={rmse_m:.3f}  r={r_m:.4f}")
    print("=" * 60)
    print("\n✅ T5-bis terminée. Figures dans data/stats/figures/")

if __name__ == "__main__":
    main()