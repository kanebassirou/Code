"""
=============================================================================
T6 - Cartographies saisonnières CODE vs JPL 2023
Cartes de différence TEC pour les 4 dates clés :
  - Jour 080 : Équinoxe de printemps (21 mars 2023)
  - Jour 172 : Solstice d'été        (21 juin 2023)
  - Jour 264 : Équinoxe d'automne    (21 septembre 2023)
  - Jour 355 : Solstice d'hiver      (21 décembre 2023)
=============================================================================
Auteur  : Bassirou Kane
Mémoire : Master 2 Informatique - Génie Logiciel

PRÉREQUIS :
    Base de données TEC 2023 déjà construite (data/base_tec/*.npz)

UTILISATION :
    python cartographies_saisonnieres.py

RÉSULTATS :
    data/stats/figures/
    ├── cartes_saisons_diff.png       → 4 cartes de différence CODE-JPL
    ├── cartes_saisons_code.png       → 4 cartes TEC CODE seul
    ├── cartes_saisons_jpl.png        → 4 cartes TEC JPL seul
    └── comparaison_saisonniere.png   → profils latitudinaux moyens par saison
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from outils_figures import sauver_figure
from datetime import datetime, timedelta

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

# Les 4 dates clés de l'année (jours juliens)
SAISONS = [
    (80,  "Équinoxe de printemps", "21 mars 2023"),
    (172, "Solstice d'été",        "21 juin 2023"),
    (264, "Équinoxe d'automne",    "21 septembre 2023"),
    (355, "Solstice d'hiver",      "21 décembre 2023"),
]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.titlesize': 12, 'figure.dpi': 150})

# =============================================================================
# UTILITAIRES
# =============================================================================

def aligner_code_jpl(tec_code_jour, tec_jpl_jour):
    """Sous-échantillonne CODE (25 cartes/j) sur les époques JPL (13 cartes/j)."""
    return tec_code_jour[0::2], tec_jpl_jour

def charger_base():
    print(">>> Chargement de la base de données TEC 2023...")
    tec_code = np.load(os.path.join(DOSSIER_BASE, "code_tec_2023.npz"))['tec']
    tec_jpl  = np.load(os.path.join(DOSSIER_BASE, "jpl_tec_2023.npz"))['tec']
    print(f"    CODE : {tec_code.shape}  |  JPL : {tec_jpl.shape}")
    return tec_code, tec_jpl

def carte_moyenne_jour(tec_code, tec_jpl, jour):
    """
    Retourne les cartes moyennes (moyennées sur les époques du jour) :
    carte_code, carte_jpl, carte_diff -- toutes de shape (71, 73)
    """
    idx = jour - 1
    code_a, jpl_a = aligner_code_jpl(tec_code[idx], tec_jpl[idx])
    carte_code = np.nanmean(code_a, axis=0)
    carte_jpl  = np.nanmean(jpl_a, axis=0)
    carte_diff = carte_code - carte_jpl
    return carte_code, carte_jpl, carte_diff

# =============================================================================
# FIGURE 1 : CARTES DE DIFFÉRENCE (4 saisons)
# =============================================================================

def fig_cartes_diff_saisons(tec_code, tec_jpl, nom_fich):
    print("\n>>> Génération des cartes de différence saisonnières...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for ax, (jour, nom_saison, date_str) in zip(axes, SAISONS):
        _, _, diff = carte_moyenne_jour(tec_code, tec_jpl, jour)
        vmax = np.nanpercentile(np.abs(diff), 95)

        im = ax.pcolormesh(LON, LAT, diff, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax, shrink=0.8, label='TEC CODE−JPL (TECU)')
        ax.set_title(f"{nom_saison}\nJour {jour:03d} — {date_str}",
                     fontweight='bold', fontsize=11)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-90, 91, 30))
        ax.grid(True, alpha=0.2)

    fig.suptitle("Cartes de différence TEC CODE − JPL par saison — 2023",
                 fontweight='bold', fontsize=14)
    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

# =============================================================================
# FIGURE 2 & 3 : CARTES TEC CODE ET JPL SÉPARÉES
# =============================================================================

def fig_cartes_absolues_saisons(tec_code, tec_jpl, centre, nom_fich):
    """centre = 'code' ou 'jpl'"""
    print(f"\n>>> Génération des cartes TEC {centre.upper()} par saison...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    # TEC max global pour échelle commune (meilleure comparabilité)
    vmax_global = 0
    cartes = []
    for jour, nom_saison, date_str in SAISONS:
        c_code, c_jpl, _ = carte_moyenne_jour(tec_code, tec_jpl, jour)
        carte = c_code if centre == 'code' else c_jpl
        cartes.append((carte, nom_saison, date_str, jour))
        vmax_global = max(vmax_global, np.nanmax(carte))

    for ax, (carte, nom_saison, date_str, jour) in zip(axes, cartes):
        im = ax.pcolormesh(LON, LAT, carte, cmap='viridis',
                           vmin=0, vmax=vmax_global, shading='auto')
        plt.colorbar(im, ax=ax, shrink=0.8, label='TEC (TECU)')
        ax.set_title(f"{nom_saison}\nJour {jour:03d} — {date_str}",
                     fontweight='bold', fontsize=11)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.axhline(0, color='white', linewidth=0.5, linestyle=':')
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-90, 91, 30))
        ax.grid(True, alpha=0.15)

    fig.suptitle(f"Cartes TEC {centre.upper()} par saison — 2023",
                 fontweight='bold', fontsize=14)
    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

# =============================================================================
# FIGURE 4 : PROFILS LATITUDINAUX PAR SAISON
# =============================================================================

def fig_profils_latitudinaux(tec_code, tec_jpl, nom_fich):
    """
    Profil MBE(latitude) moyenné sur toutes les longitudes, pour chaque saison.
    Permet de voir comment le biais varie selon la latitude ET la saison.
    """
    print("\n>>> Génération des profils latitudinaux saisonniers...")
    fig, ax = plt.subplots(figsize=(12, 7))
    couleurs = ["#2E7D32", "#E65100", "#C62828", "#1565C0"]

    for (jour, nom_saison, date_str), couleur in zip(SAISONS, couleurs):
        _, _, diff = carte_moyenne_jour(tec_code, tec_jpl, jour)
        profil = np.nanmean(diff, axis=1)   # moyenne sur les longitudes -> (71,)
        ax.plot(profil, LAT, color=couleur, linewidth=2,
                label=f"{nom_saison} ({date_str})")

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel("MBE moyen (TECU)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Profil latitudinal du biais CODE − JPL par saison — 2023",
                 fontweight='bold')
    ax.set_yticks(range(-90, 91, 15))
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

# =============================================================================
# TABLEAU RÉCAPITULATIF
# =============================================================================

def tableau_recap_saisons(tec_code, tec_jpl):
    print("\n" + "=" * 70)
    print(f"{'Saison':<25} {'Jour':>6} {'MBE (TECU)':>12} {'RMSE (TECU)':>13}")
    print("-" * 70)

    resultats = []
    for jour, nom_saison, date_str in SAISONS:
        idx = jour - 1
        code_a, jpl_a = aligner_code_jpl(tec_code[idx], tec_jpl[idx])
        diff = code_a - jpl_a
        mbe  = float(np.nanmean(diff))
        rmse = float(np.sqrt(np.nanmean(diff ** 2)))
        print(f"{nom_saison:<25} {jour:>6} {mbe:>+12.3f} {rmse:>13.3f}")
        resultats.append((nom_saison, jour, date_str, mbe, rmse))

    print("=" * 70)
    return resultats

# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    print("=" * 60)
    print("T6 - Cartographies saisonnières CODE vs JPL 2023")
    print("=" * 60)

    tec_code, tec_jpl = charger_base()

    resultats = tableau_recap_saisons(tec_code, tec_jpl)

    fig_cartes_diff_saisons(tec_code, tec_jpl, "cartes_saisons_diff.png")
    fig_cartes_absolues_saisons(tec_code, tec_jpl, "code", "cartes_saisons_code.png")
    fig_cartes_absolues_saisons(tec_code, tec_jpl, "jpl",  "cartes_saisons_jpl.png")
    fig_profils_latitudinaux(tec_code, tec_jpl, "profils_latitudinaux_saisons.png")

    print("\n" + "=" * 60)
    print("RÉSUMÉ T6")
    print("=" * 60)
    for nom, jour, date_str, mbe, rmse in resultats:
        print(f"  {nom:<25} MBE={mbe:+.3f}  RMSE={rmse:.3f}")
    print(f"  4 figures générées dans {os.path.abspath(DOSSIER_FIGS)}/")
    print("=" * 60)
    print("\n✅ T6 terminée.")

if __name__ == "__main__":
    main()