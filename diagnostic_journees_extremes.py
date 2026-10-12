"""
=============================================================================
Diagnostic des journées extrêmes CODE − JPL (complément aux tâches T4/T5)
=============================================================================
Auteur  : Bassirou Kane
Mémoire : Master 2 Informatique - Génie Logiciel

OBJECTIF
    1. Dater précisément les valeurs extrêmes des indicateurs journaliers
       (MBE, RMSE, r) et classer les journées par RMSE décroissante.
    2. Caractériser la journée de RMSE maximale : profil par époque, biais
       par bande de latitude, carte de différence à l'époque la plus écartée.
    3. Test de cohérence temporelle propre à chaque produit : pour chaque
       époque, écart quadratique entre la carte du jour J et la moyenne des
       cartes du même produit aux jours J−1 et J+1. Une valeur élevée pour un
       seul produit signale une carte atypique de ce produit, indépendamment
       de l'autre.
    4. Positionner les journées d'orage géomagnétique de 2023 dans la
       distribution annuelle des indicateurs.

PRÉREQUIS
    pip install numpy scipy matplotlib
    Base TEC construite (data/base_tec/*.npz)

UTILISATION
    python diagnostic_journees_extremes.py

RÉSULTATS
    Tableaux en console + data/stats/figures/diagnostic_jour_rmse_max.png

IMPACT SUR LES AUTRES MODULES
    Aucun (lecture seule de la base TEC).
=============================================================================
"""

import os
from outils_figures import sauver_figure
import numpy as np
import matplotlib.pyplot as plt
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
LAT = np.linspace(87.5, -87.5, 71)
LON = np.linspace(-180.0, 180.0, 73)
HEURES = np.arange(0, 25, 2)
BANDES = [("Pôle Nord", 60, 90), ("Moy. lat. Nord", 30, 60), ("Tropiques Nord", 0, 30),
          ("Tropiques Sud", -30, 0), ("Moy. lat. Sud", -60, -30), ("Pôle Sud", -90, -60)]

# Journées d'orage géomagnétique de 2023 (Kp maximal journalier, GFZ Potsdam)
ORAGES = [(58, "7−"), (82, "7"), (83, "8"), (113, "8+"), (114, "8"), (309, "7+"), (335, "7−")]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.titlesize': 13, 'figure.dpi': 150})

def date_du_jour(jour):
    return datetime(ANNEE, 1, 1) + timedelta(days=int(jour) - 1)

# =============================================================================
# CALCULS
# =============================================================================

def charger_base():
    tec_code = np.load(os.path.join(DOSSIER_BASE, "code_tec_2023.npz"))['tec'].astype(np.float64)
    tec_jpl  = np.load(os.path.join(DOSSIER_BASE, "jpl_tec_2023.npz"))['tec'].astype(np.float64)
    jours    = np.load(os.path.join(DOSSIER_BASE, "jours_communs.npy"))
    return tec_code[:, 0::2], tec_jpl, jours          # CODE aligné sur 13 époques

def indicateurs_journaliers(code, jpl):
    diff = code - jpl
    mbe  = np.nanmean(diff, axis=(1, 2, 3))
    rmse = np.sqrt(np.nanmean(diff ** 2, axis=(1, 2, 3)))
    r = np.array([pearsonr(code[i].ravel(), jpl[i].ravel())[0] for i in range(len(code))])
    return mbe, rmse, r

def ecart_temporel(produit, i):
    """RMS, par époque, de la carte du jour i à la moyenne des jours i−1 et i+1."""
    voisins = (produit[i - 1] + produit[i + 1]) / 2
    return np.sqrt(np.nanmean((produit[i] - voisins) ** 2, axis=(1, 2)))

def continuite_minuit(produit):
    """RMS entre la carte de 24 h du jour J et celle de 0 h du jour J+1 (même instant)."""
    return np.array([np.sqrt(np.nanmean((produit[i, -1] - produit[i + 1, 0]) ** 2))
                     for i in range(len(produit) - 1)])

# =============================================================================
# FIGURE
# =============================================================================

def fig_diagnostic(code, jpl, i_max, ec_code, ec_jpl, ec_med_code, ec_med_jpl, nom_fich):
    jour = i_max + 1
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.38, wspace=0.22)

    # (a) RMSE par époque, jour extrême et jours voisins
    ax = fig.add_subplot(gs[0, 0])
    for k, (dj, st) in enumerate([(-2, ':'), (-1, '--'), (0, '-'), (1, '--'), (2, ':')]):
        j = i_max + dj
        if 0 <= j < len(code):
            pe = np.sqrt(np.nanmean((code[j] - jpl[j]) ** 2, axis=(1, 2)))
            ax.plot(HEURES, pe, linestyle=st, marker='o', markersize=3.5,
                    color='#C62828' if dj == 0 else '#757575',
                    linewidth=2.0 if dj == 0 else 1.0,
                    label=date_du_jour(j + 1).strftime('%d/%m'))
    ax.set_xticks(HEURES); ax.set_xlabel("Époque (heure TU)"); ax.set_ylabel("RMSE CODE − JPL (TECU)")
    ax.set_title(f"(a) RMSE par époque autour du {date_du_jour(jour).strftime('%d/%m/%Y')}", fontweight='bold')
    ax.grid(True, alpha=0.25); ax.legend(fontsize=9, ncol=5, loc='upper right')

    # (b) Test de cohérence temporelle propre à chaque produit
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(HEURES, ec_jpl, color='#E65100', marker='o', markersize=3.5, linewidth=2, label=f"JPL — {date_du_jour(jour).strftime('%d/%m')}")
    ax.plot(HEURES, ec_code, color='#1565C0', marker='o', markersize=3.5, linewidth=2, label=f"CODE — {date_du_jour(jour).strftime('%d/%m')}")
    ax.plot(HEURES, ec_med_jpl, color='#E65100', linestyle=':', linewidth=1.2, label="JPL — médiane annuelle")
    ax.plot(HEURES, ec_med_code, color='#1565C0', linestyle=':', linewidth=1.2, label="CODE — médiane annuelle")
    ax.set_xticks(HEURES); ax.set_xlabel("Époque (heure TU)")
    ax.set_ylabel("Écart à la moyenne J−1 / J+1 (TECU)")
    ax.set_title("(b) Cohérence temporelle de chaque produit", fontweight='bold')
    ax.grid(True, alpha=0.25); ax.legend(fontsize=9)

    # (c) Carte de différence à l'époque la plus écartée
    ax = fig.add_subplot(gs[1, :])
    pe = np.sqrt(np.nanmean((code[i_max] - jpl[i_max]) ** 2, axis=(1, 2)))
    k = int(np.argmax(pe))
    d = code[i_max, k] - jpl[i_max, k]
    v = np.nanpercentile(np.abs(d), 99.5)
    im = ax.pcolormesh(LON, LAT, d, cmap='RdBu_r', vmin=-v, vmax=v, shading='nearest')
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.set_title(f"(c) Différence TEC CODE − JPL le {date_du_jour(jour).strftime('%d/%m/%Y')} à {HEURES[k]:02d} h TU", fontweight='bold')
    ax.set_aspect('equal')
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01); cb.set_label("TEC CODE − JPL (TECU)")
    fig.suptitle(f"Diagnostic de la journée de RMSE maximale (jour {jour:03d})", fontweight='bold', fontsize=14)
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")
    return k

# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    print("=" * 70)
    print("Diagnostic des journées extrêmes CODE − JPL — 2023")
    print("=" * 70)
    code, jpl, jours = charger_base()
    mbe, rmse, r = indicateurs_journaliers(code, jpl)

    print("\n1. VALEURS EXTRÊMES DATÉES")
    for nom, s in [("MBE", mbe), ("RMSE", rmse), ("r", r)]:
        a, b = int(np.nanargmin(s)), int(np.nanargmax(s))
        print(f"   {nom:<5} min = {s[a]:+.3f} le {date_du_jour(a+1):%d/%m} (j{a+1:03d})   "
              f"max = {s[b]:+.3f} le {date_du_jour(b+1):%d/%m} (j{b+1:03d})")
    print("\n   Dix journées de RMSE la plus élevée :")
    for i in np.argsort(rmse)[::-1][:10]:
        print(f"     j{i+1:03d} {date_du_jour(i+1):%d/%m}  RMSE={rmse[i]:.3f}  MBE={mbe[i]:+.3f}  r={r[i]:.4f}")

    i_max = int(np.nanargmax(rmse))
    print(f"\n2. JOURNÉE DE RMSE MAXIMALE : j{i_max+1:03d} ({date_du_jour(i_max+1):%d/%m/%Y})")
    pe = np.sqrt(np.nanmean((code[i_max] - jpl[i_max]) ** 2, axis=(1, 2)))
    print("   RMSE par époque : " + " ".join(f"{h:02d}h={v:.2f}" for h, v in zip(HEURES, pe)))
    for nom, a, b in BANDES:
        il = np.where((LAT >= a) & (LAT <= b))[0]
        d = code[i_max][:, il] - jpl[i_max][:, il]
        print(f"   {nom:<16} MBE={np.nanmean(d):+.3f}  RMSE={np.sqrt(np.nanmean(d**2)):.3f}")

    print("\n3. COHÉRENCE TEMPORELLE DE CHAQUE PRODUIT (écart à la moyenne J−1/J+1, TECU)")
    ec_all_c = np.array([ecart_temporel(code, i) for i in range(1, len(code) - 1)])
    ec_all_j = np.array([ecart_temporel(jpl, i) for i in range(1, len(jpl) - 1)])
    ec_c, ec_j = ec_all_c[i_max - 1], ec_all_j[i_max - 1]
    print(f"   CODE j{i_max+1:03d} : max={ec_c.max():.2f}  (médiane annuelle de ce max : {np.median(ec_all_c.max(axis=1)):.2f} ; "
          f"rang {int((ec_all_c.max(axis=1) > ec_c.max()).sum())+1}/{len(ec_all_c)})")
    print(f"   JPL  j{i_max+1:03d} : max={ec_j.max():.2f}  (médiane annuelle de ce max : {np.median(ec_all_j.max(axis=1)):.2f} ; "
          f"rang {int((ec_all_j.max(axis=1) > ec_j.max()).sum())+1}/{len(ec_all_j)})")
    top_j = np.argsort(ec_all_j.max(axis=1))[::-1][:5] + 2
    top_c = np.argsort(ec_all_c.max(axis=1))[::-1][:5] + 2
    print("   5 journées les plus atypiques JPL  :", ", ".join(f"j{j:03d}" for j in top_j))
    print("   5 journées les plus atypiques CODE :", ", ".join(f"j{j:03d}" for j in top_c))

    print("\n4. CONTINUITÉ À MINUIT (carte 24 h du jour J vs carte 0 h du jour J+1, TECU)")
    for nom, p in [("CODE", code), ("JPL", jpl)]:
        c = continuite_minuit(p)
        print(f"   {nom:<5} médiane={np.median(c):.2f}  moyenne={c.mean():.2f}  P95={np.percentile(c,95):.2f}  max={c.max():.2f}")

    print("\n5. JOURNÉES D'ORAGE GÉOMAGNÉTIQUE (rang de la RMSE parmi 365 jours)")
    for j, kp in ORAGES:
        i = j - 1
        mois = date_du_jour(j).month
        idx_m = [k for k in range(365) if date_du_jour(k + 1).month == mois]
        print(f"   j{j:03d} {date_du_jour(j):%d/%m}  Kp max={kp:<3} MBE={mbe[i]:+.3f}  RMSE={rmse[i]:.3f} "
              f"(rang {int((rmse > rmse[i]).sum())+1}/365 ; moyenne du mois {np.mean(rmse[idx_m]):.3f})  r={r[i]:.4f}")

    print("\n>>> Figure...")
    k = fig_diagnostic(code, jpl, i_max, ec_c, ec_j, np.median(ec_all_c, axis=0), np.median(ec_all_j, axis=0),
                       "diagnostic_jour_rmse_max.png")
    d = code[i_max, k] - jpl[i_max, k]
    ii = np.unravel_index(np.nanargmax(np.abs(d)), d.shape)
    print(f"   Époque la plus écartée : {HEURES[k]:02d} h TU ; écart extrême {d[ii]:+.1f} TECU à "
          f"{LAT[ii[0]]:+.1f}° / {LON[ii[1]]:+.1f}°")
    print("\n✅ Diagnostic terminé.")

if __name__ == "__main__":
    main()
