"""
=============================================================================
T5 - Analyse complémentaire suite aux remarques de l'encadreur
1. MBE en pourcentage (biais normalisé) - MÉTHODE UNIQUE ET COHÉRENTE
2. MBE normalisé au nombre de jours du mois
3. Analyse par bandes de latitude (MBE et RMSE par région)
4. Comparaison multi-centres sur l'événement du 27-28 février 2023
    (CODE, JPL, ESAG, AIR/CAS, UPC)
=============================================================================
Auteur  : Bassirou Kane
Mémoire : Master 2 Informatique - Génie Logiciel

CORRECTION IMPORTANTE :
Le MBE% est calculé UNE SEULE FOIS par jour (mbe_pct[i]), puis cette
même série journalière est réutilisée par np.nanmean() pour produire :
  - la valeur affichée sur les figures (ligne pointillée "Moyenne")
  - la valeur affichée dans le tableau mensuel (colonne MBE %)
  - la valeur annuelle du résumé final
Il n'existe donc qu'UNE SEULE méthode et UNE SEULE source de vérité
(la série journalière mbe_pct) pour toutes les valeurs de pourcentage
affichées dans ce script, que ce soit en figure ou en tableau.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from outils_figures import sauver_figure
from tqdm import tqdm
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '.')
from parseur_ionex import lire_ionex

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

JOURS_PAR_MOIS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

BANDES = [
    ("Pôle Nord",       60,  90),
    ("Moy. lat. Nord",  30,  60),
    ("Tropiques Nord",   0,  30),
    ("Tropiques Sud",  -30,   0),
    ("Moy. lat. Sud",  -60, -30),
    ("Pôle Sud",       -90, -60),
]

CENTRES_EVT = {
    "CODE": {
        "058": ("COD0OPSFIN", "COD0OPSFIN_20230580000_01D_01H_GIM.INX"),
        "059": ("COD0OPSFIN", "COD0OPSFIN_20230590000_01D_01H_GIM.INX"),
    },
    "JPL": {
        "058": ("jplg",       "jplg0580.23i"),
        "059": ("jplg",       "jplg0590.23i"),
    },
    "AIR/CAS": {
        "058": ("CAS0OPSFIN", "CAS0OPSFIN_20230580000_01D_30M_GIM.INX"),
        "059": ("CAS0OPSFIN", "CAS0OPSFIN_20230590000_01D_30M_GIM.INX"),
    },
    "ESAG": {
        "058": ("ESA0OPSFIN", "ESA0OPSFIN_20230580000_01D_02H_GIM.INX"),
        "059": ("ESA0OPSFIN", "ESA0OPSFIN_20230590000_01D_02H_GIM.INX"),
    },
    "UPC": {
        "058": ("upcg",       "upcg0580.23i"),
        "059": ("upcg",       "upcg0590.23i"),
    },
}

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

def indices_lat(lat_min, lat_max):
    return np.where((LAT >= lat_min) & (LAT <= lat_max))[0]

# =============================================================================
# CHARGEMENT BASE
# =============================================================================

def charger_base():
    print(">>> Chargement de la base de données...")
    tec_code      = np.load(os.path.join(DOSSIER_BASE, "code_tec_2023.npz"))['tec']
    tec_jpl       = np.load(os.path.join(DOSSIER_BASE, "jpl_tec_2023.npz"))['tec']
    jours_communs = np.load(os.path.join(DOSSIER_BASE, "jours_communs.npy"))
    stats         = np.load(os.path.join(DOSSIER_STATS, "stats_journalieres.npz"))
    mbe           = stats['mbe']
    print(f"    {len(jours_communs)} jours communs chargés.")
    return tec_code, tec_jpl, jours_communs, mbe

# =============================================================================
# 1. SÉRIE MBE% JOURNALIÈRE — SOURCE UNIQUE DE VÉRITÉ
# =============================================================================

def calculer_mbe_pct(tec_code, tec_jpl, jours_communs):
    """
    Calcule la série journalière mbe_pct[i] = (MBE_jour / TEC_moyen_jour) x 100.
    Cette série est LA SEULE source utilisée ensuite pour :
      - le tracé des figures (moyenne = np.nanmean(mbe_pct))
      - le tableau mensuel (moyenne par mois = np.nanmean(mbe_pct[mois]))
      - le résumé annuel (moyenne annuelle = np.nanmean(mbe_pct))
    Une seule formule, une seule série, donc une seule valeur cohérente
    partout où le pourcentage est affiché.
    """
    print("\n>>> Calcul du MBE en pourcentage (série journalière unique)...")
    mbe_pct = np.full(len(jours_communs), np.nan)
    tec_moy = np.full(len(jours_communs), np.nan)

    for i, jour in enumerate(tqdm(jours_communs, desc="MBE%", unit="jour")):
        idx = int(jour) - 1
        code_a, jpl_a = aligner_code_jpl(tec_code[idx], tec_jpl[idx])
        diff     = code_a - jpl_a
        mbe_abs  = float(np.nanmean(diff))
        moy_jour = float(np.nanmean((code_a + jpl_a) / 2))
        if moy_jour != 0 and not np.isnan(moy_jour):
            mbe_pct[i] = (mbe_abs / moy_jour) * 100
            tec_moy[i] = moy_jour

    return mbe_pct, tec_moy

def tableau_mensuel_complet(jours_communs, mbe, mbe_pct, tec_moy):
    """
    Tableau mensuel utilisant EXCLUSIVEMENT la série mbe_pct comme source
    pour la colonne MBE(%), afin de garantir la cohérence avec les figures.
    """
    print("\n" + "=" * 75)
    print(f"{'Mois':<10} {'Nb jours':>8} {'TEC moy':>9} "
          f"{'MBE/jour (TECU)':>16} {'MBE (%)':>10}")
    print(f"{'':10} {'du mois':>8} {'(TECU)':>9} "
          f"{'= Σ MBE / N_jours':>16} {'= moy(mbe_pct)':>10}")
    print("-" * 75)

    resultats = []
    for m in range(1, 13):
        idx_m   = [i for i, j in enumerate(jours_communs) if mois_du_jour(j) == m]
        nb_jours_mois = JOURS_PAR_MOIS[m - 1]
        if not idx_m:
            continue

        somme_mbe = float(np.nansum(mbe[idx_m]))
        mbe_norm  = somme_mbe / nb_jours_mois

        # Source unique : la même série mbe_pct utilisée par les figures
        mbe_pct_m = float(np.nanmean(mbe_pct[idx_m]))
        tec_m     = float(np.nanmean(tec_moy[idx_m]))

        print(f"{NOMS_MOIS[m-1]:<10} {nb_jours_mois:>8} {tec_m:>9.2f} "
              f"{mbe_norm:>+16.3f} {mbe_pct_m:>+10.2f}%")
        resultats.append((NOMS_MOIS[m-1], nb_jours_mois, tec_m, mbe_norm, mbe_pct_m))

    print("=" * 75)
    mbe_norm_ann  = float(np.nansum(mbe)) / 365
    # Même source unique pour l'annuel : np.nanmean(mbe_pct)
    mbe_pct_ann   = float(np.nanmean(mbe_pct))
    tec_ann       = float(np.nanmean(tec_moy))
    print(f"{'ANNUEL':<10} {'365':>8} {tec_ann:>9.2f} "
          f"{mbe_norm_ann:>+16.3f} {mbe_pct_ann:>+10.2f}%")
    print("=" * 75)
    print("\nNote : le MBE(%) affiché (tableau ET figures) est TOUJOURS")
    print("       la moyenne de la série journalière mbe_pct — une seule")
    print("       méthode, une seule valeur, aucune ambiguïté possible.")
    return resultats

def fig_mbe_pct(jours_communs, mbe_pct, nom_fich):
    """Figure utilisant la même série mbe_pct que le tableau."""
    dates = [date_du_jour(j) for j in jours_communs]
    moy   = np.nanmean(mbe_pct)   # <- identique à mbe_pct_ann du tableau
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, mbe_pct, color="#E74C3C", linewidth=1.0, alpha=0.85)
    ax.fill_between(dates, mbe_pct, alpha=0.12, color="#E74C3C")
    ax.axhline(0,   color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.axhline(moy, color="#E74C3C", linewidth=1.8, linestyle='-.',
               label=f'Moyenne annuelle : {moy:+.2f}%')
    ax.set_title("Biais relatif journalier MBE% (CODE − JPL) / TEC_moyen — 2023",
                 fontweight='bold', pad=10)
    ax.set_xlabel("Date (2023)")
    ax.set_ylabel("MBE (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

def fig_comparaison_mbe_absolu_pct(jours_communs, mbe, mbe_pct, nom_fich):
    dates = [date_du_jour(j) for j in jours_communs]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(dates, mbe, color="#E74C3C", linewidth=1.0)
    ax1.axhline(np.nanmean(mbe), color="#E74C3C", linewidth=1.5, linestyle='-.',
                label=f'Moy. : {np.nanmean(mbe):+.3f} TECU')
    ax1.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.4)
    ax1.set_ylabel("MBE (TECU)")
    ax1.set_title("Biais CODE − JPL : MBE (TECU) et MBE relatif (%)", fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.25)
    ax2.plot(dates, mbe_pct, color="#8E44AD", linewidth=1.0)
    ax2.axhline(np.nanmean(mbe_pct), color="#8E44AD", linewidth=1.5, linestyle='-.',
                label=f'Moy. : {np.nanmean(mbe_pct):+.2f}%')
    ax2.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.4)
    ax2.set_ylabel("MBE (%)")
    ax2.set_xlabel("Date (2023)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

# =============================================================================
# 2. ANALYSE PAR BANDES DE LATITUDE
# =============================================================================

def calculer_stats_par_bandes(tec_code, tec_jpl, jours_communs):
    print("\n>>> Calcul MBE/RMSE par bandes de latitude...")
    resultats = {nom: {'mbe': np.full(len(jours_communs), np.nan),
                       'rmse': np.full(len(jours_communs), np.nan)}
                 for nom, _, _ in BANDES}
    for i, jour in enumerate(tqdm(jours_communs, desc="Bandes lat", unit="jour")):
        idx = int(jour) - 1
        code_a, jpl_a = aligner_code_jpl(tec_code[idx], tec_jpl[idx])
        for nom, lat_min, lat_max in BANDES:
            idx_lat = indices_lat(lat_min, lat_max)
            if len(idx_lat) == 0:
                continue
            diff = (code_a - jpl_a)[:, idx_lat, :]
            resultats[nom]['mbe'][i]  = float(np.nanmean(diff))
            resultats[nom]['rmse'][i] = float(np.sqrt(np.nanmean(diff ** 2)))
    return resultats

def tableau_stats_bandes_annuel(stats_bandes):
    print("\n" + "=" * 60)
    print(f"{'Bande de latitude':<25} {'Lat':>10} {'MBE (TECU)':>12} {'RMSE (TECU)':>13}")
    print("-" * 60)
    resultats = []
    for nom, lat_min, lat_max in BANDES:
        mbe_m  = float(np.nanmean(stats_bandes[nom]['mbe']))
        rmse_m = float(np.nanmean(stats_bandes[nom]['rmse']))
        lat_str = f"{lat_min}° à {lat_max}°"
        print(f"{nom:<25} {lat_str:>10} {mbe_m:>+12.3f} {rmse_m:>13.3f}")
        resultats.append((nom, lat_min, lat_max, mbe_m, rmse_m))
    print("=" * 60)
    return resultats

def fig_mbe_par_bandes(jours_communs, stats_bandes, nom_fich):
    dates    = [date_du_jour(j) for j in jours_communs]
    couleurs = ["#1565C0","#1976D2","#42A5F5","#EF9A9A","#E53935","#B71C1C"]
    fig, ax = plt.subplots(figsize=(14, 6))
    for (nom, lat_min, lat_max), couleur in zip(BANDES, couleurs):
        mbe_b = stats_bandes[nom]['mbe']
        label = f"{nom} ({lat_min}°/{lat_max}°) moy={np.nanmean(mbe_b):+.2f}"
        ax.plot(dates, mbe_b, color=couleur, linewidth=1.0, alpha=0.8, label=label)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.set_title("MBE (CODE − JPL) par bande de latitude — 2023", fontweight='bold', pad=10)
    ax.set_xlabel("Date (2023)")
    ax.set_ylabel("MBE (TECU)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

def fig_barres_bandes(resultats_bandes, nom_fich):
    noms  = [r[0] for r in resultats_bandes]
    mbes  = [r[3] for r in resultats_bandes]
    rmses = [r[4] for r in resultats_bandes]
    x     = np.arange(len(noms))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.barh(x, mbes, color="#E53935", alpha=0.8)
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.set_yticks(x); ax1.set_yticklabels(noms, fontsize=10)
    ax1.set_xlabel("MBE (TECU)")
    ax1.set_title("MBE moyen annuel par bande de latitude", fontweight='bold')
    ax1.grid(True, axis='x', alpha=0.25)
    for i, v in enumerate(mbes):
        ax1.text(v - 0.05, i, f"{v:+.3f}", va='center', ha='right', fontsize=9)
    ax2.barh(x, rmses, color="#1565C0", alpha=0.8)
    ax2.set_yticks(x); ax2.set_yticklabels(noms, fontsize=10)
    ax2.set_xlabel("RMSE (TECU)")
    ax2.set_title("RMSE moyen annuel par bande de latitude", fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.25)
    for i, v in enumerate(rmses):
        ax2.text(v + 0.02, i, f"{v:.3f}", va='center', ha='left', fontsize=9)
    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

def telecharger_centres_fevrier():
    print("\n>>> Localisation des fichiers multi-centres (local)...")
    fichiers_disponibles = {}
    for centre, jours in CENTRES_EVT.items():
        fichiers_disponibles[centre] = {}
        for jour_str, (prefixe, nom_exact) in jours.items():
            dossier = os.path.join("data", "ionex", "2023", jour_str)
            chemin  = os.path.join(dossier, nom_exact)
            if os.path.exists(chemin) and os.path.getsize(chemin) > 0:
                fichiers_disponibles[centre][jour_str] = chemin
                print(f"    ✓ {centre} j{jour_str} : {nom_exact}")
            else:
                trouve = None
                if os.path.exists(dossier):
                    for f in os.listdir(dossier):
                        if f.upper().startswith(prefixe.upper()):
                            trouve = os.path.join(dossier, f)
                            break
                if trouve:
                    fichiers_disponibles[centre][jour_str] = trouve
                    print(f"    ✓ {centre} j{jour_str} : {os.path.basename(trouve)}")
                else:
                    print(f"    ✗ {centre} j{jour_str} : introuvable dans {dossier}")
    return fichiers_disponibles

def aligner_13_epoques(tec):
    """
    Ramène un jeu de cartes journalier à 13 époques communes (pas de 2 h).
    Le pas de sous-échantillonnage est déduit du nombre de cartes :
    25 cartes -> pas 2 ; 49 cartes -> pas 4 ; 13 cartes -> pas 1.
    """
    n = tec.shape[0]
    if (n - 1) % 12 != 0:
        raise ValueError(f"Nombre de cartes inattendu : {n}")
    tec13 = tec[0::(n - 1) // 12]
    assert tec13.shape[0] == 13
    return tec13

def analyser_multi_centres(fichiers_disponibles):
    print("\n>>> Analyse multi-centres sur l'événement du 27-28 février 2023...")
    resultats = {}
    for centre, jours in fichiers_disponibles.items():
        resultats[centre] = {}
        for jour_str, chemin in jours.items():
            try:
                data = lire_ionex(chemin)
                # Alignement sur les 13 époques communes (0 h, 2 h, ..., 24 h),
                # conformément à la section 3.4 du mémoire :
                #   CODE (25 cartes, 1 h)   -> une carte sur 2
                #   CAS  (49 cartes, 30 min) -> une carte sur 4
                #   JPL, ESA, UPC (13 cartes, 2 h) -> inchangés
                tec  = aligner_13_epoques(data['tec'])
                tec_glob = float(np.nanmean(tec))
                bandes_res = {}
                for nom, lat_min, lat_max in BANDES:
                    idx_lat = indices_lat(lat_min, lat_max)
                    tec_b   = tec[:, idx_lat, :]
                    bandes_res[nom] = float(np.nanmean(tec_b))
                resultats[centre][jour_str] = {
                    'tec_global': tec_glob, 'bandes': bandes_res,
                    'n_maps': data['n_maps'], 'interval': data['interval'],
                }
            except Exception as e:
                print(f"    Erreur lecture {centre} j{jour_str} : {e}")
    return resultats

def afficher_tableau_multi_centres(resultats):
    centres_dispo = [c for c in CENTRES_EVT if c in resultats and resultats[c]]
    for jour_str, label in [("058", "27 février 2023"), ("059", "28 février 2023")]:
        print(f"\n{'=' * 80}")
        print(f"Jour {jour_str} — {label}")
        print(f"{'=' * 80}")
        header = f"{'Bande / Région':<25}"
        for centre in centres_dispo:
            if jour_str in resultats.get(centre, {}):
                header += f"  {centre:>10}"
        print(header)
        print("-" * 80)
        ligne = f"{'TEC moyen global':<25}"
        for centre in centres_dispo:
            if jour_str in resultats.get(centre, {}):
                val = resultats[centre][jour_str]['tec_global']
                ligne += f"  {val:>10.2f}"
            else:
                ligne += f"  {'N/D':>10}"
        print(ligne + " TECU")
        print("-" * 80)
        for nom, _, _ in BANDES:
            ligne = f"{nom:<25}"
            for centre in centres_dispo:
                if jour_str in resultats.get(centre, {}):
                    val = resultats[centre][jour_str]['bandes'].get(nom, np.nan)
                    ligne += f"  {val:>10.2f}"
                else:
                    ligne += f"  {'N/D':>10}"
            print(ligne + " TECU")
        print(f"{'=' * 80}")

def fig_multi_centres(resultats, nom_fich):
    centres_dispo = [c for c in CENTRES_EVT if c in resultats and resultats[c]]
    if len(centres_dispo) < 2:
        print("    ⚠️  Moins de 2 centres disponibles — figure multi-centres ignorée")
        return
    noms_bandes = [b[0] for b in BANDES]
    x     = np.arange(len(noms_bandes))
    width = 0.8 / len(centres_dispo)
    couleurs = ["#1565C0","#E53935","#2E7D32","#E65100","#6A1B9A"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, (jour_str, label) in zip(axes, [("058","27 fév. 2023"),("059","28 fév. 2023")]):
        for k, (centre, couleur) in enumerate(zip(centres_dispo, couleurs)):
            if jour_str not in resultats.get(centre, {}):
                continue
            vals = [resultats[centre][jour_str]['bandes'].get(nom, np.nan)
                    for nom in noms_bandes]
            offset = (k - len(centres_dispo)/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=centre, color=couleur, alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(noms_bandes, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel("TEC moyen (TECU)")
        ax.set_title(f"TEC moyen par bande — {label}", fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.25)
    fig.suptitle("Comparaison multi-centres — TEC moyen par bande de latitude\n"
                 "Événement géomagnétique 27-28 février 2023",
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

def fig_cartes_evenement(tec_code, tec_jpl, nom_fich):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (titre, idx) in zip(axes,
        [("Jour 058 — 27 février 2023", 57),
         ("Jour 059 — 28 février 2023", 58)]):
        code_a = tec_code[idx][0::2]
        jpl_a  = tec_jpl[idx]
        diff   = np.nanmean(code_a - jpl_a, axis=0)
        vmax   = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
        im = ax.pcolormesh(LON, LAT, diff, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax, shrink=0.75, label='TEC CODE−JPL (TECU)')
        ax.set_title(titre, fontweight='bold', fontsize=11)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.grid(True, alpha=0.2)
    fig.suptitle("Cartes de différence TEC CODE − JPL\n"
                 "Événement géomagnétique — 27-28 février 2023",
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

def analyser_evenement_fevrier(tec_code, tec_jpl, jours_communs):
    print("\n>>> Analyse CODE vs JPL par bandes — 27-28 février 2023...")
    print("\n" + "=" * 75)
    print(f"{'Bande':<25} {'Lat':>10}  "
          f"{'MBE j058':>10} {'RMSE j058':>10}  "
          f"{'MBE j059':>10} {'RMSE j059':>10}")
    print("-" * 75)
    resultats_evt = []
    for nom, lat_min, lat_max in BANDES:
        ligne = [nom, lat_min, lat_max]
        for jour in [58, 59]:
            if jour not in list(jours_communs):
                ligne += [np.nan, np.nan]; continue
            idx     = int(jour) - 1
            code_a, jpl_a = aligner_code_jpl(tec_code[idx], tec_jpl[idx])
            idx_lat = indices_lat(lat_min, lat_max)
            diff    = (code_a - jpl_a)[:, idx_lat, :]
            ligne.append(float(np.nanmean(diff)))
            ligne.append(float(np.sqrt(np.nanmean(diff ** 2))))
        lat_str = f"{lat_min}° à {lat_max}°"
        print(f"{nom:<25} {lat_str:>10}  "
              f"{ligne[3]:>+10.3f} {ligne[4]:>10.3f}  "
              f"{ligne[5]:>+10.3f} {ligne[6]:>10.3f}")
        resultats_evt.append(ligne)
    print("=" * 75)
    return resultats_evt

# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    print("=" * 60)
    print("T5 - Analyse complémentaire CODE vs JPL 2023 (méthode unique)")
    print("=" * 60)

    tec_code, tec_jpl, jours_communs, mbe = charger_base()

    mbe_pct, tec_moy = calculer_mbe_pct(tec_code, tec_jpl, jours_communs)
    tableau_mensuel_complet(jours_communs, mbe, mbe_pct, tec_moy)

    print("\n>>> Génération figures MBE%...")
    fig_mbe_pct(jours_communs, mbe_pct, "serie_MBE_pct.png")
    fig_comparaison_mbe_absolu_pct(jours_communs, mbe, mbe_pct,
                                   "comparaison_MBE_absolu_pct.png")

    stats_bandes     = calculer_stats_par_bandes(tec_code, tec_jpl, jours_communs)
    resultats_bandes = tableau_stats_bandes_annuel(stats_bandes)

    print("\n>>> Génération figures bandes latitude...")
    fig_mbe_par_bandes(jours_communs, stats_bandes, "mbe_par_bandes_lat.png")
    fig_barres_bandes(resultats_bandes, "barres_mbe_rmse_bandes.png")

    analyser_evenement_fevrier(tec_code, tec_jpl, jours_communs)

    print("\n>>> Génération cartes 27-28 février...")
    fig_cartes_evenement(tec_code, tec_jpl, "cartes_evenement_fevrier.png")

    fichiers_dispo = telecharger_centres_fevrier()
    resultats_mc   = analyser_multi_centres(fichiers_dispo)
    afficher_tableau_multi_centres(resultats_mc)

    print("\n>>> Génération figure multi-centres...")
    fig_multi_centres(resultats_mc, "multi_centres_evenement_fevrier.png")

    np.savez(os.path.join(DOSSIER_STATS, "stats_complementaires.npz"),
             mbe_pct=mbe_pct, tec_moy=tec_moy, jours=jours_communs)

    print("\n" + "=" * 60)
    print("RÉSUMÉ T5")
    print("=" * 60)
    print(f"  MBE% moyen annuel    : {np.nanmean(mbe_pct):+.2f}%  (source unique : série mbe_pct)")
    print(f"  Bande la + affectée  : {max(resultats_bandes, key=lambda x: abs(x[3]))[0]}")
    centres_ok = [c for c in fichiers_dispo if fichiers_dispo[c]]
    print(f"  Centres disponibles  : {', '.join(centres_ok)}")
    print(f"  Figures générées     : 6")
    print("=" * 60)
    print("\n✅ T5 terminée.")

if __name__ == "__main__":
    main()