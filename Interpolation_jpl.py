"""
=============================================================================
T9 - Sensibilité de l'alignement temporel CODE/JPL
Suite à la proposition de M. Fleury (26/07/2026) :
Au lieu de sous-échantillonner CODE (25→13 cartes/j) pour l'aligner sur JPL,
on interpole JPL (13 cartes/j, 2h) vers une résolution horaire (25 cartes/j)
pour l'aligner sur CODE, en testant plusieurs méthodes d'interpolation.

Objectif : vérifier si le MBE et la RMSE obtenus par cette approche inverse
sont cohérents avec ceux obtenus par sous-échantillonnage de CODE (T4/T5),
et déterminer la meilleure méthode d'interpolation.
=============================================================================
Auteur  : Bassirou Kane
Mémoire : Master 2 Informatique - Génie Logiciel

PRÉREQUIS :
    pip install scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
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

# Heures des cartes CODE (25 cartes, 0h à 24h par pas de 1h)
HEURES_CODE = np.arange(0, 25, 1)     # [0, 1, 2, ..., 24]
# Heures des cartes JPL (13 cartes, 0h à 24h par pas de 2h)
HEURES_JPL  = np.arange(0, 25, 2)     # [0, 2, 4, ..., 24]

METHODES = ["linear", "quadratic", "cubic_spline"]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.titlesize': 13, 'figure.dpi': 150})

# =============================================================================
# CHARGEMENT
# =============================================================================

def charger_base():
    print(">>> Chargement de la base de données...")
    tec_code = np.load(os.path.join(DOSSIER_BASE, "code_tec_2023.npz"))['tec']
    tec_jpl  = np.load(os.path.join(DOSSIER_BASE, "jpl_tec_2023.npz"))['tec']
    jours_communs = np.load(os.path.join(DOSSIER_BASE, "jours_communs.npy"))
    print(f"    CODE : {tec_code.shape}  |  JPL : {tec_jpl.shape}")
    return tec_code, tec_jpl, jours_communs

# =============================================================================
# INTERPOLATION TEMPORELLE DE JPL VERS RÉSOLUTION HORAIRE
# =============================================================================

def interpoler_jpl_1h(jpl_jour, methode):
    """
    Interpole une journée JPL (13 cartes à 2h) vers 25 cartes horaires.
    jpl_jour : tableau (13, 71, 73)
    methode  : "linear", "quadratic" ou "cubic_spline"
    Retourne : tableau (25, 71, 73) interpolé
    """
    n_lat, n_lon = jpl_jour.shape[1], jpl_jour.shape[2]
    jpl_interp = np.full((25, n_lat, n_lon), np.nan)

    # Interpolation point par point de la grille (vectorisé par latitude/longitude)
    # Reshape pour traiter toutes les cellules de la grille en une fois
    jpl_flat = jpl_jour.reshape(13, -1)   # (13, 71*73)

    if np.all(np.isnan(jpl_flat)):
        return jpl_interp

    if methode == "linear":
        f = interp1d(HEURES_JPL, jpl_flat, axis=0, kind='linear',
                     bounds_error=False, fill_value="extrapolate")
        jpl_interp_flat = f(HEURES_CODE)
    elif methode == "quadratic":
        f = interp1d(HEURES_JPL, jpl_flat, axis=0, kind='quadratic',
                     bounds_error=False, fill_value="extrapolate")
        jpl_interp_flat = f(HEURES_CODE)
    elif methode == "cubic_spline":
        cs = CubicSpline(HEURES_JPL, jpl_flat, axis=0, extrapolate=True)
        jpl_interp_flat = cs(HEURES_CODE)
    else:
        raise ValueError(f"Méthode inconnue : {methode}")

    jpl_interp = jpl_interp_flat.reshape(25, n_lat, n_lon)
    return jpl_interp

# =============================================================================
# CALCUL DES STATISTIQUES POUR CHAQUE MÉTHODE
# =============================================================================

def calculer_stats_par_methode(tec_code, tec_jpl, jours_communs):
    """
    Pour chaque méthode d'interpolation, calcule MBE et RMSE journaliers
    en comparant CODE (25 cartes, résolution native 1h) à JPL interpolé
    (25 cartes, interpolées depuis 13 cartes à 2h).
    """
    print("\n>>> Calcul des statistiques pour chaque méthode d'interpolation...")

    resultats = {m: {'mbe': np.full(len(jours_communs), np.nan),
                     'rmse': np.full(len(jours_communs), np.nan)}
                 for m in METHODES}

    for i, jour in enumerate(jours_communs):
        if i % 50 == 0:
            print(f"    Jour {jour:03d} ({i+1}/{len(jours_communs)})...")

        idx = int(jour) - 1
        code_jour = tec_code[idx]   # (25, 71, 73) résolution native
        jpl_jour  = tec_jpl[idx]    # (13, 71, 73) résolution native

        for methode in METHODES:
            jpl_interp = interpoler_jpl_1h(jpl_jour, methode)
            diff = code_jour - jpl_interp
            resultats[methode]['mbe'][i]  = float(np.nanmean(diff))
            resultats[methode]['rmse'][i] = float(np.sqrt(np.nanmean(diff ** 2)))

    return resultats

# =============================================================================
# COMPARAISON AVEC LA MÉTHODE DE RÉFÉRENCE (sous-échantillonnage CODE)
# =============================================================================

def calculer_stats_reference(tec_code, tec_jpl, jours_communs):
    """
    Méthode de référence utilisée en T4/T5 : sous-échantillonnage de CODE
    sur les époques paires pour aligner sur JPL (13 cartes communes).
    """
    print("\n>>> Calcul de la méthode de référence (sous-échantillonnage CODE)...")
    mbe_ref  = np.full(len(jours_communs), np.nan)
    rmse_ref = np.full(len(jours_communs), np.nan)

    for i, jour in enumerate(jours_communs):
        idx = int(jour) - 1
        code_a = tec_code[idx][0::2]   # sous-échantillonnage 25 -> 13
        jpl_a  = tec_jpl[idx]
        diff = code_a - jpl_a
        mbe_ref[i]  = float(np.nanmean(diff))
        rmse_ref[i] = float(np.sqrt(np.nanmean(diff ** 2)))

    return mbe_ref, rmse_ref

# =============================================================================
# TABLEAU COMPARATIF
# =============================================================================

def tableau_comparatif(resultats, mbe_ref, rmse_ref):
    print("\n" + "=" * 75)
    print(f"{'Méthode':<20} {'MBE moyen':>12} {'RMSE moyen':>12} "
          f"{'Écart MBE/réf':>15} {'Écart RMSE/réf':>15}")
    print("-" * 75)

    mbe_ref_moy  = np.nanmean(mbe_ref)
    rmse_ref_moy = np.nanmean(rmse_ref)
    print(f"{'Référence (T4/T5)':<20} {mbe_ref_moy:>+12.3f} {rmse_ref_moy:>12.3f} "
          f"{'—':>15} {'—':>15}")

    resultats_tableau = []
    for methode in METHODES:
        mbe_m  = np.nanmean(resultats[methode]['mbe'])
        rmse_m = np.nanmean(resultats[methode]['rmse'])
        ecart_mbe  = mbe_m - mbe_ref_moy
        ecart_rmse = rmse_m - rmse_ref_moy
        print(f"{methode:<20} {mbe_m:>+12.3f} {rmse_m:>12.3f} "
              f"{ecart_mbe:>+15.3f} {ecart_rmse:>+15.3f}")
        resultats_tableau.append((methode, mbe_m, rmse_m, ecart_mbe, ecart_rmse))

    print("=" * 75)

    # Identifier la meilleure méthode (écart RMSE minimal par rapport à la référence)
    meilleure = min(resultats_tableau, key=lambda x: abs(x[4]))
    print(f"\nMeilleure méthode (écart RMSE minimal vs référence) : {meilleure[0]}")
    print(f"  Écart RMSE : {meilleure[4]:+.3f} TECU")

    return resultats_tableau, meilleure[0]

# =============================================================================
# VALIDATION INTRINSÈQUE PAR RÉTROPROJECTION SUR CODE
# =============================================================================
# Principe : CODE possède déjà de vraies valeurs horaires. On retire
# artificiellement les époques impaires (pour simuler une résolution 2h
# comme JPL), on réinterpole vers 1h avec chaque méthode, puis on compare
# le résultat interpolé aux vraies valeurs CODE retirées. Cela donne une
# mesure OBJECTIVE de la qualité de chaque méthode d'interpolation,
# indépendante du biais CODE/JPL (qui n'intervient pas ici).
# =============================================================================

def valider_interpolation_sur_code(tec_code, jours_communs):
    """
    Pour chaque jour et chaque méthode :
      1. Prend les 25 cartes CODE réelles (vérité connue)
      2. Ne garde que les époques paires (0h,2h,...,24h) -> 13 cartes, comme JPL
      3. Réinterpole vers les 25 époques horaires avec chaque méthode
      4. Compare uniquement aux époques IMPAIRES (1h,3h,...,23h), qui ont
         été retirées puis reconstruites : ce sont les seules valeurs que
         l'interpolation devait "deviner", donc la seule mesure honnête
         de sa qualité.
    Retourne l'erreur de reconstruction (RMSE) par méthode.
    """
    print("\n>>> Validation intrinsèque : reconstruction des époques impaires de CODE...")
    print("    (test de qualité pure de l'interpolation, indépendant du biais CODE/JPL)")

    idx_impairs = np.arange(1, 24, 2)   # époques 1h,3h,...,23h -> 12 valeurs retirées
    erreurs = {m: [] for m in METHODES}

    for i, jour in enumerate(jours_communs):
        if i % 50 == 0:
            print(f"    Jour {jour:03d} ({i+1}/{len(jours_communs)})...")

        idx = int(jour) - 1
        code_verite = tec_code[idx]          # (25, 71, 73) vraies valeurs horaires
        code_pairs  = code_verite[0::2]      # (13, 71, 73) simulation "résolution JPL"

        for methode in METHODES:
            # Réinterpoler vers 1h à partir des seules époques paires
            code_reconstruit = interpoler_jpl_1h(code_pairs, methode)  # (25,71,73)

            # Comparer UNIQUEMENT aux époques impaires (les seules "devinées")
            vrai_impair  = code_verite[idx_impairs]
            recon_impair = code_reconstruit[idx_impairs]
            diff = vrai_impair - recon_impair
            rmse_jour = float(np.sqrt(np.nanmean(diff ** 2)))
            if not np.isnan(rmse_jour):
                erreurs[methode].append(rmse_jour)

    print("\n" + "=" * 60)
    print("RÉSULTAT DE LA VALIDATION INTRINSÈQUE")
    print("(erreur de reconstruction des époques impaires — plus bas = meilleur)")
    print("-" * 60)
    resultats_validation = []
    for methode in METHODES:
        rmse_moyen = np.mean(erreurs[methode])
        rmse_median = np.median(erreurs[methode])
        print(f"  {methode:<15} RMSE reconstruction moyen = {rmse_moyen:.4f} TECU "
              f"(médiane = {rmse_median:.4f})")
        resultats_validation.append((methode, rmse_moyen, rmse_median))
    print("=" * 60)

    meilleure = min(resultats_validation, key=lambda x: x[1])
    print(f"\n✅ Meilleure méthode selon la validation intrinsèque : {meilleure[0]}")
    print(f"   (erreur de reconstruction la plus faible : {meilleure[1]:.4f} TECU)")

    return resultats_validation, meilleure[0]


def fig_validation_intrinseque(resultats_validation, nom_fich):
    """Diagramme en barres de l'erreur de reconstruction par méthode."""
    noms  = [r[0] for r in resultats_validation]
    rmse  = [r[1] for r in resultats_validation]
    couleurs = ["#1565C0", "#2E7D32", "#E65100"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(noms, rmse, color=couleurs, alpha=0.85)
    ax.set_ylabel("RMSE de reconstruction (TECU)")
    ax.set_title("Validation intrinsèque : qualité de reconstruction\n"
                 "des époques horaires retirées (test sur CODE)",
                 fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25)
    for i, v in enumerate(rmse):
        ax.text(i, v + max(rmse)*0.02, f"{v:.4f}", ha='center', fontsize=10)

    # Marquer la meilleure méthode
    idx_min = int(np.argmin(rmse))
    bars[idx_min].set_edgecolor('black')
    bars[idx_min].set_linewidth(2.5)

    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")


def date_du_jour(jour):
    return datetime(ANNEE, 1, 1) + timedelta(days=int(jour) - 1)

def fig_comparaison_methodes(jours_communs, resultats, mbe_ref, rmse_ref, nom_fich):
    """Compare les séries temporelles MBE et RMSE des différentes méthodes."""
    dates = [date_du_jour(j) for j in jours_communs]
    couleurs = {"linear": "#1565C0", "quadratic": "#2E7D32", "cubic_spline": "#E65100"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax1.plot(dates, mbe_ref, color='black', linewidth=1.5, linestyle='--',
             label='Référence (sous-échant. CODE)', alpha=0.7)
    for methode in METHODES:
        ax1.plot(dates, resultats[methode]['mbe'], color=couleurs[methode],
                 linewidth=1.0, alpha=0.8, label=f'JPL interpolé ({methode})')
    ax1.axhline(0, color='gray', linewidth=0.6, linestyle=':')
    ax1.set_ylabel("MBE (TECU)")
    ax1.set_title("Comparaison des méthodes d'alignement temporel — MBE", fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax2.plot(dates, rmse_ref, color='black', linewidth=1.5, linestyle='--',
             label='Référence (sous-échant. CODE)', alpha=0.7)
    for methode in METHODES:
        ax2.plot(dates, resultats[methode]['rmse'], color=couleurs[methode],
                 linewidth=1.0, alpha=0.8, label=f'JPL interpolé ({methode})')
    ax2.set_ylabel("RMSE (TECU)")
    ax2.set_xlabel("Date (2023)")
    ax2.set_title("Comparaison des méthodes d'alignement temporel — RMSE", fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

def fig_barres_methodes(resultats_tableau, mbe_ref_moy, rmse_ref_moy, nom_fich):
    """Diagramme en barres comparant les 3 méthodes à la référence."""
    noms  = ["Référence"] + [r[0] for r in resultats_tableau]
    mbes  = [mbe_ref_moy] + [r[1] for r in resultats_tableau]
    rmses = [rmse_ref_moy] + [r[2] for r in resultats_tableau]
    couleurs = ["#616161", "#1565C0", "#2E7D32", "#E65100"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(noms))

    ax1.bar(x, mbes, color=couleurs, alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(noms, rotation=15, ha='right')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_ylabel("MBE moyen (TECU)")
    ax1.set_title("MBE moyen annuel par méthode", fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.25)
    for i, v in enumerate(mbes):
        ax1.text(i, v - 0.03, f"{v:+.3f}", ha='center', va='top', fontsize=9)

    ax2.bar(x, rmses, color=couleurs, alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(noms, rotation=15, ha='right')
    ax2.set_ylabel("RMSE moyenne (TECU)")
    ax2.set_title("RMSE moyenne annuelle par méthode", fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.25)
    for i, v in enumerate(rmses):
        ax2.text(i, v + 0.03, f"{v:.3f}", ha='center', fontsize=9)

    fig.suptitle("Sensibilité au choix de la méthode d'alignement temporel CODE/JPL",
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    print("=" * 60)
    print("T9 - Sensibilité de l'alignement temporel CODE/JPL")
    print("     Proposition de M. Fleury (26/07/2026)")
    print("=" * 60)

    tec_code, tec_jpl, jours_communs = charger_base()

    # Méthode de référence (T4/T5)
    mbe_ref, rmse_ref = calculer_stats_reference(tec_code, tec_jpl, jours_communs)

    # Nouvelles méthodes par interpolation de JPL
    resultats = calculer_stats_par_methode(tec_code, tec_jpl, jours_communs)

    # Tableau comparatif (cohérence avec la référence)
    resultats_tableau, meilleure_methode_coherence = tableau_comparatif(resultats, mbe_ref, rmse_ref)

    # Validation intrinsèque (qualité pure d'interpolation, indépendante du biais CODE/JPL)
    resultats_validation, meilleure_methode_qualite = valider_interpolation_sur_code(tec_code, jours_communs)

    # Figures
    print("\n>>> Génération des figures...")
    fig_comparaison_methodes(jours_communs, resultats, mbe_ref, rmse_ref,
                             "comparaison_methodes_interpolation.png")
    fig_barres_methodes(resultats_tableau, np.nanmean(mbe_ref), np.nanmean(rmse_ref),
                        "barres_methodes_interpolation.png")
    fig_validation_intrinseque(resultats_validation, "validation_intrinseque_interpolation.png")

    # Sauvegarde
    np.savez(os.path.join(DOSSIER_STATS, "stats_interpolation_jpl.npz"),
             jours=jours_communs, mbe_ref=mbe_ref, rmse_ref=rmse_ref,
             **{f"mbe_{m}": resultats[m]['mbe'] for m in METHODES},
             **{f"rmse_{m}": resultats[m]['rmse'] for m in METHODES})

    print("\n" + "=" * 60)
    print("RÉSUMÉ T9")
    print("=" * 60)
    print(f"  Meilleure méthode (cohérence avec réf. T4/T5) : {meilleure_methode_coherence}")
    print(f"  Meilleure méthode (qualité intrinsèque)        : {meilleure_methode_qualite}")
    if meilleure_methode_coherence == meilleure_methode_qualite:
        print(f"\n  ✅ Les deux critères convergent vers la même méthode : {meilleure_methode_qualite}")
    else:
        print(f"\n  ⚠️  Les deux critères divergent — à discuter dans le rapport.")
    print(f"\n  Cohérence globale avec la méthode de référence : ", end="")
    ecart_max = max(abs(r[4]) for r in resultats_tableau)
    if ecart_max < 0.1:
        print("EXCELLENTE (écart RMSE < 0.1 TECU)")
    elif ecart_max < 0.3:
        print("BONNE (écart RMSE < 0.3 TECU)")
    else:
        print("À EXAMINER (écart RMSE > 0.3 TECU)")
    print("=" * 60)
    print("\n✅ T9 terminée.")

if __name__ == "__main__":
    main()