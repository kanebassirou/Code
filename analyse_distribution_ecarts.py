"""
=============================================================================
T10 - Distribution des écarts CODE − JPL : histogrammes et fonctions de
      répartition (complément à la RMSE demandé par M. Fleury, 17/08/2026)
=============================================================================
Auteur  : Bassirou Kane
Mémoire : Master 2 Informatique - Génie Logiciel

OBJECTIF
    La RMSE résume l'amplitude des écarts par un seul nombre. Ce module
    décrit la distribution complète des écarts ponctuels ΔTEC = CODE − JPL
    (histogrammes, fonctions de répartition empiriques, centiles), afin de
    quantifier les queues de distribution, pour les quatre méthodes
    d'alignement temporel étudiées en T9 :
        - référence      : sous-échantillonnage de CODE (13 époques / jour)
        - linear         : JPL interpolé linéairement      (25 époques / jour)
        - quadratic      : JPL interpolé (quadratique)     (25 époques / jour)
        - cubic_spline   : JPL interpolé (spline cubique)  (25 époques / jour)
    Il décrit aussi la distribution des erreurs de reconstruction de la
    validation intrinsèque de T9 (époques impaires de CODE retirées puis
    reconstruites).

PRINCIPE DE CALCUL (économe en mémoire)
    Les écarts ne sont pas stockés : pour chaque jour, ils sont accumulés
    dans des histogrammes à pas fin (0,01 TECU). Moyenne et écart-type sont
    calculés à partir de sommes cumulées exactes ; les centiles sont lus
    sur l'histogramme cumulé (précision ± 0,005 TECU). La mémoire utilisée
    reste ainsi indépendante du nombre de jours traités.

PRÉREQUIS
    pip install numpy scipy matplotlib tqdm
    Base TEC construite (data/base_tec/*.npz, voir construire_base_tec.py)

UTILISATION
    python analyse_distribution_ecarts.py

RÉSULTATS
    data/stats/stats_distribution_ecarts.npz
    data/stats/figures/histogrammes_ecarts_alignement.png
    data/stats/figures/cdf_ecarts_alignement.png
    data/stats/figures/distribution_erreurs_reconstruction.png

IMPACT SUR LES AUTRES MODULES
    Aucun : module en lecture seule sur la base TEC ; il ne modifie ni la
    base ni les résultats des tâches T4 à T9.
=============================================================================
"""

import os
from outils_figures import sauver_figure
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

DOSSIER_BASE  = "data/base_tec"
DOSSIER_STATS = "data/stats"
DOSSIER_FIGS  = os.path.join(DOSSIER_STATS, "figures")
os.makedirs(DOSSIER_FIGS, exist_ok=True)

HEURES_CODE = np.arange(0, 25, 1)        # 25 époques horaires
HEURES_JPL  = np.arange(0, 25, 2)        # 13 époques bi-horaires
IDX_IMPAIRS = np.arange(1, 24, 2)        # époques 1 h, 3 h, ..., 23 h

METHODES   = ["linear", "quadratic", "cubic_spline"]
ALIGNEMENTS = ["reference"] + METHODES
LIBELLES = {"reference": "Référence (sous-échantillonnage CODE)",
            "linear": "JPL interpolé – linéaire",
            "quadratic": "JPL interpolé – quadratique",
            "cubic_spline": "JPL interpolé – spline cubique"}
COULEURS = {"reference": "#424242", "linear": "#1565C0",
            "quadratic": "#2E7D32", "cubic_spline": "#E65100"}

# Histogrammes : pas de 0,01 TECU sur [-80, 80] TECU (couvre toutes les valeurs)
PAS   = 0.01
BORNE = 80.0
BORDS = np.arange(-BORNE, BORNE + PAS / 2, PAS)
CENTRES = (BORDS[:-1] + BORDS[1:]) / 2

SEUILS = [2.0, 5.0, 10.0]                 # seuils |ΔTEC| (TECU)
CENTILES = [1, 5, 25, 50, 75, 95, 99]     # centiles de ΔTEC
CENTILES_ABS = [50, 90, 95, 99, 99.9]     # centiles de |ΔTEC|

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.titlesize': 13, 'figure.dpi': 150})

# =============================================================================
# CHARGEMENT
# =============================================================================

def charger_base():
    print(">>> Chargement de la base de données TEC 2023...")
    tec_code = np.load(os.path.join(DOSSIER_BASE, "code_tec_2023.npz"))['tec']
    tec_jpl  = np.load(os.path.join(DOSSIER_BASE, "jpl_tec_2023.npz"))['tec']
    jours    = np.load(os.path.join(DOSSIER_BASE, "jours_communs.npy"))
    print(f"    CODE : {tec_code.shape}  |  JPL : {tec_jpl.shape}  |  {len(jours)} jours communs")
    return tec_code, tec_jpl, jours

# =============================================================================
# INTERPOLATION (identique à interpoler_jpl_1h du module T9)
# =============================================================================

def interpoler_1h(cartes_2h, methode):
    """(13, 71, 73) à pas de 2 h -> (25, 71, 73) à pas de 1 h."""
    plat = cartes_2h.reshape(13, -1).astype(np.float64)
    if methode in ("linear", "quadratic"):
        f = interp1d(HEURES_JPL, plat, axis=0, kind=methode,
                     bounds_error=False, fill_value="extrapolate")
        res = f(HEURES_CODE)
    elif methode == "cubic_spline":
        res = CubicSpline(HEURES_JPL, plat, axis=0, extrapolate=True)(HEURES_CODE)
    else:
        raise ValueError(f"Méthode inconnue : {methode}")
    return res.reshape(25, cartes_2h.shape[1], cartes_2h.shape[2])

# =============================================================================
# ACCUMULATEUR STATISTIQUE
# =============================================================================

class Accumulateur:
    """Histogramme fin + sommes exactes, alimentés jour après jour."""

    def __init__(self):
        self.hist = np.zeros(len(BORDS) - 1, dtype=np.int64)
        self.n = 0
        self.s1 = 0.0
        self.s2 = 0.0
        self.vmin = np.inf
        self.vmax = -np.inf
        self.hors_bornes = 0

    def ajouter(self, valeurs):
        v = np.asarray(valeurs, dtype=np.float64).ravel()
        v = v[~np.isnan(v)]
        if v.size == 0:
            return
        self.n += v.size
        self.s1 += v.sum()
        self.s2 += (v ** 2).sum()
        self.vmin = min(self.vmin, v.min())
        self.vmax = max(self.vmax, v.max())
        self.hors_bornes += int((np.abs(v) >= BORNE).sum())
        self.hist += np.histogram(np.clip(v, -BORNE + PAS / 2, BORNE - PAS / 2), bins=BORDS)[0]

    # --- grandeurs dérivées -------------------------------------------------
    def moyenne(self):
        return self.s1 / self.n

    def ecart_type(self):
        return np.sqrt(self.s2 / self.n - self.moyenne() ** 2)

    def rmse(self):
        return np.sqrt(self.s2 / self.n)

    def centile(self, q):
        cum = np.cumsum(self.hist) / self.n
        return float(CENTRES[np.searchsorted(cum, q / 100.0)])

    def hist_abs(self):
        """Histogramme de |ΔTEC| sur [0, BORNE] au même pas."""
        milieu = len(self.hist) // 2
        return self.hist[milieu:] + self.hist[:milieu][::-1]

    def centile_abs(self, q):
        cum = np.cumsum(self.hist_abs()) / self.n
        return float(np.searchsorted(cum, q / 100.0) * PAS + PAS / 2)

    def part_au_dela(self, seuil):
        h = self.hist_abs()
        k = int(round(seuil / PAS))
        return 100.0 * h[k:].sum() / self.n

    def part_gauss_au_dela(self, k_sigma=3.0):
        """Part observée au-delà de moyenne ± k·σ (loi normale : 0,27 % pour k = 3)."""
        m, s = self.moyenne(), self.ecart_type()
        bas = np.searchsorted(CENTRES, m - k_sigma * s)
        haut = np.searchsorted(CENTRES, m + k_sigma * s)
        return 100.0 * (self.hist[:bas].sum() + self.hist[haut:].sum()) / self.n

    def asymetrie_kurtosis(self):
        """Coefficients d'asymétrie et d'aplatissement (excès), estimés sur l'histogramme."""
        p = self.hist / self.n
        m = (p * CENTRES).sum()
        c2 = (p * (CENTRES - m) ** 2).sum()
        c3 = (p * (CENTRES - m) ** 3).sum()
        c4 = (p * (CENTRES - m) ** 4).sum()
        return c3 / c2 ** 1.5, c4 / c2 ** 2 - 3.0

# =============================================================================
# CALCUL
# =============================================================================

def accumuler(tec_code, tec_jpl, jours, jours_exclus=()):
    """Accumule ΔTEC (4 alignements) et les erreurs de reconstruction (3 méthodes)."""
    ecarts = {a: Accumulateur() for a in ALIGNEMENTS}
    recon  = {m: Accumulateur() for m in METHODES}
    for jour in tqdm(jours, desc="Jours", unit="jour"):
        if int(jour) in jours_exclus:
            continue
        i = int(jour) - 1
        code = tec_code[i].astype(np.float64)        # (25, 71, 73)
        jpl  = tec_jpl[i].astype(np.float64)         # (13, 71, 73)
        # Méthode de référence (T4/T5) : 13 époques communes
        ecarts["reference"].ajouter(code[0::2] - jpl)
        code_pairs = code[0::2]
        for m in METHODES:
            # Alignement par interpolation de JPL vers 25 époques
            ecarts[m].ajouter(code - interpoler_1h(jpl, m))
            # Validation intrinsèque : reconstruction des époques impaires de CODE
            rec = interpoler_1h(code_pairs, m)
            recon[m].ajouter(code[IDX_IMPAIRS] - rec[IDX_IMPAIRS])
    return ecarts, recon

# =============================================================================
# AFFICHAGE
# =============================================================================

def tableau(titre, accs, cles):
    print("\n" + "=" * 100)
    print(titre)
    print("-" * 100)
    entete = f"{'':<16}{'N':>12}{'moy':>8}{'σ':>7}{'RMSE':>7}"
    entete += "".join(f"{'P'+str(q):>7}" for q in CENTILES)
    print(entete)
    for c in cles:
        a = accs[c]
        ligne = f"{c:<16}{a.n:>12d}{a.moyenne():>+8.3f}{a.ecart_type():>7.3f}{a.rmse():>7.3f}"
        ligne += "".join(f"{a.centile(q):>+7.2f}" for q in CENTILES)
        print(ligne)
    print("-" * 100)
    entete = f"{'':<16}" + "".join(f"{'|Δ| P'+str(q):>10}" for q in CENTILES_ABS)
    entete += "".join(f"{'>'+str(int(s))+' TECU':>11}" for s in SEUILS)
    entete += f"{'>3σ':>8}{'asym.':>8}{'kurt.':>8}{'max|Δ|':>9}"
    print(entete)
    for c in cles:
        a = accs[c]
        asym, kurt = a.asymetrie_kurtosis()
        ligne = f"{c:<16}" + "".join(f"{a.centile_abs(q):>10.2f}" for q in CENTILES_ABS)
        ligne += "".join(f"{a.part_au_dela(s):>10.2f}%" for s in SEUILS)
        ligne += f"{a.part_gauss_au_dela(3):>7.2f}%{asym:>+8.2f}{kurt:>+8.2f}"
        ligne += f"{max(abs(a.vmin), abs(a.vmax)):>9.2f}"
        print(ligne)
    print("=" * 100)

# =============================================================================
# FIGURES
# =============================================================================

def _densite(acc, largeur=0.5):
    """Regroupe l'histogramme fin en classes de `largeur` TECU (densité)."""
    k = int(round(largeur / PAS))
    n = (len(acc.hist) // k) * k
    h = acc.hist[:n].reshape(-1, k).sum(axis=1)
    x = BORDS[:n + 1:k]
    return x, h / (acc.n * largeur)

def fig_histogrammes(ecarts, nom_fich):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    for c in ALIGNEMENTS:
        x, d = _densite(ecarts[c])
        kw = dict(color=COULEURS[c], linewidth=1.6 if c == "reference" else 1.1,
                  linestyle="--" if c == "reference" else "-", label=LIBELLES[c])
        ax1.stairs(d, x, **kw)
        ax2.stairs(np.where(d > 0, d, np.nan), x, **kw)
    a = ecarts["reference"]
    xs = np.linspace(-30, 30, 600)
    gauss = np.exp(-0.5 * ((xs - a.moyenne()) / a.ecart_type()) ** 2) / (a.ecart_type() * np.sqrt(2 * np.pi))
    ax2.plot(xs, gauss, color="#9E9E9E", linewidth=1.0, linestyle=":",
             label="Loi normale de même moyenne et écart-type")
    ax1.set_xlim(-15, 15)
    ax1.set_title("Histogramme des écarts (échelle linéaire)", fontweight='bold')
    ax2.set_xlim(-40, 40); ax2.set_yscale("log"); ax2.set_ylim(1e-7, 1)
    ax2.set_title("Histogramme des écarts (échelle logarithmique)", fontweight='bold')
    for ax in (ax1, ax2):
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlabel("ΔTEC = TEC CODE − TEC JPL (TECU)")
        ax.set_ylabel("Densité de probabilité")
        ax.grid(True, alpha=0.25)
    ax2.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Distribution des écarts ponctuels CODE − JPL selon la méthode d'alignement temporel (2023)",
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

def fig_cdf(ecarts, nom_fich):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    for c in ALIGNEMENTS:
        a = ecarts[c]
        kw = dict(color=COULEURS[c], linewidth=1.6 if c == "reference" else 1.1,
                  linestyle="--" if c == "reference" else "-", label=LIBELLES[c])
        cum = np.cumsum(a.hist) / a.n
        ax1.plot(CENTRES, cum, **kw)
        h = a.hist_abs()
        xa = np.arange(len(h)) * PAS + PAS
        ax2.plot(xa, 1 - np.cumsum(h) / a.n, **kw)
    ax1.set_xlim(-15, 15); ax1.set_ylim(0, 1)
    ax1.set_xlabel("ΔTEC = TEC CODE − TEC JPL (TECU)"); ax1.set_ylabel("F(ΔTEC)")
    ax1.set_title("Fonction de répartition empirique de ΔTEC", fontweight='bold')
    ax1.axvline(0, color="black", linewidth=0.6)
    a = ecarts["reference"]
    for q, st in [(50, ":"), (95, "-."), (99, "--")]:
        v = a.centile_abs(q)
        ax2.axvline(v, color="#9E9E9E", linewidth=0.9, linestyle=st)
        ax2.text(v, 2e-6, f" P{q} = {v:.1f}", rotation=90, va="bottom", fontsize=8.5, color="#616161")
    ax2.set_xlim(0, 40); ax2.set_yscale("log"); ax2.set_ylim(1e-6, 1)
    ax2.set_xlabel("|ΔTEC| (TECU)"); ax2.set_ylabel("P(|ΔTEC| > x)")
    ax2.set_title("Fonction de survie de |ΔTEC| (queues de distribution)", fontweight='bold')
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25, which="both")
    ax1.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Fonctions de répartition des écarts CODE − JPL selon la méthode d'alignement temporel (2023)",
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    sauver_figure(fig, os.path.join(DOSSIER_FIGS, nom_fich))
    plt.close(fig)
    print(f"    ✓ {nom_fich}")

def fig_reconstruction(recon, nom_fich):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    for m in METHODES:
        a = recon[m]
        x, d = _densite(a, largeur=0.25)
        ax1.stairs(np.where(d > 0, d, np.nan), x, color=COULEURS[m], linewidth=1.2, label=LIBELLES[m].replace("JPL interpolé – ", "").capitalize())
        h = a.hist_abs()
        xa = np.arange(len(h)) * PAS + PAS
        ax2.plot(xa, 1 - np.cumsum(h) / a.n, color=COULEURS[m], linewidth=1.2,
                 label=LIBELLES[m].replace("JPL interpolé – ", "").capitalize())
    ax1.set_xlim(-20, 20); ax1.set_yscale("log"); ax1.set_ylim(1e-7, 1)
    ax1.set_xlabel("Erreur de reconstruction = CODE réel − CODE reconstruit (TECU)")
    ax1.set_ylabel("Densité de probabilité")
    ax1.set_title("Histogramme des erreurs de reconstruction", fontweight='bold')
    ax2.set_xlim(0, 20); ax2.set_yscale("log"); ax2.set_ylim(1e-6, 1)
    ax2.set_xlabel("|erreur de reconstruction| (TECU)"); ax2.set_ylabel("P(|erreur| > x)")
    ax2.set_title("Fonction de survie des erreurs de reconstruction", fontweight='bold')
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=9)
    fig.suptitle("Validation intrinsèque : distribution des erreurs de reconstruction des époques impaires de CODE",
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
    print("T10 - Distribution des écarts CODE − JPL (histogrammes, CDF)")
    print("=" * 60)
    tec_code, tec_jpl, jours = charger_base()

    ecarts, recon = accumuler(tec_code, tec_jpl, jours)
    tableau("ÉCARTS PONCTUELS ΔTEC = CODE − JPL — 365 jours (TECU)", ecarts, ALIGNEMENTS)
    tableau("ERREURS DE RECONSTRUCTION (validation intrinsèque) — 365 jours (TECU)", recon, METHODES)

    # Sensibilité des queues à la journée atypique de l'année (RMSE maximale)
    rmse_jour = np.array([np.sqrt(np.nanmean((tec_code[int(j)-1][0::2] - tec_jpl[int(j)-1]) ** 2))
                          for j in jours])
    jour_max = int(jours[int(np.nanargmax(rmse_jour))])
    print(f"\n>>> Journée de RMSE maximale : jour {jour_max:03d} — recalcul sans cette journée")
    ecarts_sans, _ = accumuler(tec_code, tec_jpl, jours, jours_exclus=(jour_max,))
    tableau(f"ÉCARTS ΔTEC — 364 jours (jour {jour_max:03d} exclu)", ecarts_sans, ["reference"])
    a, b = ecarts["reference"], ecarts_sans["reference"]
    for s in SEUILS:
        n_tot = a.part_au_dela(s) * a.n / 100
        n_sans = b.part_au_dela(s) * b.n / 100
        print(f"    Part des points |ΔTEC| > {s:>4.0f} TECU provenant du jour {jour_max:03d} : "
              f"{100 * (n_tot - n_sans) / n_tot:5.1f} %")

    print("\n>>> Génération des figures...")
    fig_histogrammes(ecarts, "histogrammes_ecarts_alignement.png")
    fig_cdf(ecarts, "cdf_ecarts_alignement.png")
    fig_reconstruction(recon, "distribution_erreurs_reconstruction.png")

    np.savez_compressed(
        os.path.join(DOSSIER_STATS, "stats_distribution_ecarts.npz"),
        bords=BORDS,
        **{f"hist_{c}": ecarts[c].hist for c in ALIGNEMENTS},
        **{f"hist_recon_{m}": recon[m].hist for m in METHODES},
        **{f"hist_sans_j{jour_max:03d}_reference": ecarts_sans["reference"].hist})
    print(f"\n✅ T10 terminée — résultats dans {os.path.abspath(DOSSIER_STATS)}")

if __name__ == "__main__":
    main()
