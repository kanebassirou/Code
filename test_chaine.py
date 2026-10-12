"""
=============================================================================
Protocole de tests de la chaîne de traitement (Annexe B du mémoire)
=============================================================================
Auteur  : Bassirou Kane
Mémoire : Master 2 Informatique - Génie Logiciel

Ce fichier regroupe, sous forme automatisée, les contrôles décrits au
chapitre 4 (section 4.4) et en Annexe B :
  TU : tests unitaires des fonctions de calcul (données synthétiques)
  TI : tests d'intégrité des données réelles (fichiers IONEX, base TEC)
  TR : test de reproduction des valeurs publiées dans le mémoire

UTILISATION (depuis la racine du projet, là où se trouve le dossier data/)
    pip install pytest
    python -m pytest test_chaine.py -v

Les tests TI et TR sont ignorés (« skipped ») si les données sont absentes.

IMPACT SUR LES AUTRES MODULES
    Aucun : les modules sont seulement importés ; aucun fichier n'est modifié.
=============================================================================
"""

import os
import sys
import glob
from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy.stats import pearsonr

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)

from parseur_ionex import extraire_flottants, lire_ionex          # noqa: E402
from calculer_statistiques import calculer_stats_journalieres       # noqa: E402
from analyse_distribution_ecarts import Accumulateur, interpoler_1h # noqa: E402
from diagnostic_journees_extremes import ecart_temporel             # noqa: E402
from analyse_complementaire import aligner_13_epoques               # noqa: E402

IONEX = os.path.join(ICI, "data", "ionex", "2023")
BASE = os.path.join(ICI, "data", "base_tec")
donnees_ionex = pytest.mark.skipif(not os.path.isdir(IONEX), reason="fichiers IONEX absents")
donnees_base = pytest.mark.skipif(not os.path.isdir(BASE), reason="base TEC absente")


# =============================================================================
# TU – Décodage
# =============================================================================

def test_TU01_coordonnees_collees():
    """Les coordonnées de rangée collées sans séparateur sont correctement extraites."""
    assert extraire_flottants("    87.5-180.0 180.0   5.0 450.0") == [87.5, -180.0, 180.0, 5.0, 450.0]
    assert extraire_flottants("   -87.5-180.0 180.0   5.0 450.0") == [-87.5, -180.0, 180.0, 5.0, 450.0]


@donnees_ionex
def test_TI02_lecture_code_jour001():
    d = lire_ionex(os.path.join(IONEX, "001", "COD0OPSFIN_20230010000_01D_01H_GIM.INX"))
    assert d["n_maps"] == 25 and d["interval"] == 3600
    assert d["tec"].shape == (25, 71, 73)
    assert not np.isnan(d["tec"]).any()
    assert d["lat"][0] == 87.5 and d["lat"][-1] == -87.5
    assert d["lon"][0] == -180.0 and d["lon"][-1] == 180.0
    attendu = [datetime(2023, 1, 1) + timedelta(hours=h) for h in range(25)]
    assert d["epochs"] == attendu
    # Première valeur du fichier : 104 en unités de 0,1 TECU (EXPONENT = -1)
    assert d["tec"][0, 0, 0] == pytest.approx(10.4)


@donnees_ionex
def test_TI03_lecture_jpl_nom_court():
    d = lire_ionex(os.path.join(IONEX, "001", "jplg0010.23i"))
    assert d["n_maps"] == 13 and d["interval"] == 7200
    assert d["tec"].shape == (13, 71, 73) and not np.isnan(d["tec"]).any()


@donnees_ionex
def test_TI04_lecture_jpl_nom_long():
    fichiers = sorted(glob.glob(os.path.join(IONEX, "*", "jpl0opsfin_*.inx")))
    assert fichiers, "aucun fichier JPL au nom long"
    d = lire_ionex(fichiers[0])
    assert d["n_maps"] == 13 and d["tec"].shape == (13, 71, 73)
    assert not np.isnan(d["tec"]).any()


@donnees_ionex
@pytest.mark.parametrize("fichier,n", [
    ("ESA0OPSFIN_20230580000_01D_02H_GIM.INX", 13),
    ("upcg0580.23i", 13),
    ("CAS0OPSFIN_20230580000_01D_30M_GIM.INX", 49),
])
def test_TI05_lecture_autres_centres(fichier, n):
    d = lire_ionex(os.path.join(IONEX, "058", fichier))
    assert d["n_maps"] == n and d["tec"].shape == (n, 71, 73)
    assert not np.isnan(d["tec"]).any()


# =============================================================================
# TI – Base TEC
# =============================================================================

@pytest.fixture(scope="module")
def base():
    code = np.load(os.path.join(BASE, "code_tec_2023.npz"))["tec"]
    jpl = np.load(os.path.join(BASE, "jpl_tec_2023.npz"))["tec"]
    jours = np.load(os.path.join(BASE, "jours_communs.npy"))
    return code, jpl, jours


@donnees_base
def test_TI06_completude_base(base):
    code, jpl, jours = base
    assert code.shape == (365, 25, 71, 73) and jpl.shape == (365, 13, 71, 73)
    assert len(jours) == 365
    assert not np.isnan(code).any() and not np.isnan(jpl).any()


@donnees_base
def test_TI07_vraisemblance_physique(base):
    """Aucune valeur négative ni aberrante (> 250 TECU) : un facteur d'échelle mal
    appliqué (EXPONENT) produirait des valeurs 10 fois trop grandes ou trop petites."""
    code, jpl, _ = base
    for tec in (code, jpl):
        assert tec.min() >= 0.0 and tec.max() <= 250.0
        assert 100.0 <= tec.max()          # ordre de grandeur attendu en 2023 (forte activité solaire)


@donnees_ionex
def test_TI08_alignement_epoques():
    """Les époques paires de CODE coïncident exactement avec celles de JPL."""
    c = lire_ionex(os.path.join(IONEX, "001", "COD0OPSFIN_20230010000_01D_01H_GIM.INX"))
    j = lire_ionex(os.path.join(IONEX, "001", "jplg0010.23i"))
    assert c["epochs"][0::2] == j["epochs"]


# =============================================================================
# TU – Calculs
# =============================================================================

def test_TU09_indicateurs_synthetiques():
    """MBE, RMSE et r sur des champs synthétiques de biais et dispersion connus."""
    rng = np.random.default_rng(2023)
    jpl = rng.uniform(5, 60, size=(2, 13, 71, 73))
    bruit = rng.normal(0.0, 3.0, size=jpl.shape)
    code = np.empty((2, 25, 71, 73))
    code[:, 0::2] = jpl - 2.0 + bruit          # biais imposé : -2 TECU ; dispersion : 3 TECU
    code[:, 1::2] = 0.0                         # époques impaires ignorées par l'alignement
    mbe, rmse, r = calculer_stats_journalieres(code, jpl, np.array([1, 2]))
    d = (code[0, 0::2] - jpl[0])
    assert mbe[0] == pytest.approx(d.mean(), abs=1e-12)
    assert rmse[0] == pytest.approx(np.sqrt((d ** 2).mean()), abs=1e-12)
    assert r[0] == pytest.approx(np.corrcoef(code[0, 0::2].ravel(), jpl[0].ravel())[0, 1], abs=1e-12)
    assert mbe[0] == pytest.approx(-2.0, abs=0.05)
    assert rmse[0] == pytest.approx(np.sqrt(2.0 ** 2 + 3.0 ** 2), abs=0.05)


@pytest.mark.parametrize("methode", ["linear", "quadratic", "cubic_spline"])
def test_TU10_interpolation_passe_par_les_noeuds(methode):
    rng = np.random.default_rng(1)
    cartes = rng.uniform(0, 80, size=(13, 71, 73))
    res = interpoler_1h(cartes, methode)
    assert res.shape == (25, 71, 73)
    assert np.allclose(res[0::2], cartes, atol=1e-9)


@pytest.mark.parametrize("methode,degre", [("linear", 1), ("quadratic", 2), ("cubic_spline", 3)])
def test_TU11_interpolation_exacte_sur_polynome(methode, degre):
    h2 = np.arange(0, 25, 2, dtype=float)
    h1 = np.arange(0, 25, 1, dtype=float)
    f = lambda t: 10 + 0.5 * t ** degre / 24 ** (degre - 1)
    cartes = np.broadcast_to(f(h2)[:, None, None], (13, 2, 3)).copy()
    res = interpoler_1h(cartes, methode)
    assert np.allclose(res[:, 0, 0], f(h1), atol=1e-9)


def test_TU12_accumulateur_histogramme():
    """Moyenne, écart-type et centiles de l'accumulateur comparés à NumPy."""
    rng = np.random.default_rng(7)
    x = rng.standard_t(df=5, size=1_000_000) * 3.0 - 2.0
    acc = Accumulateur()
    for morceau in np.array_split(x, 10):
        acc.ajouter(morceau)
    assert acc.n == x.size
    assert acc.moyenne() == pytest.approx(x.mean(), abs=1e-9)
    assert acc.ecart_type() == pytest.approx(x.std(), abs=1e-9)
    for q in (1, 5, 50, 95, 99):
        assert acc.centile(q) == pytest.approx(np.percentile(x, q), abs=0.011)
    for q in (50, 95, 99):
        assert acc.centile_abs(q) == pytest.approx(np.percentile(np.abs(x), q), abs=0.011)
    assert acc.part_au_dela(5.0) == pytest.approx(100 * (np.abs(x) >= 5.0).mean(), abs=0.01)


def test_TU13_coherence_temporelle_nulle_si_evolution_lineaire():
    jours = np.arange(5, dtype=float)[:, None, None, None]
    produit = 20.0 + 1.5 * jours + np.zeros((5, 13, 4, 6))
    assert np.allclose(ecart_temporel(produit, 2), 0.0)


def test_TU14_alignement_13_epoques():
    for n, pas in [(25, 2), (49, 4), (13, 1)]:
        cartes = np.arange(n, dtype=float)[:, None, None] * np.ones((n, 2, 2))
        res = aligner_13_epoques(cartes)
        assert res.shape[0] == 13 and np.allclose(res[:, 0, 0], np.arange(0, n, pas))
    with pytest.raises(ValueError):
        aligner_13_epoques(np.zeros((24, 2, 2)))


# =============================================================================
# TR – Reproduction des valeurs publiées
# =============================================================================

@donnees_base
def test_TR15_reproduction_valeurs_publiees(base):
    code, jpl, jours = base
    mbe, rmse, r = calculer_stats_journalieres(code.astype(float), jpl.astype(float), jours)
    assert round(float(np.mean(mbe)), 3) == -1.947
    assert round(float(np.mean(rmse)), 3) == 3.908
    assert round(float(np.mean(r)), 3) == 0.987
    assert int(np.argmax(rmse)) + 1 == 29          # maximum annuel le 29 janvier
    assert (mbe < 0).all()                          # biais négatif les 365 jours
