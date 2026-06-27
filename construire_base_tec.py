"""
=============================================================================
T3 - Constitution de la base de données TEC 2023 - Version 2
Gère les deux conventions de nommage JPL :
  - Ancienne : jplg{JJJ}0.23i
  - Nouvelle : JPL0OPSFIN_{ANNEE}{JJJ}0000_01D_02H_GIM.INX
=============================================================================
Auteur  : Bassirou Kane
Mémoire : Master 2 Informatique - Génie Logiciel
"""

import os
import numpy as np
from tqdm import tqdm
from parseur_ionex import lire_ionex

# =============================================================================
# CONFIGURATION
# =============================================================================

ANNEE         = 2023
DOSSIER_IONEX = "data/ionex/2023"
DOSSIER_BASE  = "data/base_tec"
os.makedirs(DOSSIER_BASE, exist_ok=True)

N_LAT       = 71
N_LON       = 73
N_MAPS_CODE = 25
N_MAPS_JPL  = 13

# =============================================================================
# FONCTIONS DE NOMMAGE
# =============================================================================

def chemin_code(jour):
    """Chemin du fichier CODE — une seule convention."""
    nom = f"COD0OPSFIN_{ANNEE}{jour:03d}0000_01D_01H_GIM.INX"
    return os.path.join(DOSSIER_IONEX, f"{jour:03d}", nom)

def chemin_jpl(jour):
    """
    Chemin du fichier JPL — essaie les deux conventions dans l'ordre :
    1. Ancienne : jplg{JJJ}0.23i
    2. Nouvelle : JPL0OPSFIN_{ANNEE}{JJJ}0000_01D_02H_GIM.INX
    Retourne le premier chemin trouvé sur le disque, ou None si absent.
    """
    annee_2c = str(ANNEE)[2:]
    dossier  = os.path.join(DOSSIER_IONEX, f"{jour:03d}")

    candidats = [
        # Ancienne convention (minuscules)
        os.path.join(dossier, f"jplg{jour:03d}0.{annee_2c}i"),
        # Nouvelle convention (majuscules ou minuscules)
        os.path.join(dossier, f"JPL0OPSFIN_{ANNEE}{jour:03d}0000_01D_02H_GIM.INX"),
        os.path.join(dossier, f"jpl0opsfin_{ANNEE}{jour:03d}0000_01d_02h_gim.inx"),
    ]

    for chemin in candidats:
        if os.path.exists(chemin) and os.path.getsize(chemin) > 0:
            return chemin

    return None   # Fichier absent pour ce jour

# =============================================================================
# CONSTRUCTION DE LA BASE
# =============================================================================

def construire_base():
    print("=" * 60)
    print("T3 v2 - Constitution de la base de données TEC 2023")
    print("       (gestion des deux conventions JPL)")
    print("=" * 60)

    tec_code = np.full((365, N_MAPS_CODE, N_LAT, N_LON), np.nan, dtype=np.float32)
    tec_jpl  = np.full((365, N_MAPS_JPL,  N_LAT, N_LON), np.nan, dtype=np.float32)

    jours_code   = []
    jours_jpl    = []
    erreurs_code = []
    erreurs_jpl  = []

    # ── Lecture CODE ──────────────────────────────────────────────────────
    print("\n>>> Lecture des fichiers CODE (365 jours)...")
    for jour in tqdm(range(1, 366), desc="CODE", unit="jour"):
        chemin = chemin_code(jour)
        if not os.path.exists(chemin):
            continue
        try:
            data = lire_ionex(chemin)
            tec_code[jour-1] = data['tec']
            jours_code.append(jour)
        except Exception as e:
            erreurs_code.append((jour, str(e)))

    # ── Lecture JPL ───────────────────────────────────────────────────────
    print("\n>>> Lecture des fichiers JPL (toutes conventions)...")
    convention_utilisee = {'ancienne': 0, 'nouvelle': 0}

    for jour in tqdm(range(1, 366), desc="JPL ", unit="jour"):
        chemin = chemin_jpl(jour)
        if chemin is None:
            continue
        try:
            data = lire_ionex(chemin)
            tec_jpl[jour-1] = data['tec']
            jours_jpl.append(jour)
            # Compter quelle convention a été utilisée
            if 'jplg' in os.path.basename(chemin).lower():
                convention_utilisee['ancienne'] += 1
            else:
                convention_utilisee['nouvelle'] += 1
        except Exception as e:
            erreurs_jpl.append((jour, str(e)))

    # ── Jours communs ─────────────────────────────────────────────────────
    jours_communs = sorted(set(jours_code) & set(jours_jpl))

    # ── Sauvegarde ────────────────────────────────────────────────────────
    print("\n>>> Sauvegarde de la base de données...")
    np.savez_compressed(os.path.join(DOSSIER_BASE, "code_tec_2023.npz"), tec=tec_code)
    np.savez_compressed(os.path.join(DOSSIER_BASE, "jpl_tec_2023.npz"),  tec=tec_jpl)
    np.save(os.path.join(DOSSIER_BASE, "jours_code.npy"),    np.array(jours_code))
    np.save(os.path.join(DOSSIER_BASE, "jours_jpl.npy"),     np.array(jours_jpl))
    np.save(os.path.join(DOSSIER_BASE, "jours_communs.npy"), np.array(jours_communs))

    # ── Résumé ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RÉSUMÉ DE LA BASE DE DONNÉES TEC 2023")
    print("=" * 60)
    print(f"  Jours CODE lus         : {len(jours_code)} / 365")
    print(f"  Jours JPL  lus         : {len(jours_jpl)} / 365")
    print(f"    dont convention ancien (jplg...) : {convention_utilisee['ancienne']}")
    print(f"    dont convention nouveau (JPL0...) : {convention_utilisee['nouvelle']}")
    print(f"  Jours communs          : {len(jours_communs)}")
    print(f"  Erreurs CODE           : {len(erreurs_code)}")
    print(f"  Erreurs JPL            : {len(erreurs_jpl)}")
    print(f"  Taille base CODE       : {tec_code.nbytes / 1e6:.1f} Mo")
    print(f"  Taille base JPL        : {tec_jpl.nbytes  / 1e6:.1f} Mo")
    print("=" * 60)

    if erreurs_code:
        print(f"\nErreurs CODE :")
        for j, e in erreurs_code[:5]:
            print(f"  Jour {j:03d} : {e}")
    if erreurs_jpl:
        print(f"\nErreurs JPL :")
        for j, e in erreurs_jpl[:5]:
            print(f"  Jour {j:03d} : {e}")

    print(f"\n✅ Base constituée. {len(jours_communs)} jours communs disponibles.")
    return jours_code, jours_jpl, jours_communs

# =============================================================================
# VÉRIFICATION
# =============================================================================

def verifier_base():
    print("\n>>> Vérification de la base...")
    tec_code      = np.load(os.path.join(DOSSIER_BASE, "code_tec_2023.npz"))['tec']
    tec_jpl       = np.load(os.path.join(DOSSIER_BASE, "jpl_tec_2023.npz"))['tec']
    jours_communs = np.load(os.path.join(DOSSIER_BASE, "jours_communs.npy"))

    print(f"  Shape CODE : {tec_code.shape}")
    print(f"  Shape JPL  : {tec_jpl.shape}")
    print(f"  Jours communs : {len(jours_communs)}")

    idx = jours_communs[0] - 1
    tec_c_aligne = tec_code[idx][::2]
    tec_j        = tec_jpl[idx]

    print(f"\n  Vérification jour {jours_communs[0]:03d} :")
    print(f"    TEC CODE moy : {np.nanmean(tec_code[idx]):.2f} TECU")
    print(f"    TEC JPL  moy : {np.nanmean(tec_j):.2f} TECU")
    print(f"    Écart moyen  : {np.nanmean(tec_c_aligne - tec_j):.2f} TECU")
    print("\n✅ Vérification réussie.")

if __name__ == "__main__":
    construire_base()
    verifier_base()