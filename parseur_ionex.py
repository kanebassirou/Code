"""
=============================================================================
Parseur IONEX paramétrable - Version 3 (finale)
Gère les coordonnées collées type "87.5-180.0"
=============================================================================
Auteur  : Bassirou Kane
Mémoire : Master 2 Informatique - Génie Logiciel

PRÉREQUIS :
    pip install numpy

UTILISATION :
    from parseur_ionex_v3 import lire_ionex, afficher_resume

    code = lire_ionex("data/ionex/2023/001/COD0OPSFIN_20230010000_01D_01H_GIM.INX")
    jpl  = lire_ionex("data/ionex/2023/001/JPL0OPSFIN_20230010000_01D_02H_GIM.INX")

    print(code['tec'].shape)   # (25, 71, 73)
    print(jpl['tec'].shape)    # (13, 71, 73)
=============================================================================
"""

import numpy as np
import os
import re
from datetime import datetime


# =============================================================================
# UTILITAIRES
# =============================================================================

def extraire_flottants(chaine):
    """
    Extrait tous les flottants d'une chaîne, même collés.
    Exemple : "87.5-180.0 180.0   5.0 450.0" -> [87.5, -180.0, 180.0, 5.0, 450.0]
    Le regex gère les signes + et - collés à un autre nombre.
    """
    return [float(x) for x in re.findall(r'[+-]?\d+\.?\d*', chaine)]


def lire_epoch(ligne):
    """Parse une ligne d'époque IONEX."""
    try:
        parts = ligne[:60].split()
        return datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                        int(parts[3]), int(parts[4]), int(parts[5]))
    except Exception:
        return None


# =============================================================================
# LECTURE DE L'EN-TÊTE
# =============================================================================

def lire_entete(lignes):
    """
    Lit l'en-tête IONEX de façon robuste.
    Cherche les labels dans toute la ligne (insensible à la casse).
    """
    meta = {
        'version'    : None,
        'centre'     : None,
        'programme'  : None,
        'interval'   : None,
        'n_maps'     : None,
        'lat1'       : None,
        'lat2'       : None,
        'dlat'       : None,
        'lon1'       : None,
        'lon2'       : None,
        'dlon'       : None,
        'hgt1'       : None,
        'base_radius': None,
        'exponent'   : -1,
        'epoch_first': None,
        'epoch_last' : None,
    }
    idx_fin = 0

    for i, ligne in enumerate(lignes):
        lu = ligne.upper()

        if 'END OF HEADER' in lu:
            idx_fin = i + 1
            break
        elif 'IONEX VERSION / TYPE' in lu:
            meta['version'] = ligne[:8].strip()
        elif 'PGM / RUN BY / DATE' in lu:
            meta['programme'] = ligne[:20].strip()
            meta['centre']    = ligne[20:40].strip()
        elif '# OF MAPS IN FILE' in lu:
            vals = extraire_flottants(ligne[:6])
            if vals:
                meta['n_maps'] = int(vals[0])
        elif 'INTERVAL' in lu:
            vals = extraire_flottants(ligne[:6])
            if vals:
                meta['interval'] = int(vals[0])
        elif 'HGT1 / HGT2 / DHGT' in lu:
            vals = extraire_flottants(ligne[:60])
            if vals:
                meta['hgt1'] = vals[0]
        elif 'LAT1 / LAT2 / DLAT' in lu:
            vals = extraire_flottants(ligne[:60])
            if len(vals) >= 3:
                meta['lat1'], meta['lat2'], meta['dlat'] = vals[0], vals[1], vals[2]
        elif 'LON1 / LON2 / DLON' in lu:
            vals = extraire_flottants(ligne[:60])
            if len(vals) >= 3:
                meta['lon1'], meta['lon2'], meta['dlon'] = vals[0], vals[1], vals[2]
        elif 'BASE RADIUS' in lu:
            vals = extraire_flottants(ligne[:6])
            if vals:
                meta['base_radius'] = vals[0]
        elif 'EXPONENT' in lu:
            vals = extraire_flottants(ligne[:6])
            if vals:
                meta['exponent'] = int(vals[0])
        elif 'EPOCH OF FIRST MAP' in lu:
            meta['epoch_first'] = lire_epoch(ligne)
        elif 'EPOCH OF LAST MAP' in lu:
            meta['epoch_last'] = lire_epoch(ligne)

    return meta, idx_fin


# =============================================================================
# CALCUL DE LA GRILLE
# =============================================================================

def calculer_grille(meta):
    """Calcule les vecteurs lat et lon."""
    n_lat = int(round((meta['lat1'] - meta['lat2']) / abs(meta['dlat']))) + 1
    n_lon = int(round((meta['lon2'] - meta['lon1']) / abs(meta['dlon']))) + 1
    lat = np.linspace(meta['lat1'], meta['lat2'], n_lat)
    lon = np.linspace(meta['lon1'], meta['lon2'], n_lon)
    return lat, lon


# =============================================================================
# LECTURE DES BLOCS TEC
# =============================================================================

def lire_blocs_tec(lignes, idx_debut, meta, n_lat, n_lon):
    """
    Lit tous les blocs TEC.
    Gère correctement les lignes de coordonnées avec nombres collés
    comme "87.5-180.0 180.0   5.0 450.0".
    """
    facteur   = 10.0 ** meta['exponent']   # 0.1 pour exponent=-1
    tec       = np.full((meta['n_maps'], n_lat, n_lon), np.nan)
    epochs    = []

    num_carte  = -1
    idx_lat    = -1
    dans_bloc  = False
    buffer_lat = []

    for ligne in lignes[idx_debut:]:
        lu = ligne.upper()

        # ── Début / fin de bloc ──────────────────────────────────────────
        if 'START OF TEC MAP' in lu:
            num_carte += 1
            dans_bloc  = True
            idx_lat    = -1
            buffer_lat = []
            continue

        if 'END OF TEC MAP' in lu:
            dans_bloc = False
            continue

        if 'START OF AUX DATA' in lu or 'START OF RMS MAP' in lu or 'END OF FILE' in lu:
            break

        if not dans_bloc or num_carte >= meta['n_maps']:
            continue

        # ── Époque de la carte ───────────────────────────────────────────
        if 'EPOCH OF CURRENT MAP' in lu:
            epochs.append(lire_epoch(ligne))
            continue

        # ── Ligne de coordonnées de rangée ───────────────────────────────
        # Format : "    87.5-180.0 180.0   5.0 450.0    LAT/LON1/LON2/DLON/H"
        # Détection fiable : contient le label LAT/LON
        if 'LAT/LON' in lu:
            # Sauvegarder le buffer de la rangée précédente si complet
            if idx_lat >= 0 and idx_lat < n_lat and len(buffer_lat) >= n_lon:
                arr = np.array(buffer_lat[:n_lon], dtype=float)
                arr[arr == 9999] = np.nan
                arr *= facteur
                tec[num_carte, idx_lat, :] = arr

            idx_lat   += 1
            buffer_lat = []
            continue

        # ── Lignes de valeurs TEC ────────────────────────────────────────
        # Les valeurs sont des entiers sur 5 caractères chacun
        # On lit par blocs de 5 caractères (format fixe IONEX)
        if idx_lat >= 0 and idx_lat < n_lat:
            ligne_data = ligne.rstrip('\n\r')
            for k in range(0, len(ligne_data), 5):
                chunk = ligne_data[k:k+5]
                if chunk.strip():
                    try:
                        buffer_lat.append(int(chunk))
                    except ValueError:
                        pass

            # Sauvegarde quand la rangée est complète
            if len(buffer_lat) >= n_lon:
                arr = np.array(buffer_lat[:n_lon], dtype=float)
                arr[arr == 9999] = np.nan
                arr *= facteur
                tec[num_carte, idx_lat, :] = arr
                buffer_lat = []

    return tec, epochs


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def lire_ionex(chemin_fichier):
    """
    Lit un fichier IONEX décompressé (CODE ou JPL).

    Retourne un dictionnaire :
        'fichier'   : nom du fichier
        'centre'    : institution (ex: "AIUB", "JPL - GNISD")
        'programme' : programme (ex: "ADDNEQ2 V5.5", "GIM V3.0")
        'interval'  : intervalle en secondes (3600 ou 7200)
        'n_maps'    : nombre de cartes (25 ou 13)
        'lat'       : tableau numpy des latitudes
        'lon'       : tableau numpy des longitudes
        'epochs'    : liste de datetime
        'tec'       : tableau numpy (n_maps, n_lat, n_lon) en TECU
        'meta'      : métadonnées complètes de l'en-tête
    """
    if not os.path.exists(chemin_fichier):
        raise FileNotFoundError(f"Fichier introuvable : {chemin_fichier}")

    with open(chemin_fichier, 'r', encoding='utf-8', errors='replace') as f:
        lignes = f.readlines()

    # 1. En-tête
    meta, idx_debut = lire_entete(lignes)

    # Vérification
    manquants = [c for c in ['interval','n_maps','lat1','lat2','dlat',
                              'lon1','lon2','dlon'] if meta[c] is None]
    if manquants:
        raise ValueError(f"Champs manquants dans l'en-tête : {manquants}")

    # 2. Grille
    lat, lon = calculer_grille(meta)

    # 3. Blocs TEC
    tec, epochs = lire_blocs_tec(lignes, idx_debut, meta, len(lat), len(lon))

    return {
        'fichier'  : os.path.basename(chemin_fichier),
        'centre'   : meta['centre'],
        'programme': meta['programme'],
        'interval' : meta['interval'],
        'n_maps'   : meta['n_maps'],
        'lat'      : lat,
        'lon'      : lon,
        'epochs'   : epochs,
        'tec'      : tec,
        'meta'     : meta,
    }


# =============================================================================
# RÉSUMÉ
# =============================================================================

def afficher_resume(data):
    print("=" * 55)
    print(f"Fichier   : {data['fichier']}")
    print(f"Centre    : {data['centre']}")
    print(f"Programme : {data['programme']}")
    print(f"Intervalle: {data['interval']} s  |  Cartes : {data['n_maps']}")
    print(f"Grille    : {len(data['lat'])} lat x {len(data['lon'])} lon")
    print(f"Latitude  : {data['lat'][0]}° à {data['lat'][-1]}°")
    print(f"Longitude : {data['lon'][0]}° à {data['lon'][-1]}°")
    if data['epochs']:
        print(f"1ère carte: {data['epochs'][0]}")
        print(f"Der carte : {data['epochs'][-1]}")
    valides = data['tec'][~np.isnan(data['tec'])]
    if len(valides) > 0:
        print(f"TEC min   : {valides.min():.2f} TECU")
        print(f"TEC max   : {valides.max():.2f} TECU")
        print(f"TEC moy   : {valides.mean():.2f} TECU")
        pct_nan = 100 * np.isnan(data['tec']).sum() / data['tec'].size
        print(f"NaN       : {pct_nan:.1f}%")
    else:
        print("ATTENTION : toutes les valeurs TEC sont NaN !")
    print("=" * 55)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    FICHIER_CODE = os.path.join("data","ionex","2023","001",
                                "COD0OPSFIN_20230010000_01D_01H_GIM.INX")
    FICHIER_JPL  = os.path.join("data","ionex","2023","001",
                                "JPL0OPSFIN_20230010000_01D_02H_GIM.INX")

    # Fallback ancien nom JPL
    if not os.path.exists(FICHIER_JPL):
        FICHIER_JPL = os.path.join("data","ionex","2023","001","jplg0010.23i")

    print("\n===== TEST PARSEUR IONEX v3 =====\n")

    for label, chemin in [("CODE", FICHIER_CODE), ("JPL", FICHIER_JPL)]:
        print(f">>> Lecture {label}...")
        try:
            data = lire_ionex(chemin)
            afficher_resume(data)
            print(f"Shape TEC : {data['tec'].shape}")
            idx_eq = int(np.argmin(np.abs(data['lat'])))
            print(f"TEC équateur carte 0 (5 val) : {data['tec'][0, idx_eq, :5]}")
        except FileNotFoundError as e:
            print(f"  Fichier non trouvé : {e}")
        except Exception as e:
            print(f"  Erreur : {e}")
        print()

    print("===== FIN DU TEST =====")