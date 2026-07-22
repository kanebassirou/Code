"""
=============================================================================
Script de téléchargement des fichiers JPL manquants
Source : https://sideshow.jpl.nasa.gov/pub/iono_daily/IONEX_final/y2023/
Pas d'authentification requise sur ce serveur
=============================================================================
Auteur  : Bassirou Kane
Mémoire : Master 2 Informatique - Génie Logiciel

PRÉREQUIS :
    pip install requests tqdm

UTILISATION :
    python telecharger_jpl_sideshow.py

RÉSULTAT :
    Fichiers JPL décompressés dans data/ionex/2023/{JJJ}/
    Même structure que les fichiers CODE déjà téléchargés
=============================================================================
"""

import requests
import os
import gzip
import time
import logging
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

ANNEE        = 2023
DOSSIER_BASE = "data/ionex/2023"
MAX_TENTATIVES = 3
PAUSE        = 0.5   # secondes entre requêtes (serveur sans auth = plus rapide)

# URL de base du serveur JPL sideshow (sans authentification)
URL_BASE = "https://sideshow.jpl.nasa.gov/pub/iono_daily/IONEX_final/y2023/"

# =============================================================================
# CONVENTIONS DE NOMMAGE
# =============================================================================

def nom_fichier_jpl_nouveau(jour):
    """
    Nouvelle convention : JPL0OPSFIN_20230010000_01D_02H_GIM.INX.gz
    """
    return f"JPL0OPSFIN_{ANNEE}{jour:03d}0000_01D_02H_GIM.INX.gz"

def nom_fichier_jpl_ancien(jour):
    """
    Ancienne convention : JPLG0010.23I.gz  (majuscules sur sideshow)
    """
    annee_2c = str(ANNEE)[2:]
    return f"JPLG{jour:03d}0.{annee_2c}I.gz"

def chemin_local(jour, nom_fichier):
    """
    Chemin de destination — on sauvegarde sans l'extension .gz
    et avec le nom en minuscules pour cohérence avec les fichiers existants.
    """
    dossier = os.path.join(DOSSIER_BASE, f"{jour:03d}")
    os.makedirs(dossier, exist_ok=True)
    # Nom final : supprimer .gz, mettre en minuscules
    nom_final = nom_fichier.replace(".gz", "").lower()
    return os.path.join(dossier, nom_final)

def fichier_deja_present(jour):
    """
    Vérifie si le fichier JPL de ce jour est déjà téléchargé
    (ancienne convention jplg ou nouvelle convention JPL0OPSFIN).
    """
    annee_2c = str(ANNEE)[2:]
    chemin_ancien  = os.path.join(DOSSIER_BASE, f"{jour:03d}", f"jplg{jour:03d}0.{annee_2c}i")
    chemin_nouveau = os.path.join(DOSSIER_BASE, f"{jour:03d}",
                                  f"JPL0OPSFIN_{ANNEE}{jour:03d}0000_01D_02H_GIM.INX")
    chemin_nouveau_min = chemin_nouveau.lower()

    for chemin in [chemin_ancien, chemin_nouveau, chemin_nouveau_min]:
        if os.path.exists(chemin) and os.path.getsize(chemin) > 0:
            return True
    return False

# =============================================================================
# TÉLÉCHARGEMENT
# =============================================================================

def telecharger(session, url, chemin_sortie):
    """
    Télécharge et décompresse un fichier .gz depuis sideshow.
    Retourne : "ok", "absent", "echec"
    """
    for tentative in range(1, MAX_TENTATIVES + 1):
        try:
            reponse = session.get(url, timeout=60)

            if reponse.status_code == 404:
                return "absent"
            if reponse.status_code != 200:
                logging.warning(f"  HTTP {reponse.status_code} tentative {tentative}/{MAX_TENTATIVES}")
                time.sleep(2 * tentative)
                continue

            # Décompression gzip en mémoire
            contenu = gzip.decompress(reponse.content)

            with open(chemin_sortie, 'wb') as f:
                f.write(contenu)

            return "ok"

        except requests.exceptions.ConnectionError:
            logging.warning(f"  Erreur réseau tentative {tentative}/{MAX_TENTATIVES}")
            time.sleep(5 * tentative)
        except requests.exceptions.Timeout:
            logging.warning(f"  Timeout tentative {tentative}/{MAX_TENTATIVES}")
            time.sleep(5 * tentative)
        except Exception as e:
            logging.error(f"  Erreur : {e} tentative {tentative}/{MAX_TENTATIVES}")
            time.sleep(3 * tentative)

    return "echec"

# =============================================================================
# JOURNAL
# =============================================================================

def initialiser_journal():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("telechargement_jpl_sideshow.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    initialiser_journal()
    logging.info("=" * 60)
    logging.info("Téléchargement JPL sideshow - Fichiers manquants 2023")
    logging.info(f"Source : {URL_BASE}")
    logging.info("=" * 60)

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})

    nb_ok        = 0
    nb_deja_la   = 0
    nb_absents   = 0
    nb_echecs    = 0
    echecs_liste = []

    with tqdm(total=365, desc="JPL sideshow", unit="jour") as barre:
        for jour in range(1, 366):

            # Déjà téléchargé lors de la première session ?
            if fichier_deja_present(jour):
                nb_deja_la += 1
                barre.update(1)
                barre.set_postfix(ok=nb_ok, deja=nb_deja_la, echecs=nb_echecs)
                continue

            # Essayer d'abord la nouvelle convention (JPL0OPSFIN_...)
            statut = "absent"
            nom_utilise = ""

            for nom_fich in [nom_fichier_jpl_nouveau(jour), nom_fichier_jpl_ancien(jour)]:
                url    = URL_BASE + nom_fich
                sortie = chemin_local(jour, nom_fich)
                statut = telecharger(session, url, sortie)
                nom_utilise = nom_fich

                if statut == "ok":
                    break
                # Si absent → essayer l'autre convention

            if statut == "ok":
                nb_ok += 1
                logging.info(f"OK      Jour {jour:03d} : {nom_utilise}")
            elif statut == "absent":
                nb_absents += 1
                logging.warning(f"ABSENT  Jour {jour:03d} : non disponible sur sideshow")
            else:
                nb_echecs += 1
                echecs_liste.append(jour)
                logging.error(f"ÉCHEC   Jour {jour:03d}")

            barre.update(1)
            barre.set_postfix(ok=nb_ok, deja=nb_deja_la, echecs=nb_echecs)
            time.sleep(PAUSE)

    # Résumé
    logging.info("=" * 60)
    logging.info("RÉSUMÉ")
    logging.info(f"  Déjà présents  : {nb_deja_la}")
    logging.info(f"  Téléchargés    : {nb_ok}")
    logging.info(f"  Absents        : {nb_absents}")
    logging.info(f"  Échecs         : {nb_echecs}")
    logging.info(f"  TOTAL JPL      : {nb_deja_la + nb_ok} / 365")
    logging.info("=" * 60)

    if echecs_liste:
        logging.warning(f"Jours en échec : {echecs_liste}")

    total = nb_deja_la + nb_ok
    print(f"\n✅ Terminé. Total JPL disponibles : {total} / 365")
    print(f"📄 Journal : telechargement_jpl_sideshow.log")

    if total == 365:
        print("\n🎉 Tous les 365 jours JPL sont maintenant disponibles !")
        print("   Vous pouvez relancer construire_base_tec.py pour mettre à jour la base.")

if __name__ == "__main__":
    main()