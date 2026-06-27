"""
=============================================================================
Script de téléchargement automatique des fichiers IONEX
Centres : CODE et JPL
Période : 365 jours de l'année 2023
Source  : NASA CDDIS - https://cddis.nasa.gov
=============================================================================
Auteur  : Bassirou Kane
Mémoire : Master 2 Informatique - Génie Logiciel

PRÉREQUIS :
    pip install requests tqdm

UTILISATION :
    1. Renseignez votre login et mot de passe NASA Earthdata ci-dessous
    2. Exécutez : python telecharger_ionex.py
    3. Les fichiers sont enregistrés dans le dossier : data/ionex/2023/
=============================================================================
"""

import requests
import os
import time
import logging
import gzip
import unlzw3
from tqdm import tqdm

# =============================================================================
# CONFIGURATION - À MODIFIER
# =============================================================================

NASA_LOGIN    = "bassiroukane"       # <- Remplacez par votre login
NASA_PASSWORD = "eT&57igjaC&Lf@Z"          # <- Remplacez par votre mot de passe

ANNEE        = 2023
DOSSIER_BASE = "data/ionex/2023"              # Dossier de sortie
MAX_TENTATIVES = 3                             # Nombre de tentatives par fichier
PAUSE_ENTRE_REQUETES = 1                       # Secondes entre chaque téléchargement

# =============================================================================
# CONVENTIONS DE NOMMAGE DES FICHIERS
# =============================================================================

def nom_fichier_code(jour):
    """
    Nouvelle convention CDDIS pour CODE.
    Exemple jour 001 : COD0OPSFIN_20230010000_01D_01H_GIM.INX.gz
    Format : COD0OPSFIN_{ANNEE}{JJJ}0000_01D_01H_GIM.INX.gz
    """
    return f"COD0OPSFIN_{ANNEE}{jour:03d}0000_01D_01H_GIM.INX.gz"

def nom_fichier_jpl(jour):
    """
    Ancienne convention CDDIS pour JPL.
    Exemple jour 001 : jplg0010.23i.Z
    Format : jplg{JJJ}0.{AA}i.Z
    """
    annee_2c = str(ANNEE)[2:]   # "23"
    return f"jplg{jour:03d}0.{annee_2c}i.Z"

def url_fichier(jour, nom_fichier):
    """
    Construit l'URL CDDIS pour un fichier donné.
    Structure : https://cddis.nasa.gov/archive/gnss/products/ionex/{ANNEE}/{JJJ}/{fichier}
    """
    return f"https://cddis.nasa.gov/archive/gnss/products/ionex/{ANNEE}/{jour:03d}/{nom_fichier}"

def chemin_local(jour, nom_fichier):
    """
    Construit le chemin local de sauvegarde.
    Structure : data/ionex/2023/{JJJ}/{fichier_dezippe}
    """
    dossier = os.path.join(DOSSIER_BASE, f"{jour:03d}")
    os.makedirs(dossier, exist_ok=True)
    # Enregistrer sans l'extension de compression
    nom_final = nom_fichier.replace('.gz', '').replace('.Z', '')
    return os.path.join(dossier, nom_final)

# =============================================================================
# TÉLÉCHARGEMENT AVEC GESTION D'ERREURS ET REPRISE
# =============================================================================

def telecharger_fichier(session, url, chemin_sortie, nom_fich):
    """
    Télécharge un fichier depuis CDDIS avec gestion des erreurs et décompression.
    Retourne True si succès, False sinon.
    """
    for tentative in range(1, MAX_TENTATIVES + 1):
        try:
            reponse = session.get(url, timeout=60)

            # Fichier introuvable sur le serveur (jour manquant)
            if reponse.status_code == 404:
                return "absent"

            # Erreur d'authentification
            if reponse.status_code == 401:
                logging.error("Erreur d'authentification. Vérifiez votre login/mot de passe.")
                return "auth_error"

            # Autre erreur HTTP
            if reponse.status_code != 200:
                logging.warning(f"  HTTP {reponse.status_code} - tentative {tentative}/{MAX_TENTATIVES}")
                time.sleep(2 * tentative)
                continue

            # Décompression du fichier en mémoire
            try:
                contenu = reponse.content
                if nom_fich.endswith('.gz'):
                    contenu_clair = gzip.decompress(contenu)
                elif nom_fich.endswith('.Z'):
                    contenu_clair = unlzw3.unlzw(contenu)
                else:
                    contenu_clair = contenu

                # Écriture du fichier décompressé
                with open(chemin_sortie, 'wb') as f:
                    f.write(contenu_clair)
            except Exception as e:
                # Si le téléchargement coupe ou la décompression échoue, on supprime le fichier corrompu
                if os.path.exists(chemin_sortie):
                    os.remove(chemin_sortie)
                raise e # Remonter l'erreur pour la capturer dans les blocs except en dessous

            return "ok"

        except requests.exceptions.ConnectionError:
            logging.warning(f"  Erreur réseau - tentative {tentative}/{MAX_TENTATIVES}")
            time.sleep(5 * tentative)

        except requests.exceptions.Timeout:
            logging.warning(f"  Timeout - tentative {tentative}/{MAX_TENTATIVES}")
            time.sleep(5 * tentative)

        except Exception as e:
            logging.error(f"  Erreur inattendue : {e} - tentative {tentative}/{MAX_TENTATIVES}")
            time.sleep(5 * tentative)

    return "echec"

# =============================================================================
# JOURNAL DE TÉLÉCHARGEMENT
# =============================================================================

def initialiser_journal():
    """
    Configure le journal de téléchargement.
    Les logs sont écrits dans telechargement.log et affichés dans la console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("telechargement.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    initialiser_journal()
    logging.info("=" * 60)
    logging.info("Démarrage du téléchargement IONEX 2023")
    logging.info(f"Centres : CODE + JPL | Jours : 001 à 365")
    logging.info(f"Dossier de sortie : {os.path.abspath(DOSSIER_BASE)}")
    logging.info("=" * 60)

    # Vérification des identifiants
    if NASA_LOGIN == "votre_login_earthdata":
        print("\n⚠️  ATTENTION : Renseignez votre login et mot de passe NASA Earthdata")
        print("   Modifiez les variables NASA_LOGIN et NASA_PASSWORD en haut du script.\n")
        return

    # Création de la session avec authentification
    class EarthdataSession(requests.Session):
        def rebuild_auth(self, prepared_request, response):
            # Ne pas supprimer l'authentification lors de la redirection vers Earthdata
            pass

    session = EarthdataSession()
    session.auth = (NASA_LOGIN, NASA_PASSWORD)

    # Compteurs
    nb_ok       = 0
    nb_absents  = 0
    nb_echecs   = 0
    fichiers_echec = []

    # Liste de toutes les tâches : (jour, centre, nom_fichier)
    taches = []
    for jour in range(1, 366):
        taches.append((jour, "CODE", nom_fichier_code(jour)))
        taches.append((jour, "JPL",  nom_fichier_jpl(jour)))

    # Téléchargement avec barre de progression
    with tqdm(total=len(taches), desc="Téléchargement", unit="fichier") as barre:
        for jour, centre, nom_fich in taches:
            url     = url_fichier(jour, nom_fich)
            sortie  = chemin_local(jour, nom_fich)

            # Ne pas retélécharger un fichier déjà présent
            if os.path.exists(sortie) and os.path.getsize(sortie) > 0:
                nb_ok += 1
                barre.update(1)
                barre.set_postfix(ok=nb_ok, absents=nb_absents, echecs=nb_echecs)
                continue

            statut = telecharger_fichier(session, url, sortie, nom_fich)

            if statut == "ok":
                nb_ok += 1
                logging.info(f"OK       Jour {jour:03d} {centre} : {nom_fich}")
            elif statut == "absent":
                nb_absents += 1
                logging.warning(f"ABSENT   Jour {jour:03d} {centre} : {nom_fich}")
            elif statut == "auth_error":
                logging.error("Authentification échouée. Arrêt du script.")
                break
            else:
                nb_echecs += 1
                fichiers_echec.append((jour, centre, nom_fich))
                logging.error(f"ÉCHEC    Jour {jour:03d} {centre} : {nom_fich}")

            barre.update(1)
            barre.set_postfix(ok=nb_ok, absents=nb_absents, echecs=nb_echecs)
            time.sleep(PAUSE_ENTRE_REQUETES)

    # Résumé final
    logging.info("=" * 60)
    logging.info("RÉSUMÉ DU TÉLÉCHARGEMENT")
    logging.info(f"  Fichiers téléchargés : {nb_ok}")
    logging.info(f"  Fichiers absents     : {nb_absents}  (jours manquants sur CDDIS)")
    logging.info(f"  Échecs               : {nb_echecs}")
    logging.info("=" * 60)

    if fichiers_echec:
        logging.warning("Fichiers en échec à relancer :")
        for jour, centre, nom in fichiers_echec:
            logging.warning(f"  Jour {jour:03d} {centre} : {nom}")

    print(f"\n✅ Terminé. {nb_ok} fichiers téléchargés.")
    print(f"📄 Journal complet : telechargement.log")
    print(f"📁 Fichiers dans   : {os.path.abspath(DOSSIER_BASE)}")

if __name__ == "__main__":
    main()
