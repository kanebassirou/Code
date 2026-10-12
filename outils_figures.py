"""
Sauvegarde robuste des figures dans un dossier synchronisé par OneDrive.

OneDrive verrouille parfois les fichiers existants pendant la synchro, ce qui
fait échouer l'écrasement direct avec « OSError: [Errno 22] Invalid argument ».
On écrit donc d'abord dans un fichier temporaire puis on remplace l'ancien,
avec quelques nouvelles tentatives si nécessaire.
"""

import os
import time


def sauver_figure(fig, chemin, tentatives=5):
    racine, ext = os.path.splitext(chemin)
    temp = racine + ".tmp" + ext
    fig.savefig(temp, bbox_inches='tight')
    for k in range(tentatives):
        try:
            os.replace(temp, chemin)
            return chemin
        except OSError:
            time.sleep(1 + k)
    # Fichier toujours verrouillé : on garde la nouvelle version à côté
    secours = racine + "_nouveau" + ext
    os.replace(temp, secours)
    print(f"  [!] {chemin} verrouillé (OneDrive ?) -> sauvegardé dans {secours}")
    return secours
