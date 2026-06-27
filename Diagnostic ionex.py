"""
Script de diagnostic : affiche les 80 premières lignes après END OF HEADER
pour voir exactement le format des blocs TEC dans vos vrais fichiers.
"""
import os, sys

def diagnostiquer(chemin):
    if not os.path.exists(chemin):
        print(f"Fichier introuvable : {chemin}")
        return

    with open(chemin, 'r', encoding='utf-8', errors='replace') as f:
        lignes = f.readlines()

    # Trouver END OF HEADER
    idx_debut = 0
    for i, ligne in enumerate(lignes):
        if 'END OF HEADER' in ligne:
            idx_debut = i + 1
            break

    print(f"\n{'='*60}")
    print(f"Fichier : {os.path.basename(chemin)}")
    print(f"END OF HEADER à la ligne {idx_debut-1}")
    print(f"Affichage des lignes {idx_debut} à {idx_debut+60} :")
    print(f"{'='*60}")

    for i, ligne in enumerate(lignes[idx_debut:idx_debut+60]):
        # Afficher numéro, longueur et contenu avec repr pour voir les espaces
        print(f"L{idx_debut+i:4d} (len={len(ligne):3d}) | {repr(ligne[:80])}")

if __name__ == "__main__":
    FICHIER_CODE = os.path.join("data","ionex","2023","001",
                                "COD0OPSFIN_20230010000_01D_01H_GIM.INX")
    FICHIER_JPL  = os.path.join("data","ionex","2023","001",
                                "jplg0010.23i")
    diagnostiquer(FICHIER_CODE)
    diagnostiquer(FICHIER_JPL)