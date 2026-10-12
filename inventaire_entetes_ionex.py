"""
=============================================================================
Inventaire des en-têtes IONEX et des valeurs décodées (Annexe A du mémoire)
=============================================================================
Auteur  : Bassirou Kane
Mémoire : Master 2 Informatique - Génie Logiciel

OBJECTIF
    Produire les informations de l'Annexe A qui ne proviennent pas des modules
    d'analyse :
      - nombre de stations déclaré dans l'en-tête (# OF STATIONS) et nombre de
        stations dont le biais instrumental est fourni (STATION / BIAS / RMS),
        pour CODE et JPL, jour par jour (figure A.1) ;
      - valeurs extrêmes et valeurs nulles de la base TEC (tableau A.5) ;
      - contribution des valeurs nulles de CODE au biais.

UTILISATION (depuis la racine du projet)
    python inventaire_entetes_ionex.py

RÉSULTATS
    Tableaux en console + data/stats/figures/stations_entetes_ionex.png

IMPACT SUR LES AUTRES MODULES
    Aucun (lecture seule des fichiers IONEX et de la base TEC).
=============================================================================
"""

import os
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DOSSIER_IONEX = "data/ionex/2023"
DOSSIER_BASE = "data/base_tec"
DOSSIER_FIGS = "data/stats/figures"
os.makedirs(DOSSIER_FIGS, exist_ok=True)
LAT = np.linspace(87.5, -87.5, 71)
LON = np.linspace(-180.0, 180.0, 73)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 150})


def centre_du_fichier(nom):
    n = nom.lower()
    if n.startswith("cod0opsfin"):
        return "CODE"
    if n.startswith("jplg") or n.startswith("jpl0opsfin"):
        return "JPL"
    return None


def lire_entete(chemin):
    """Retourne (# OF STATIONS, nombre de lignes STATION / BIAS / RMS)."""
    n_stations, n_biais = None, 0
    with open(chemin, encoding="utf-8", errors="replace") as f:
        for ligne in f:
            etiquette = ligne[60:].strip()
            if etiquette == "# OF STATIONS":
                n_stations = int(ligne[:60])
            elif etiquette == "STATION / BIAS / RMS":
                n_biais += 1
            elif etiquette == "END OF HEADER":
                break
    return n_stations, n_biais


def inventaire_stations():
    res = {c: {"stations": np.full(365, np.nan), "biais": np.full(365, np.nan)} for c in ("CODE", "JPL")}
    for jour in range(1, 366):
        dossier = os.path.join(DOSSIER_IONEX, f"{jour:03d}")
        if not os.path.isdir(dossier):
            continue
        for nom in os.listdir(dossier):
            c = centre_du_fichier(nom)
            if c:
                s, b = lire_entete(os.path.join(dossier, nom))
                res[c]["stations"][jour - 1] = s if s is not None else np.nan
                res[c]["biais"][jour - 1] = b
    return res


def date_du_jour(j):
    return date(2023, 1, 1) + timedelta(days=int(j) - 1)


def main():
    print("=" * 70)
    print("Inventaire des en-têtes IONEX et des valeurs décodées — 2023")
    print("=" * 70)
    res = inventaire_stations()
    for c in ("CODE", "JPL"):
        s, b = res[c]["stations"], res[c]["biais"]
        print(f"\n{c} : # OF STATIONS min={np.nanmin(s):.0f} ({date_du_jour(np.nanargmin(s)+1):%d/%m}) "
              f"max={np.nanmax(s):.0f} ({date_du_jour(np.nanargmax(s)+1):%d/%m}) médiane={np.nanmedian(s):.0f}")
        print(f"{c} : stations avec biais  min={np.nanmin(b):.0f} ({date_du_jour(np.nanargmin(b)+1):%d/%m}) "
              f"max={np.nanmax(b):.0f}  ; j029={b[28]:.0f}  j031={b[30]:.0f}")

    dates = [date_du_jour(j) for j in range(1, 366)]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(dates, res["CODE"]["stations"], color="#1565C0", lw=1.3, label="CODE : champ « # OF STATIONS »")
    ax.plot(dates, res["JPL"]["biais"], color="#E65100", lw=1.3, label="JPL : stations dont le biais (DCB) est fourni")
    ax.plot(dates, res["JPL"]["stations"], color="#E65100", lw=1.0, ls=":", label="JPL : champ « # OF STATIONS »")
    ax.set_ylabel("Nombre de stations")
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title("Nombre de stations déclaré dans l'en-tête des fichiers IONEX journaliers (2023)", fontweight="bold")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8.8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(DOSSIER_FIGS, "stations_entetes_ionex.png"), bbox_inches="tight")
    plt.close(fig)

    print("\nValeurs décodées (base TEC) :")
    code = np.load(os.path.join(DOSSIER_BASE, "code_tec_2023.npz"))["tec"]
    jpl = np.load(os.path.join(DOSSIER_BASE, "jpl_tec_2023.npz"))["tec"]
    for nom, t, pas_h in (("CODE", code, 1), ("JPL", jpl, 2)):
        i = np.unravel_index(np.nanargmax(t), t.shape)
        nuls = (t == 0)
        print(f"  {nom}: N={t.size}  absentes={int(np.isnan(t).sum())}  min={np.nanmin(t):.1f}  "
              f"max={np.nanmax(t):.1f} ({date_du_jour(i[0]+1):%d/%m}, {i[1]*pas_h} h TU, "
              f"{LAT[i[2]]:+.1f}°, {LON[i[3]]:+.1f}°)  P99,99={np.nanpercentile(t, 99.99):.1f}  "
              f"nulles={100*nuls.mean():.2f} % (dont {100*nuls[:, :, np.abs(LAT) >= 60].sum()/max(nuls.sum(),1):.0f} % "
              f"au-delà de 60°)")
    c13 = code[:, 0::2].astype(float); d = c13 - jpl.astype(float); z = (c13 == 0)
    for nom, sel in (("global", np.ones(71, bool)), ("pôle Sud", LAT <= -60)):
        dd, zz = d[:, :, sel], z[:, :, sel]
        print(f"  Contribution des valeurs nulles de CODE au MBE ({nom}) : {dd[zz].sum() / dd.size:+.3f} TECU")
    print("\n✅ Inventaire terminé.")


if __name__ == "__main__":
    main()
