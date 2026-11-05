import itertools
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import config as c

REF = "Referenz_nur_CG_GFR"


def lade_oof(df: pd.DataFrame, pfad="results/oof_vorhersagen.csv") -> pd.DataFrame:
    """OOF-Vorhersagen laden und Patienten-ID anhängen (row = Zeilenposition in df)."""
    oof = pd.read_csv(pfad)
    oof["pid"] = df.loc[oof["row"].to_numpy(), c.GROUP_COL].to_numpy()
    return oof


def bootstrap_metriken(oof, modell, B=2000, seed=c.SEED):
    """Cluster-Bootstrap für ein Modell. Gibt (Punktwerte, Bootstrap-Werte) je Metrik
    zurück, jeweils als DataFrame mit den Feature-Sets als Spalten."""
    d = oof[oof.modell == modell].copy()
    err = d["y_pred"] - d["y_true"]
    d["se"], d["ae"], d["y2"] = err**2, err.abs(), d["y_true"]**2

    sse = d.groupby(["pid", "featureset"])["se"].sum().unstack("featureset")
    sae = d.groupby(["pid", "featureset"])["ae"].sum().unstack("featureset")
    erste = sse.columns[0]
    basis = d[d.featureset == erste].groupby("pid").agg(
        n=("y_true", "size"), sy=("y_true", "sum"), syy=("y2", "sum")).loc[sse.index]

    P = len(sse)
    rng = np.random.default_rng(seed)                 # gleicher Seed -> gleiche Ziehungen
    w = np.vstack([np.ones(P, dtype=int),             # Zeile 0 = Originaldaten
                   rng.multinomial(P, np.full(P, 1 / P), size=B)])

    N, SY, SYY = w @ basis["n"].to_numpy(), w @ basis["sy"].to_numpy(), w @ basis["syy"].to_numpy()
    SSE, SAE = w @ sse.to_numpy(), w @ sae.to_numpy()
    sst = SYY - SY**2 / N

    roh = {"RMSE": np.sqrt(SSE / N[:, None]),
           "MAE":  SAE / N[:, None],
           "R2":   1 - SSE / sst[:, None]}
    return {m: pd.DataFrame(v, columns=sse.columns) for m, v in roh.items()}


def vergleiche(oof, modelle=None, B=2000):
    """Gepaarte Vergleiche je Modell. Familie 1: 6 Laufrate-gegen-Laufrate,
    Familie 2: 4 Laufrate-gegen-Referenz. Holm je (Modell, Familie, Metrik).
    Differenz = A - B. Bei RMSE/MAE bedeutet negativ: A besser. Bei R2: positiv = A besser."""
    modelle = modelle or sorted(oof.modell.unique())
    familien = {"Laufrate_vs_Laufrate": list(itertools.combinations(c.RATE_COLS, 2)),
                "Laufrate_vs_Referenz": [(r, REF) for r in c.RATE_COLS]}
    zeilen = []
    for m in modelle:
        boot = bootstrap_metriken(oof, m, B=B)
        for fam, paare in familien.items():
            for metrik, tab in boot.items():
                teil = []
                for a, b in paare:
                    diff = (tab[a] - tab[b]).to_numpy()
                    punkt, bs = diff[0], diff[1:]
                    p = 2 * min((bs <= 0).sum() + 1, (bs >= 0).sum() + 1) / (len(bs) + 1)
                    teil.append({"modell": m, "familie": fam, "metrik": metrik,
                                 "A": a, "B": b, "diff": punkt,
                                 "ci_low": np.percentile(bs, 2.5),
                                 "ci_high": np.percentile(bs, 97.5),
                                 "p_roh": min(p, 1.0)})
                t = pd.DataFrame(teil)
                t["p_holm"] = multipletests(t["p_roh"], method="holm")[1]
                zeilen.append(t)
    return pd.concat(zeilen, ignore_index=True)