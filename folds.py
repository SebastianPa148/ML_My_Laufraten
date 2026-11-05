import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import config as c


def make_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Äußere und innere Folds einmal erzeugen, nach Patient gruppiert,
    stratifiziert nach Quantilen des Spiegels. Wird für ALLE Läufe wiederverwendet."""
    df = df.reset_index(drop=True).copy()
    strata = pd.qcut(df[c.TARGET], q=c.N_STRATA, labels=False, duplicates="drop").to_numpy()
    groups = df[c.GROUP_COL].to_numpy()

    outer = StratifiedGroupKFold(n_splits=c.OUTER_FOLDS, shuffle=True, random_state=c.SEED)
    outer_fold = np.full(len(df), -1)

    for k, (train_idx, test_idx) in enumerate(outer.split(df, strata, groups)):
        outer_fold[test_idx] = k

        inner = StratifiedGroupKFold(n_splits=c.INNER_FOLDS, shuffle=True, random_state=c.SEED)
        inner_fold = np.full(len(df), -1)
        for j, (_, inner_test) in enumerate(inner.split(train_idx, strata[train_idx], groups[train_idx])):
            inner_fold[train_idx[inner_test]] = j
        df[f"inner_fold_{k}"] = inner_fold

    df["outer_fold"] = outer_fold
    return df


def check_folds(df: pd.DataFrame) -> None:
    # 1) Kein Patient in mehreren äußeren Folds
    n_folds_pro_patient = df.groupby(c.GROUP_COL)["outer_fold"].nunique()
    assert (n_folds_pro_patient == 1).all(), "Patient in mehreren äußeren Folds!"
    # 2) Größe und Spiegelverteilung pro Fold
    print(df.groupby("outer_fold").agg(
        messungen=(c.TARGET, "size"),
        patienten=(c.GROUP_COL, "nunique"),
        spiegel_median=(c.TARGET, "median"),
        spiegel_mittel=(c.TARGET, "mean"),
    ))