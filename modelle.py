import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import config as c

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

EN_GRID = {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
           "l1_ratio": [0.1, 0.5, 0.9]}


class LogLinearRegressor(RegressorMixin, BaseEstimator):
    """log(Spiegel) ~ log(Features) mit Elastic Net.
    smear=False: Rückrechnung per exp() (Median).
    smear=True : exp() mal Duan-Smearing-Faktor (Mittelwert).
    Der Faktor wird nur aus den Residuen der jeweiligen Trainingsdaten
    geschätzt (bei jedem Fit neu), nie aus Test-Daten."""

    def __init__(self, alpha=1e-3, l1_ratio=0.5, smear=False):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.smear = smear

    def fit(self, X, y):
        self.pipe_ = Pipeline([
            ("log", FunctionTransformer(np.log)),
            ("scale", StandardScaler()),
            ("en", ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                              max_iter=100_000)),
        ])
        log_y = np.log(y)
        self.pipe_.fit(X, log_y)
        resid = log_y - self.pipe_.predict(X)
        self.factor_ = float(np.mean(np.exp(resid))) if self.smear else 1.0
        return self

    def predict(self, X):
        return np.exp(self.pipe_.predict(X)) * self.factor_


def lr_linear():
    est = Pipeline([("scale", StandardScaler()),
                    ("en", ElasticNet(max_iter=100_000))])
    return est, {f"en__{k}": v for k, v in EN_GRID.items()}


def lr_log():
    return LogLinearRegressor(smear=False), dict(EN_GRID)


def lr_log_smear():
    return LogLinearRegressor(smear=True), dict(EN_GRID)


def lgbm():
    est = LGBMRegressor(random_state=c.SEED, n_jobs=4, verbose=-1,
                        deterministic=True, force_row_wise=True)
    grid = {"num_leaves":        [4, 8, 16],
            "learning_rate":     [0.03, 0.1],
            "n_estimators":      [100, 300],
            "min_child_samples": [20, 50, 100]}
    return est, grid


MODELS = {"LR_linear": lr_linear, "LR_log": lr_log,
          "LR_log_smear": lr_log_smear, "LightGBM": lgbm}


def run_nested(df: pd.DataFrame, model_name: str, set_name: str):
    """Nested CV für ein Modell und ein Feature-Set (c.FEATURE_SETS).
    Nutzt die vorab erzeugten Folds (outer_fold, inner_fold_k)."""
    X = df[c.FEATURE_SETS[set_name]].to_numpy()
    y = df[c.TARGET].to_numpy()
    outer = df["outer_fold"].to_numpy()
    fold_rows, oof_rows = [], []

    for k in range(c.OUTER_FOLDS):
        train_idx = np.where(outer != k)[0]
        test_idx = np.where(outer == k)[0]
        inner = df[f"inner_fold_{k}"].to_numpy()

        cv = []
        for j in range(c.INNER_FOLDS):
            val = train_idx[inner[train_idx] == j]
            tr = train_idx[inner[train_idx] != j]
            cv.append((np.searchsorted(train_idx, tr),
                       np.searchsorted(train_idx, val)))

        est, grid = MODELS[model_name]()
        gs = GridSearchCV(est, grid, cv=cv, refit=True, n_jobs=1,
                          scoring="neg_root_mean_squared_error")
        gs.fit(X[train_idx], y[train_idx])

        pred = gs.predict(X[test_idx])
        fold_rows.append({
            "modell": model_name, "featureset": set_name, "outer_fold": k,
            "n_test": len(test_idx),
            "R2": r2_score(y[test_idx], pred),
            "MAE": mean_absolute_error(y[test_idx], pred),
            "RMSE": root_mean_squared_error(y[test_idx], pred),
            "smear_factor": getattr(gs.best_estimator_, "factor_", np.nan),
            "best_params": str(gs.best_params_),
        })
        oof_rows.append(pd.DataFrame({
            "row": test_idx, "outer_fold": k, "modell": model_name,
            "featureset": set_name, "y_true": y[test_idx], "y_pred": pred}))

    return pd.DataFrame(fold_rows), pd.concat(oof_rows, ignore_index=True)


def run_all(df: pd.DataFrame, out_dir="results"):
    folds_all, oof_all = [], []
    for model_name in MODELS:
        for set_name in c.FEATURE_SETS:
            print(f"läuft: {model_name} | {set_name}", flush=True)
            f, o = run_nested(df, model_name, set_name)
            folds_all.append(f); oof_all.append(o)
            pd.concat(folds_all).to_csv(f"{out_dir}/metriken_pro_fold.csv", index=False)
            pd.concat(oof_all).to_csv(f"{out_dir}/oof_vorhersagen.csv", index=False)
    return pd.concat(folds_all, ignore_index=True), pd.concat(oof_all, ignore_index=True)


def zusammenfassung(folds: pd.DataFrame) -> pd.DataFrame:
    """Mittelwert ± SD über die 5 äußeren Folds."""
    g = folds.groupby(["modell", "featureset"])[["R2", "MAE", "RMSE"]]
    return g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)


def gewinn_gegen_referenz(folds: pd.DataFrame) -> pd.DataFrame:
    """Mittlere Verbesserung jeder Laufrate gegenüber dem Referenzmodell
    (gleiches Modell, gleicher Fold). Positiv = besser als nur CG_GFR."""
    REF = "Referenz_nur_CG_GFR"
    ref = (folds[folds.featureset == REF]
           .set_index(["modell", "outer_fold"])[["R2", "MAE", "RMSE"]])
    d = (folds[folds.featureset != REF]
         .join(ref, on=["modell", "outer_fold"], rsuffix="_ref"))
    d["dR2"]   = d["R2"] - d["R2_ref"]
    d["dMAE"]  = d["MAE_ref"] - d["MAE"]      # Fehlerabnahme
    d["dRMSE"] = d["RMSE_ref"] - d["RMSE"]    # Fehlerabnahme
    return d.groupby(["modell", "featureset"])[["dR2", "dMAE", "dRMSE"]].mean().round(3)