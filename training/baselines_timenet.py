"""Baselines on the TimeNet-stored splits: the bar the OpenTSLM fine-tune has to beat."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from forecasttrust_data import FAIL_FLOOR_MM, FAIL_FRACTION, Row, load_rows


def features(row: Row, *, history: bool) -> list[float]:
    """Build the tabular features available at issue time."""
    f1, f2, f3 = row.fc1[-1], row.fc2[-1], row.fc3[-1]
    out = [
        f1, f2, f3, f1 - f2, f2 - f3, max(f1, f2, f3) - min(f1, f2, f3), row.fc1[-2],
        math.sin(2 * math.pi * row.valid_date.timetuple().tm_yday / 365.25),
        math.cos(2 * math.pi * row.valid_date.timetuple().tm_yday / 365.25),
        row.latitude, row.longitude,
    ]  # fmt: skip
    if history:
        n = len(row.observed)
        past_err = row.observed - row.fc1[:n]
        past_abs = np.abs(past_err)
        past_fail = past_abs > np.maximum(FAIL_FLOOR_MM, FAIL_FRACTION * row.fc1[:n])
        out += [past_abs.mean(), past_err.mean(), past_fail.mean(), row.observed.mean(), past_abs[-7:].mean(), past_fail[-7:].mean()]
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """AUROC, average precision and Brier score."""
    return {"auroc": roc_auc_score(y, p), "ap": average_precision_score(y, p), "brier": brier_score_loss(y, p)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(Path(__file__).resolve().parent.parent / "registry"))
    parser.add_argument("--out", default="results/baselines.json")
    args = parser.parse_args()
    rows = load_rows(args.registry)
    y = {k: np.array([r.label for r in v]) for k, v in rows.items()}
    results: dict[str, dict[str, float]] = {}
    results["train failure frequency"] = metrics(y["test"], np.full(len(y["test"]), y["train"].mean()))
    results["forecast magnitude only"] = metrics(y["test"], np.array([min(r.fc1[-1], 200.0) / 200.0 for r in rows["test"]]))
    for history in (True, False):
        tag = "with error history" if history else "no error history"
        x = {k: np.array([features(r, history=history) for r in v]) for k, v in rows.items()}
        for name, model in {
            "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3)),
            "xgboost": XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, verbosity=0),
        }.items():
            model.fit(x["train"], y["train"])
            results[f"{name} ({tag})"] = metrics(y["test"], model.predict_proba(x["test"])[:, 1])
    print(f"test n={len(y['test'])} prevalence={y['test'].mean():.3f}")
    for name, m in results.items():
        print(f"{name:34s} AUROC {m['auroc']:.3f}  AP {m['ap']:.3f}  Brier {m['brier']:.4f}")
    import pathlib

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
