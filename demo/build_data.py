"""Export real test-split predictions (OpenTSLM, 3 seeds) for the demo page."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from forecasttrust_data import load_rows

rows = load_rows(ROOT / "registry")["test"]
y = np.array([r.label for r in rows])
logits = [np.load(ROOT / f"results/qwen05_s{s}/test_logit.npy") for s in range(3)]
for l in logits:
    assert len(l) == len(rows)
    print("AUROC check", round(roc_auc_score(y, l), 4))
p = np.mean([1 / (1 + np.exp(-l)) for l in logits], axis=0)
print("ensemble AUROC", round(roc_auc_score(y, p), 4))
out = [
    {"c": r.city, "d": r.valid_date.isoformat(), "f1": round(float(r.fc1[-1]), 1), "f2": round(float(r.fc2[-1]), 1),
     "f3": round(float(r.fc3[-1]), 1), "h": [round(float(v), 1) for v in r.fc1[:-1]], "o": [round(float(v), 1) for v in r.observed],
     "y": int(r.label), "p": round(float(q), 3)}
    for r, q in zip(rows, p)
]
(ROOT / "demo" / "data.json").write_text(json.dumps(out, separators=(",", ":")))
print(len(out), "rows")
