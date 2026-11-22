"""Load the forecasttrust/india-rainfall-reliability dataset from TimeNet into plain rows.

Everything downstream (baselines and the OpenTSLM fine-tune) reads the same rows, so the models
are compared on identical samples and identical TimeNet-stored splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

DATASET_ID = "forecasttrust/india-rainfall-reliability@1.0.0"
SPLITS = ("train", "validation", "test")
LABELS = ("reliable", "failure")
FAIL_FLOOR_MM = 5.0
FAIL_FRACTION = 0.5


@dataclass(frozen=True)
class Row:
    """One city and valid day, with the inputs available when the day-1 forecast is issued."""

    city: str
    valid_date: date
    latitude: float
    longitude: float
    fc1: np.ndarray  # day-1 forecasts, 14 days ending on the valid day (mm)
    fc2: np.ndarray  # day-2 forecasts, same days
    fc3: np.ndarray  # day-3 forecasts, same days
    observed: np.ndarray  # ERA5 rainfall for the first 12 days, ending two days before the valid day
    label: int  # 1 = failure


def load_rows(registry: str | Path) -> dict[str, list[Row]]:
    """Read every split from a local TimeNet registry.

    Args:
        registry: Directory that holds ``forecasttrust/india-rainfall-reliability/1.0.0``.

    Returns:
        The rows of each split, in the stored order.
    """
    import timenet
    from timenet.types import ClassificationTask
    from timenet.types.splits import Split
    from timenet.types.text import read_value

    out: dict[str, list[Row]] = {name: [] for name in SPLITS}
    with timenet.TimeNet(sources=[str(registry)]).open(DATASET_ID.split("@")[0]) as reader:
        dataset = reader.read()
        for name in SPLITS:
            for task in dataset.iter_tasks(split=Split(name)):
                if not isinstance(task, ClassificationTask):
                    continue
                record = task.inputs[0]
                signals = {s.name: s.to_numpy().astype(np.float64) for src in record.sources for s in src.signals}
                annotations = {a.key: a.value for a in record.annotations}
                meta = record.metadata
                assert task.targets is not None
                label = LABELS.index(read_value(task.targets[0]))
                fc1 = signals["forecast_day1"]
                valid = date.fromisoformat(str(meta["valid_date"]))
                out[name].append(
                    Row(
                        city=str(meta["city"]),
                        valid_date=valid,
                        latitude=float(annotations["latitude"]),  # type: ignore[arg-type]
                        longitude=float(annotations["longitude"]),  # type: ignore[arg-type]
                        fc1=fc1,
                        fc2=signals["forecast_day2"],
                        fc3=signals["forecast_day3"],
                        observed=signals["observed_verified"],
                        label=label,
                    )
                )
    return out
