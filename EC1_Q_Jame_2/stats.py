from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Stats:
    n: int
    mean: float
    std: float
    var: float


def describe(scores: pd.Series) -> Stats:
    s = pd.to_numeric(scores, errors="coerce").dropna().astype(float)
    if s.empty:
        return Stats(n=0, mean=float("nan"), std=float("nan"), var=float("nan"))

    mean = float(s.mean())
    var = float(s.var(ddof=0))
    std = float(s.std(ddof=0))
    return Stats(n=int(s.size), mean=mean, std=std, var=var)
