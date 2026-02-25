# benchmark.py

import time
import numpy as np
import pandas as pd
from dstools_dtdy import mylm


def time_engine(n, p, engine, repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + 1.0 + rng.normal(size=n)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        mylm(X, y, engine=engine)
        times.append(time.perf_counter() - t0)

    times = np.asarray(times)
    return times.mean(), times.std(ddof=1)


def run_benchmark():
    settings = [
        (200, 5),
        (500, 10),
        (1000, 20),
        (2000, 30),
    ]

    rows = []

    for n, p in settings:
        mean_np, sd_np = time_engine(n, p, "numpy")
        mean_cy, sd_cy = time_engine(n, p, "cython")

        rows.append({
            "n": n,
            "p": p,
            "numpy_mean": mean_np,
            "numpy_sd": sd_np,
            "cython_mean": mean_cy,
            "cython_sd": sd_cy,
            "speedup": mean_np / mean_cy
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = run_benchmark()
    df
