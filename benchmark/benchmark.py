# benchmark.py (or simulation.py)

import time
import numpy as np
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

        speedup = mean_np / mean_cy

        rows.append((n, p, mean_np, sd_np, mean_cy, sd_cy, speedup))

    print("\n=== Linear Model Benchmark ===\n")

    header = (
        f"{'n':>6} {'p':>6} | "
        f"{'numpy_mean':>12} {'numpy_sd':>12} | "
        f"{'cython_mean':>12} {'cython_sd':>12} | "
        f"{'speedup':>10}"
    )
    print(header)
    print("-" * len(header))

    for n, p, mnp, snp, mcy, scy, sp in rows:
        print(
            f"{n:6d} {p:6d} | "
            f"{mnp:12.6f} {snp:12.6f} | "
            f"{mcy:12.6f} {scy:12.6f} | "
            f"{sp:10.3f}"
        )


if __name__ == "__main__":
    run_benchmark()
