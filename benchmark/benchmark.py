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

    return float(np.mean(times)), float(np.std(times, ddof=1))

settings = [(200, 5), (500, 10), (1000, 20), (2000, 30)]
rows = []
for n, p in settings:
    mean_np, sd_np = time_engine(n, p, "numpy")
    mean_cy, sd_cy = time_engine(n, p, "cython")
    speedup = mean_np / mean_cy
    rows.append((n, p, mean_np, sd_np, mean_cy, sd_cy, speedup))

for r in rows:
    print(r)
