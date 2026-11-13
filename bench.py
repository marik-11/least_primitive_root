import time
import statistics
from sympy import primerange, primitive_root
from least_primitive_root import bulk, bulk_naive

REPEATS = 3

def timed(f, *a):
    t = time.perf_counter()
    r = f(*a)
    return r, time.perf_counter() - t

for n in (10**5, 10**6, 10**7):
    tf, tn, ts = [], [], []
    ps = list(primerange(2, n + 1)) if n <= 10**6 else None
    for _ in range(REPEATS):
        f, t = timed(bulk, n)
        tf.append(t)
        nv, t = timed(bulk_naive, n)
        tn.append(t)
        assert f == nv
        if ps is not None:
            t = time.perf_counter()
            s = [primitive_root(p) for p in ps]
            ts.append(time.perf_counter() - t)
            assert s == f
    k = len(f)
    mf, mn = statistics.median(tf), statistics.median(tn)
    line = f"N={n} primes={k} fast={mf/k*1e6:.2f}us naive={mn/k*1e6:.2f}us speedup={mn/mf:.2f}x"
    if ts:
        ms = statistics.median(ts)
        line += f" sympy={ms/k*1e6:.2f}us vs_sympy={ms/mf:.1f}x"
    print(line, flush=True)
