import sys
import subprocess
import numpy as np
from sympy import primerange, primitive_root, prime
from least_primitive_root import bulk, bulk_naive, least_primitive_root, a

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000000
fast = bulk(N)
naive = bulk_naive(N)
assert fast == naive
ps = list(primerange(2, N + 1))
assert len(ps) == len(fast)
for p, v in zip(ps[:40000], fast):
    assert primitive_root(p) == v, p
for p, v in zip(ps[-2000:], fast[-2000:]):
    assert primitive_root(p) == v, p
def order(p, g):
    x, k = 1, 0
    while True:
        x = x * g % p
        k += 1
        if x == 1:
            return k
for p, v in zip(ps[:400], fast):
    if p > 2:
        assert order(p, v) == p - 1
        assert all(order(p, g) < p - 1 for g in range(1, v) if g % p)
for p in [10**12 + 39, 10**15 + 37, 10**18 + 9, 2**61 - 1, 10**24 + 7]:
    r = least_primitive_root(p)
    assert pow(r, p - 1, p) == 1
    from sympy import factorint
    assert all(pow(r, (p - 1) // q, p) != 1 for q in factorint(p - 1))
    assert primitive_root(p) == r, p
subprocess.run(['gcc', '-O2', '-march=native', '-o', 'lpr', 'lpr.c', '-lm'], check=True)
subprocess.run(['./lpr', 'bulk', str(N), 'check.bin'], check=True, capture_output=True)
c = np.fromfile('check.bin', dtype=np.uint16)
assert list(c[:len(fast)]) == fast
for n in (1, 2, 10, 1000, 100000):
    assert a(n) == primitive_root(prime(n))
print('ok', len(fast))
