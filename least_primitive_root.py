import numpy as np
from math import gcd, isqrt

PERFECT_POWER = bytearray(65536)
for _b in range(2, 256):
    _v = _b * _b
    while _v < 65536:
        PERFECT_POWER[_v] = 1
        _v *= _b

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def jacobi(a, n):
    s = 1
    a %= n
    while a:
        while not a & 1:
            a >>= 1
            if n & 7 in (3, 5):
                s = -s
        a, n = n, a
        if a & 3 == 3 and n & 3 == 3:
            s = -s
        a %= n
    return s if n == 1 else 0


def is_prime(n):
    if n < 2:
        return False
    for q in SMALL_PRIMES:
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while not d & 1:
        d >>= 1
        s += 1
    for a in SMALL_PRIMES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def rho(n):
    if n % 2 == 0:
        return 2
    c = 1
    while True:
        x = y = 2
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = gcd(abs(x - y), n)
        if d != n:
            return d
        c += 1


def distinct_odd_factors(m):
    m >>= (m & -m).bit_length() - 1
    out = set()
    for q in range(3, 1000, 2):
        if q * q > m:
            break
        if m % q == 0:
            out.add(q)
            while m % q == 0:
                m //= q
    stack = [m] if m > 1 else []
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if is_prime(x):
            out.add(x)
            continue
        d = rho(x)
        stack.append(d)
        stack.append(x // d)
    return sorted(out)


def any_one(b, qs, p):
    if len(qs) == 1:
        return b == 1
    h = len(qs) // 2
    left, right = qs[:h], qs[h:]
    pl = 1
    for q in left:
        pl *= q
    pr = 1
    for q in right:
        pr *= q
    return any_one(pow(b, pr, p), left, p) or any_one(pow(b, pl, p), right, p)


def root_from_factors(p, qs):
    if p == 2:
        return 1
    prod = 1
    for q in qs:
        prod *= q
    e = (p - 1) // prod
    g = 2
    while True:
        if not (g < 65536 and PERFECT_POWER[g]) and jacobi(g, p) == -1:
            if not qs or not any_one(pow(g, e, p), qs, p):
                return g
        g += 1


def least_primitive_root(p):
    if p == 2:
        return 1
    return root_from_factors(p, distinct_odd_factors(p - 1))


def spf_table(n):
    spf = np.zeros(n + 1, dtype=np.uint32)
    for r in range(2, isqrt(n) + 1):
        if spf[r] == 0:
            seg = spf[r * r::r]
            seg[seg == 0] = r
    return spf


def odd_factors_spf(m, spf):
    m >>= (m & -m).bit_length() - 1
    qs = []
    while m > 1:
        q = int(spf[m]) or m
        qs.append(q)
        m //= q
        while m % q == 0:
            m //= q
    return qs


def bulk(n):
    spf = spf_table(n)
    ps = np.flatnonzero(spf[2:] == 0) + 2
    out = []
    for p in ps.tolist():
        out.append(root_from_factors(p, odd_factors_spf(p - 1, spf)) if p > 2 else 1)
    return out


def naive_root(p, spf):
    if p == 2:
        return 1
    m = p - 1
    qs = [2] + odd_factors_spf(m, spf)
    g = 2
    while True:
        if all(pow(g, m // q, p) != 1 for q in qs):
            return g
        g += 1


def bulk_naive(n):
    spf = spf_table(n)
    ps = np.flatnonzero(spf[2:] == 0) + 2
    return [naive_root(p, spf) for p in ps.tolist()]


def a(n):
    if n < 6:
        limit = 15
    else:
        from math import log
        limit = int(n * (log(n) + log(log(n)))) + 10
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    p = int(np.flatnonzero(sieve)[n - 1])
    return least_primitive_root(p)
