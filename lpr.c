#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <time.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef unsigned __int128 u128;

static int vmode;
static int recmode;

typedef struct {
    u64 p, pinv, one, r2;
    u32 pinv32, one32;
} mont;

static inline u64 redc(const mont *m, u128 t) {
    u64 tl = (u64)t, th = (u64)(t >> 64);
    u64 q = tl * m->pinv;
    u64 qh = (u64)(((u128)q * m->p) >> 64);
    u64 r = th - qh;
    if (th < qh) r += m->p;
    return r;
}

static inline u64 mmul(const mont *m, u64 a, u64 b) {
    return redc(m, (u128)a * b);
}

static void mont_init(mont *m, u64 p) {
    u64 x = p;
    for (int i = 0; i < 6; i++) x *= 2 - p * x;
    m->p = p;
    m->pinv = x;
    m->one = (0 - p) % p;
    m->r2 = 0;
    m->pinv32 = (u32)x;
    m->one32 = vmode ? (u32)((1ull << 32) % p) : 0;
}

static inline u64 to_mont(const mont *m, u64 x) {
    if (m->one < (1ull << 40) && x < (1u << 20)) return (x * m->one) % m->p;
    return (u64)(((u128)x * m->one) % m->p);
}

static u64 mpow2(const mont *m, u64 e) {
    u64 p = m->p;
    if (!e) return m->one;
    int top = 63 - __builtin_clzll(e);
    u64 r = m->one + m->one;
    if (r >= p) r -= p;
    for (int i = top - 1; i >= 0; i--) {
        r = mmul(m, r, r);
        if ((e >> i) & 1) {
            r += r;
            if (r >= p) r -= p;
        }
    }
    return r;
}

static u64 mpow(const mont *m, u64 b, u64 e) {
    u64 r = m->one;
    if (!e) return r;
    int top = 63 - __builtin_clzll(e);
    r = b;
    for (int i = top - 1; i >= 0; i--) {
        r = mmul(m, r, r);
        if ((e >> i) & 1) r = mmul(m, r, b);
    }
    return r;
}

static int jac(u64 a, u64 n) {
    int s = 1;
    a %= n;
    while (a) {
        while (!(a & 1)) {
            a >>= 1;
            u64 r = n & 7;
            if (r == 3 || r == 5) s = -s;
        }
        u64 t = a;
        a = n;
        n = t;
        if ((a & 3) == 3 && (n & 3) == 3) s = -s;
        a %= n;
    }
    return n == 1 ? s : 0;
}

static int any_one(const mont *m, u64 b, const u64 *q, int k) {
    if (k == 1) return b == m->one;
    int k1 = k / 2;
    u64 pa = 1, pb = 1;
    for (int i = 0; i < k1; i++) pa *= q[i];
    for (int i = k1; i < k; i++) pb *= q[i];
    if (any_one(m, mpow(m, b, pb), q, k1)) return 1;
    return any_one(m, mpow(m, b, pa), q + k1, k - k1);
}

static unsigned char *perfpow;
#define PPN 65536
#define CTN 128
static signed char *CT[CTN];

static void init_ct(void) {
    for (u64 g = 2; g < CTN; g++) {
        u64 md = 4 * g;
        CT[g] = calloc(md, 1);
        for (u64 r = 1; r < md; r += 2) CT[g][r] = (signed char)jac(g, r);
    }
}

static inline int chi(u64 g, u64 p) {
    if (g < CTN) return CT[g][p % (4 * g)];
    return jac(g, p);
}

static void init_perfpow(void) {
    perfpow = calloc(PPN, 1);
    for (u64 b = 2; b * b < PPN; b++) {
        u64 v = b * b;
        while (v < PPN) {
            perfpow[v] = 1;
            v *= b;
        }
    }
}

static u32 fast_root(u64 p, const u64 *oq, int k, int s_unused) {
    (void)s_unused;
    if (p == 2) return 1;
    u64 m = p - 1;
    u64 prod = 1;
    for (int i = 0; i < k; i++) prod *= oq[i];
    u64 E = m / prod;
    mont M;
    mont_init(&M, p);
    for (u64 g = 2;; g++) {
        if (g < PPN && perfpow[g]) continue;
        if (jac(g, p) != -1) continue;
        if (k == 0) return (u32)g;
        u64 b = g == 2 ? mpow2(&M, E) : mpow(&M, to_mont(&M, g), E);
        if (!any_one(&M, b, oq, k)) return (u32)g;
    }
}

static u32 naive_root(u64 p, const u64 *aq, int k) {
    if (p == 2) return 1;
    u64 m = p - 1;
    mont M;
    mont_init(&M, p);
    for (u64 g = 2;; g++) {
        u64 gm = to_mont(&M, g);
        int ok = 1;
        for (int i = 0; i < k; i++) {
            if (mpow(&M, gm, m / aq[i]) == M.one) {
                ok = 0;
                break;
            }
        }
        if (ok) return (u32)g;
    }
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

#ifndef SEGB
#define SEGB 17
#endif
#define SEG (1u << SEGB)
#define MAXF 16

static u32 *base;
static u32 nbase;

static void make_base(u64 lim) {
    unsigned char *c = calloc(lim + 1, 1);
    base = malloc(sizeof(u32) * (lim / 2 + 16));
    nbase = 0;
    for (u64 i = 2; i <= lim; i++) {
        if (!c[i]) {
            base[nbase++] = (u32)i;
            for (u64 j = i * i; j <= lim; j += i) c[j] = 1;
        }
    }
    free(c);
}

static void sieve_segment(u64 L, u64 H, unsigned char *comp) {
    u64 len = H - L;
    u64 *w = (u64 *)comp;
    u64 nw = len / 8;
    for (u64 i = 0; i < nw; i++) w[i] = 0x0001000100010001ull;
    for (u64 i = nw * 8; i < len; i += 2) comp[i] = 1;
    for (u32 i = 1; i < nbase; i++) {
        u64 r = base[i];
        if (r * r >= H) break;
        u64 st = (L + r - 1) / r * r;
        if (!(st & 1)) st += r;
        if (st < r * r) st = r * r;
        for (u64 x = st; x < H; x += 2 * r) comp[x - L] = 1;
    }
    if (L == 0) {
        comp[0] = 1;
        if (H > 1) comp[1] = 1;
        if (H > 2) comp[2] = 0;
    }
}

#define W 8

static void mpow_lanes(const mont *M, const u64 *b, const u64 *e, u64 *out, int top) {
    u64 r[W];
    for (int l = 0; l < W; l++) r[l] = M[l].one;
    for (int i = top - 1; i >= 0; i--) {
#pragma GCC unroll 8
        for (int l = 0; l < W; l++) {
            u64 x = mmul(&M[l], r[l], r[l]);
            u64 y = mmul(&M[l], x, b[l]);
            r[l] = ((e[l] >> i) & 1) ? y : x;
        }
    }
    memcpy(out, r, sizeof r);
}

#ifdef __AVX2__
#define VL 32

static inline __m256i vmul(__m256i a, __m256i b, __m256i p, __m256i po, __m256i pi, __m256i pio) {
    __m256i ao = _mm256_srli_epi64(a, 32), bo = _mm256_srli_epi64(b, 32);
    __m256i te = _mm256_mul_epu32(a, b);
    __m256i to = _mm256_mul_epu32(ao, bo);
    __m256i me = _mm256_mul_epu32(te, pi);
    __m256i mo = _mm256_mul_epu32(to, pio);
    __m256i pe = _mm256_mul_epu32(me, p);
    __m256i qo = _mm256_mul_epu32(mo, po);
    __m256i th = _mm256_blend_epi32(_mm256_srli_epi64(te, 32), to, 0xAA);
    __m256i mh = _mm256_blend_epi32(_mm256_srli_epi64(pe, 32), qo, 0xAA);
    __m256i r = _mm256_sub_epi32(th, mh);
    return _mm256_min_epu32(r, _mm256_add_epi32(r, p));
}

static void mpow_vec(const u32 *P, const u32 *PI, const u32 *ONE, const u32 *B, const u32 *E, u32 *OUT, int top) {
    __m256i p[4], po[4], pi[4], pio[4], r[4], b[4], e[4];
    for (int v = 0; v < 4; v++) {
        p[v] = _mm256_loadu_si256((const __m256i *)(P + 8 * v));
        pi[v] = _mm256_loadu_si256((const __m256i *)(PI + 8 * v));
        r[v] = _mm256_loadu_si256((const __m256i *)(ONE + 8 * v));
        b[v] = _mm256_loadu_si256((const __m256i *)(B + 8 * v));
        e[v] = _mm256_loadu_si256((const __m256i *)(E + 8 * v));
        po[v] = _mm256_srli_epi64(p[v], 32);
        pio[v] = _mm256_srli_epi64(pi[v], 32);
    }
    for (int i = top - 1; i >= 0; i--) {
        __m256i bm = _mm256_set1_epi32((int)(1u << i));
#pragma GCC unroll 4
        for (int v = 0; v < 4; v++) {
            __m256i x = vmul(r[v], r[v], p[v], po[v], pi[v], pio[v]);
            __m256i y = vmul(x, b[v], p[v], po[v], pi[v], pio[v]);
            __m256i mk = _mm256_cmpeq_epi32(_mm256_and_si256(e[v], bm), bm);
            r[v] = _mm256_blendv_epi8(x, y, mk);
        }
    }
    for (int v = 0; v < 4; v++) _mm256_storeu_si256((__m256i *)(OUT + 8 * v), r[v]);
}
#endif

typedef struct {
    u32 j;
    u16 lo, hi;
    u64 base, e;
} job;

static u64 *Ev;
static u32 *gc, *todo, *todo2, *tasks;
static unsigned char *kk, *fl, *okc;
static mont *Mj;
static job *JA, *JB, *JS;

static inline u64 qat(u32 j, int t, const u32 *fac, const unsigned char *nf, const u64 *rem) {
    return t < nf[j] ? (u64)fac[(size_t)j * MAXF + t] : rem[j];
}

static void run_jobs(u32 n) {
    u32 cntb[65];
    memset(cntb, 0, sizeof cntb);
    for (u32 i = 0; i < n; i++) cntb[64 - __builtin_clzll(JA[i].e)]++;
    u32 acc = 0;
    for (int b = 0; b < 65; b++) {
        u32 c = cntb[b];
        cntb[b] = acc;
        acc += c;
    }
    for (u32 i = 0; i < n; i++) JS[cntb[64 - __builtin_clzll(JA[i].e)]++] = JA[i];
#ifdef __AVX2__
    if (vmode) {
        for (u32 s = 0; s < n; s += VL) {
            u32 m = n - s < VL ? n - s : VL;
            u32 P[VL], PI[VL], ONE[VL], B[VL], E[VL], OUT[VL];
            for (u32 l = 0; l < VL; l++) {
                const job *q = &JS[s + (l < m ? l : m - 1)];
                const mont *mm = &Mj[q->j];
                P[l] = (u32)mm->p;
                PI[l] = mm->pinv32;
                ONE[l] = mm->one32;
                B[l] = (u32)q->base;
                E[l] = (u32)q->e;
            }
            int top = 32 - __builtin_clz(E[m - 1]);
            mpow_vec(P, PI, ONE, B, E, OUT, top);
            for (u32 l = 0; l < m; l++) JS[s + l].base = OUT[l];
        }
        return;
    }
#endif
    for (u32 s = 0; s < n; s += W) {
        u32 m = n - s < W ? n - s : W;
        mont M[W];
        u64 b[W], e[W], out[W];
        for (u32 l = 0; l < W; l++) {
            const job *q = &JS[s + (l < m ? l : m - 1)];
            M[l] = Mj[q->j];
            b[l] = q->base;
            e[l] = q->e;
        }
        int top = 64 - __builtin_clzll(e[m - 1]);
        mpow_lanes(M, b, e, out, top);
        for (u32 l = 0; l < m; l++) JS[s + l].base = out[l];
    }
}

static void fast_segment(u32 np, const u64 *pv, const u32 *fac, const unsigned char *nf, const u64 *rem, u16 *res) {
    u32 ntodo = 0;
    for (u32 j = 0; j < np; j++) {
        u64 p = pv[j];
        if (p == 2) {
            res[j] = 1;
            continue;
        }
        int k = nf[j];
        u64 prod = 1;
        for (int t = 0; t < k; t++) prod *= fac[(size_t)j * MAXF + t];
        if (rem[j] > 1) {
            prod *= rem[j];
            k++;
        }
        kk[j] = (unsigned char)k;
        Ev[j] = (p - 1) / prod;
        gc[j] = 2;
        mont_init(&Mj[j], p);
        todo[ntodo++] = j;
    }
    while (ntodo) {
        u32 nt = 0;
        for (u32 i = 0; i < ntodo; i++) {
            u32 j = todo[i];
            u64 p = pv[j];
            u64 g = gc[j];
            for (;; g++) {
                if (g < PPN && perfpow[g]) continue;
                if (chi(g, p) != -1) continue;
                break;
            }
            if (kk[j] == 0) {
                res[j] = (u16)g;
                continue;
            }
            gc[j] = (u32)g;
            todo[nt++] = j;
        }
        u32 nj = 0;
        if (vmode)
        for (u32 i = 0; i < nt; i++) {
            u32 j = todo[i];
            fl[j] = 0;
            okc[j] = 0;
            u64 p = pv[j];
            u64 bs = vmode ? (u64)(((u64)gc[j] << 32) % p) : to_mont(&Mj[j], gc[j]);
            for (int t = 0; t < kk[j]; t++) {
                job q;
                q.j = j;
                q.lo = (u16)t;
                q.hi = (u16)(t + 1);
                q.base = bs;
                u64 qq = qat(j, t, fac, nf, rem);
                q.e = vmode ? (u64)((u32)(p - 1) / (u32)qq) : (p - 1) / qq;
                JA[nj++] = q;
            }
        }
        if (!vmode)
        for (u32 i = 0; i < nt; i++) {
            u32 j = todo[i];
            fl[j] = 0;
            okc[j] = 0;
            job q;
            q.j = j;
            q.lo = 0;
            q.hi = kk[j];
            q.base = vmode ? (u64)(((u64)gc[j] << 32) % Mj[j].p) : to_mont(&Mj[j], gc[j]);
            q.e = Ev[j];
            JA[nj++] = q;
        }
        for (u32 i = 0; i < nt; i++) tasks[i] = todo[i];
        while (nj) {
            run_jobs(nj);
            u32 nn = 0;
            for (u32 i = 0; i < nj; i++) {
                const job *q = &JS[i];
                u32 j = q->j;
                if (fl[j]) continue;
                if (q->hi - q->lo == 1) {
                    if (q->base == (vmode ? Mj[j].one32 : Mj[j].one)) fl[j] = 1;
                    else okc[j]++;
                } else {
                    int lo = q->lo, hi = q->hi, mid = lo + (hi - lo) / 2;
                    u64 pa = 1, pb = 1;
                    for (int t = lo; t < mid; t++) pa *= qat(j, t, fac, nf, rem);
                    for (int t = mid; t < hi; t++) pb *= qat(j, t, fac, nf, rem);
                    job x;
                    x.j = j;
                    x.lo = (u16)lo;
                    x.hi = (u16)mid;
                    x.base = q->base;
                    x.e = pb;
                    JB[nn++] = x;
                    x.lo = (u16)mid;
                    x.hi = (u16)hi;
                    x.e = pa;
                    JB[nn++] = x;
                }
            }
            job *tmp = JA;
            JA = JB;
            JB = tmp;
            nj = nn;
        }
        u32 nn2 = 0;
        for (u32 i = 0; i < nt; i++) {
            u32 j = tasks[i];
            if (fl[j]) {
                gc[j]++;
                todo2[nn2++] = j;
            } else {
                res[j] = (u16)gc[j];
            }
        }
        u32 *tmp = todo;
        todo = todo2;
        todo2 = tmp;
        ntodo = nn2;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) return 1;
    const char *mode = argv[1];
    u64 N = strtoull(argv[2], 0, 10);
    const char *outf = argc > 3 ? argv[3] : 0;
    int lanes = !strcmp(mode, "bulk");
#ifdef __AVX2__
    vmode = !strcmp(mode, "bulk") && N < (1ull << 31);
#endif
    int fast = !strcmp(mode, "scalar") || lanes;
    int trial = !strcmp(mode, "trial");
    int naive = !strcmp(mode, "naive") || trial;
    int sieveonly = !strcmp(mode, "sieve");
    recmode = getenv("LPR_REC") != 0;
    init_perfpow();
    init_ct();
    u64 sq = (u64)sqrt((double)N) + 2;
    make_base(sq);
    unsigned char *comp = malloc(SEG);
    u32 *pidx = malloc(sizeof(u32) * SEG);
    u64 *pv = malloc(sizeof(u64) * SEG);
    u64 *rem = malloc(sizeof(u64) * SEG);
    u32 *fac = malloc(sizeof(u32) * (size_t)MAXF * SEG / 4);
    unsigned char *nf = malloc(SEG);
    Ev = malloc(sizeof(u64) * SEG);
    gc = malloc(sizeof(u32) * SEG);
    todo = malloc(sizeof(u32) * SEG);
    todo2 = malloc(sizeof(u32) * SEG);
    tasks = malloc(sizeof(u32) * SEG);
    kk = malloc(SEG);
    fl = malloc(SEG);
    okc = malloc(SEG);
    Mj = malloc(sizeof(mont) * SEG);
    JA = malloc(sizeof(job) * SEG * 2);
    JB = malloc(sizeof(job) * SEG * 2);
    JS = malloc(sizeof(job) * SEG * 2);
    u16 *res = malloc(sizeof(u16) * SEG);
    u16 *out = outf ? malloc(sizeof(u16) * (size_t)(N / 2 + 16)) : 0;
    u64 cnt = 0, sum = 0, mx = 0, mxp = 0;
    double t0 = now();
    double tsv = 0, twk = 0, tts = 0, ta;
    for (u64 L = 0; L <= N; L += SEG) {
        u64 H = L + SEG;
        if (H > N + 1) H = N + 1;
        ta = now();
        sieve_segment(L, H, comp);
        u32 np = 0;
        for (u64 x = (L == 0 ? 2 : L + 1); x < H; x += (x == 2 ? 1 : 2)) {
            if (!comp[x - L]) {
                pidx[x - L] = np;
                pv[np] = x;
                rem[np] = x > 2 ? ((x - 1) >> __builtin_ctzll(x - 1)) : 1;
                nf[np] = 0;
                np++;
            }
        }
        tsv += now() - ta;
        ta = now();
        if (sieveonly) {
            cnt += np;
            continue;
        }
        if (np > SEG / 4) {
            fprintf(stderr, "overflow\n");
            return 2;
        }
        for (u32 i = 1; !trial && i < nbase; i++) {
            u64 r = base[i];
            if (r > H) break;
            u64 r2 = 2 * r;
            u64 off = (r2 + 1 - L % r2) % r2;
            for (u64 x = L + off; x < H; x += r2) {
                if (comp[x - L] || x <= r) continue;
                u32 j = pidx[x - L];
                u64 v = rem[j];
                if (v % r == 0) {
                    do v /= r; while (v % r == 0);
                    rem[j] = v;
                    fac[(size_t)j * MAXF + nf[j]++] = (u32)r;
                }
            }
        }
        twk += now() - ta;
        ta = now();
        if (lanes) {
            fast_segment(np, pv, fac, nf, rem, res);
            for (u32 j = 0; j < np; j++) {
                u32 g = res[j];
                if (out) out[cnt] = (u16)g;
                cnt++;
                sum += g;
                if (g > mx) {
                    mx = g;
                    mxp = pv[j];
                    if (recmode) fprintf(stderr, "rec %llu %u\n", (unsigned long long)mxp, g);
                }
            }
            continue;
        }
        for (u32 j = 0; j < np; j++) {
            u64 p = pv[j];
            u64 oq[MAXF + 1];
            int k = 0;
            for (int t = 0; t < nf[j]; t++) oq[k++] = fac[(size_t)j * MAXF + t];
            if (rem[j] > 1) oq[k++] = rem[j];
            u32 g;
            if (fast) {
                g = fast_root(p, oq, k, 0);
            } else if (trial) {
                u64 aq[MAXF + 2];
                int ka = 0;
                u64 mm = p - 1;
                for (u32 i = 0; i < nbase; i++) {
                    u64 r = base[i];
                    if (r * r > mm) break;
                    if (mm % r == 0) {
                        aq[ka++] = r;
                        do mm /= r; while (mm % r == 0);
                    }
                }
                if (mm > 1) aq[ka++] = mm;
                g = naive_root(p, aq, ka);
            } else if (naive) {
                u64 aq[MAXF + 2];
                int ka = 0;
                if (p > 2) aq[ka++] = 2;
                for (int t = 0; t < k; t++) aq[ka++] = oq[t];
                g = naive_root(p, aq, ka);
            } else {
                g = 0;
            }
            if (out) out[cnt] = (u16)g;
            cnt++;
            sum += g;
            if (g > mx) {
                mx = g;
                mxp = p;
                if (recmode) fprintf(stderr, "rec %llu %u\n", (unsigned long long)mxp, g);
            }
        }
    }
    double t1 = now();
    fprintf(stderr, "sieve %.1f walk %.1f test %.1f ns/prime\n", tsv * 1e9 / cnt, twk * 1e9 / cnt, (t1 - t0 - tsv - twk) * 1e9 / cnt);
    printf("mode=%s N=%llu primes=%llu sum=%llu max=%llu at p=%llu time=%.3f ns/prime=%.1f\n", mode, (unsigned long long)N, (unsigned long long)cnt, (unsigned long long)sum, (unsigned long long)mx, (unsigned long long)mxp, t1 - t0, (t1 - t0) * 1e9 / (double)cnt);
    if (out) {
        FILE *f = fopen(outf, "wb");
        fwrite(out, sizeof(u16), cnt, f);
        fclose(f);
    }
    return 0;
}
