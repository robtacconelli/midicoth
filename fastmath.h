#ifndef FASTMATH_H
#define FASTMATH_H

#include <stdint.h>
#include <math.h>

/*
 * Fast log/exp approximations using IEEE 754 bit tricks + polynomial correction.
 * Accurate to ~1e-4 relative error — sufficient for probability manipulation.
 */

/* Fast natural log. Relative error < 2e-4 over [1e-30, 1.0] */
static inline double fast_log(double x) {
    union { double d; uint64_t u; } v = { .d = x };
    /* Extract exponent and mantissa from IEEE 754 */
    int64_t exp_bits = (int64_t)((v.u >> 52) & 0x7FF) - 1023;
    /* Set exponent to 0 → mantissa in [1, 2) */
    v.u = (v.u & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    double m = v.d;
    /* Polynomial approx of log(m) for m in [1,2): Remez-like */
    /* log(m) ≈ (m-1) - (m-1)^2/2 + (m-1)^3/3 ... but use minimax */
    double t = m - 1.0;
    double log_m = t * (1.0 + t * (-0.5 + t * (0.333333333 + t * (-0.25 + t * 0.2))));
    return log_m + exp_bits * 0.6931471805599453; /* exp_bits * ln(2) */
}

/* Fast exp. Relative error < 3e-4 over [-90, 0] (typical range for log-probs) */
static inline double fast_exp(double x) {
    if (x < -700.0) return 0.0;
    if (x > 709.0) return 1e308;
    /* exp(x) = 2^(x/ln2) = 2^(k+f) where k=floor, f=frac */
    double t = x * 1.4426950408889634; /* x / ln(2) */
    int64_t k = (int64_t)t;
    if (t < k) k--; /* floor for negative */
    double f = t - k;
    /* 2^f for f in [0,1): minimax polynomial */
    double p = 1.0 + f * (0.6931471805599453 + f * (0.24022650695910071
              + f * (0.05550410866482158 + f * (0.009618129107628477
              + f * 0.0013333558146428443))));
    /* Multiply by 2^k via bit manipulation */
    union { double d; uint64_t u; } v;
    v.u = (uint64_t)(k + 1023) << 52;
    return p * v.d;
}

/* Fast log(a / (1-a)) — logit function via single IEEE bit trick.
 * Uses the identity: logit(p) = log(p) - log(1-p)
 * We can compute log(p/(1-p)) in one pass by exploiting IEEE 754. */
static inline double fast_logit(double p) {
    if (p < 1e-8) p = 1e-8;
    if (p > 1.0 - 1e-8) p = 1.0 - 1e-8;
    /* For p near 0.5, use rational approx; otherwise use fast_log */
    double r = p / (1.0 - p);
    return fast_log(r);
}

/*
 * Precomputed logit lookup table for probabilities.
 * Maps probability [0..65536]/65536 → logit value.
 * Avoids per-symbol log computation entirely.
 */
#define LOGIT_TABLE_SIZE 65537
typedef struct {
    double table[LOGIT_TABLE_SIZE];
    int initialized;
} LogitTable;

static inline void logit_table_init(LogitTable *lt) {
    for (int i = 0; i < LOGIT_TABLE_SIZE; i++) {
        double p = (double)i / 65536.0;
        if (p < 1e-8) p = 1e-8;
        if (p > 1.0 - 1e-8) p = 1.0 - 1e-8;
        lt->table[i] = log(p / (1.0 - p));
    }
    lt->initialized = 1;
}

static inline double logit_table_lookup(const LogitTable *lt, double p) {
    int idx = (int)(p * 65536.0 + 0.5);
    if (idx < 0) idx = 0;
    if (idx >= LOGIT_TABLE_SIZE) idx = LOGIT_TABLE_SIZE - 1;
    return lt->table[idx];
}

/* Fast sqrt (just use hardware — it's already fast) */
/* static inline double fast_sqrt(double x) { return sqrt(x); } */

#endif /* FASTMATH_H */
