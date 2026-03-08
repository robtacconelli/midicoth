#ifndef TWEEDIE_H
#define TWEEDIE_H

/*
 * Binary Tree Tweedie Denoiser — score-based reverse diffusion.
 *
 * Forward process (PPM Jeffreys prior):
 *   p̂(s) = (n·q(s) + 0.5) / (n + 128) = (1-γ)q(s) + γ·u(s)
 *   where γ = 128/(n+128) is the noise level, u = 1/256 uniform.
 *
 * Tweedie's formula gives the optimal denoiser:
 *   θ̂ = p̂ + σ² · s(p̂)
 *   where s(p̂) = ∇ log m(p̂) is the score of the marginal density.
 *
 * The score is estimated empirically via calibration tables that track
 * the additive correction δ = E[θ|p̂] - E[p̂] = hit_rate - avg_pred.
 * This δ equals σ²·s(p̂) — the full Tweedie correction term.
 *
 * Binary tree decomposition: 256-way → 8 binary decisions (MSB to LSB).
 * Multi-step: K=3 denoising steps with independent score tables.
 * Calibration context: (step, bit_context, order, shape, confidence, prob_bin)
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "fastmath.h"

#define TWD_NSYM 256

/* Number of reverse diffusion steps */
#define TWD_STEPS 3

/* Binary tree: 8 levels for 256 symbols */
#define TWD_N_LEVELS 8

/* 255 internal nodes: 1 + 2 + 4 + ... + 128 */
#define TWD_N_NODES 255

/* Bit context: encodes level + parent bit values. 27 total. */
#define TWD_N_BCTX 27

/* Calibration dimensions */
#define TWD_N_ORD    3    /* order groups: {-1,0,1}, {2,3}, {4+} */
#define TWD_N_SHAPE  4    /* distribution shape bins by max_p */
#define TWD_N_CONF   8    /* confidence bins (log-spaced) */
#define TWD_N_PROB  20    /* binary probability bins (logit-spaced) */

/* Smoothing pseudo-observations per bucket */
#define TWD_PRIOR_WEIGHT 32.0

/* Logit range for binary probability mapping */
#define TWD_LOGIT_RANGE 8.0

typedef struct {
    double sum_pred;   /* sum of predicted P(right) */
    double hits;       /* times true symbol went right */
    double total;      /* total observations */
} TwdCalibEntry;

typedef struct {
    /* Calibration table: [step][bctx][order][shape][conf][prob_bin]
     * Total entries: 3 × 27 × 3 × 4 × 8 × 20 = 155,520
     * Memory: 155,520 × 24 bytes = 3.6 MB */
    TwdCalibEntry table[TWD_STEPS][TWD_N_BCTX][TWD_N_ORD][TWD_N_SHAPE][TWD_N_CONF][TWD_N_PROB];

    /* Cached from denoise, reused by update */
    double cached_p_right[TWD_STEPS][TWD_N_NODES];
    int    cached_prob_bin[TWD_STEPS][TWD_N_NODES];
    int    cached_bctx[TWD_STEPS][TWD_N_NODES];
    int    cached_ord;
    int    cached_shape;
    int    cached_conf;
} TweedieDenoiser;

/* ── Bucket mapping functions ── */

static inline int twd_order_group(int ppm_order) {
    if (ppm_order <= 1) return 0;
    if (ppm_order <= 3) return 1;
    return 2;
}

static inline int twd_shape_bin(double max_p) {
    if (max_p < 0.05) return 0;   /* very flat */
    if (max_p < 0.15) return 1;   /* moderately flat */
    if (max_p < 0.40) return 2;   /* moderate peak */
    return 3;                      /* peaked */
}

static inline int twd_conf_bin(double confidence) {
    if (confidence < 4.0) return 0;
    int bin = (int)(fast_log(confidence) * (1.0 / 1.3862943611198906));
    if (bin < 0) bin = 0;
    if (bin > TWD_N_CONF - 1) bin = TWD_N_CONF - 1;
    return bin;
}

/* Binary probability bin: logit-spaced in [-8, 8]. */
static inline int twd_prob_bin(double p) {
    if (p < 1e-8) p = 1e-8;
    if (p > 1.0 - 1e-8) p = 1.0 - 1e-8;
    double logit = fast_log(p / (1.0 - p));
    int bin = (int)((logit + TWD_LOGIT_RANGE) / (2.0 * TWD_LOGIT_RANGE) * TWD_N_PROB);
    if (bin < 0) bin = 0;
    if (bin > TWD_N_PROB - 1) bin = TWD_N_PROB - 1;
    return bin;
}

/* Bin center for prior initialization */
static inline double twd_bin_center(int bin) {
    double logit = ((bin + 0.5) / TWD_N_PROB) * 2.0 * TWD_LOGIT_RANGE - TWD_LOGIT_RANGE;
    return 1.0 / (1.0 + fast_exp(-logit));
}

/* Bit context: maps (level, node_index_at_level) → context ID 0..26. */
static inline int twd_bit_context(int level, int node_at_level) {
    if (level == 0) return 0;
    if (level == 1) return 1 + node_at_level;              /* 2 contexts */
    if (level == 2) return 3 + node_at_level;              /* 4 contexts */
    /* Levels 3-7: hash node_at_level into 4 groups */
    int group = (node_at_level * 2654435761U) >> 30;       /* hash → 0..3 */
    return 7 + (level - 3) * 4 + group;
}

/* ── Initialization ── */

static inline void tweedie_init(TweedieDenoiser *td) {
    memset(td, 0, sizeof(*td));

    for (int t = 0; t < TWD_STEPS; t++)
        for (int b = 0; b < TWD_N_BCTX; b++)
            for (int o = 0; o < TWD_N_ORD; o++)
                for (int s = 0; s < TWD_N_SHAPE; s++)
                    for (int c = 0; c < TWD_N_CONF; c++)
                        for (int p = 0; p < TWD_N_PROB; p++) {
                            double center = twd_bin_center(p);
                            td->table[t][b][o][s][c][p].sum_pred = center * TWD_PRIOR_WEIGHT;
                            td->table[t][b][o][s][c][p].hits     = center * TWD_PRIOR_WEIGHT;
                            td->table[t][b][o][s][c][p].total    = TWD_PRIOR_WEIGHT;
                        }
}

/* ── Denoise: multi-step Tweedie reverse diffusion ──
 *
 * Additive Tweedie correction: p' = p + δ
 * where δ = hits/total - sum_pred/total estimates the Tweedie term σ²·s(p̂).
 *
 * This is the nonparametric Tweedie estimator: within each calibration bin,
 * the empirical hit rate is the posterior mean E[θ|p̂], and the additive
 * correction δ = E[θ|p̂] - E[p̂] equals σ²·∇log m(p̂). */

static inline void tweedie_denoise(TweedieDenoiser *td, double *probs,
                                     int ppm_order, double confidence) {
    int og = twd_order_group(ppm_order);
    int cb = twd_conf_bin(confidence);

    /* Shape from the 256-way distribution (before any correction) */
    double max_p = 0.0;
    for (int i = 0; i < TWD_NSYM; i++)
        if (probs[i] > max_p) max_p = probs[i];
    int sb = twd_shape_bin(max_p);

    td->cached_ord = og;
    td->cached_shape = sb;
    td->cached_conf = cb;

    double stree[512];
    double scale[512];

    for (int step = 0; step < TWD_STEPS; step++) {

        /* 1. Build sum tree bottom-up */
        for (int i = 0; i < TWD_NSYM; i++)
            stree[TWD_NSYM + i] = probs[i];
        for (int i = TWD_NSYM - 1; i >= 1; i--)
            stree[i] = stree[2 * i] + stree[2 * i + 1];

        /* 2. Process all nodes: compute P(right), apply Tweedie correction */
        scale[1] = 1.0;

        for (int level = 0; level < TWD_N_LEVELS; level++) {
            int level_start = 1 << level;
            int level_end   = 1 << (level + 1);

            for (int ni = level_start; ni < level_end; ni++) {
                double node_total = stree[ni];
                int node_id = ni - 1;
                int node_at_level = ni - level_start;

                if (node_total < 1e-15) {
                    scale[2 * ni]     = scale[ni];
                    scale[2 * ni + 1] = scale[ni];
                    td->cached_p_right[step][node_id] = 0.5;
                    td->cached_prob_bin[step][node_id] = twd_prob_bin(0.5);
                    td->cached_bctx[step][node_id] = twd_bit_context(level, node_at_level);
                    continue;
                }

                double sum_right = stree[2 * ni + 1];
                double p_right = sum_right / node_total;
                if (p_right < 1e-8) p_right = 1e-8;
                if (p_right > 1.0 - 1e-8) p_right = 1.0 - 1e-8;

                int bctx = twd_bit_context(level, node_at_level);
                int pbin = twd_prob_bin(p_right);
                td->cached_p_right[step][node_id] = p_right;
                td->cached_prob_bin[step][node_id] = pbin;
                td->cached_bctx[step][node_id] = bctx;

                /* Tweedie additive correction: δ = E[θ|p̂] - E[p̂] */
                TwdCalibEntry *e = &td->table[step][bctx][og][sb][cb][pbin];
                double avg_pred = e->sum_pred / e->total;
                double emp_rate = e->hits / e->total;
                double delta = emp_rate - avg_pred;

                double p_right_corr = p_right + delta;
                if (p_right_corr < 1e-8)       p_right_corr = 1e-8;
                if (p_right_corr > 1.0 - 1e-8) p_right_corr = 1.0 - 1e-8;

                double sl = (1.0 - p_right_corr) / (1.0 - p_right);
                double sr = p_right_corr / p_right;
                scale[2 * ni]     = scale[ni] * sl;
                scale[2 * ni + 1] = scale[ni] * sr;
            }
        }

        /* 3. Apply accumulated leaf scales */
        for (int i = 0; i < TWD_NSYM; i++)
            probs[i] *= scale[TWD_NSYM + i];

        /* 4. Renormalize */
        double sum = 0.0;
        for (int i = 0; i < TWD_NSYM; i++) {
            if (probs[i] < 1e-10) probs[i] = 1e-10;
            sum += probs[i];
        }
        double inv = 1.0 / sum;
        for (int i = 0; i < TWD_NSYM; i++)
            probs[i] *= inv;

        /* Recompute shape after correction for next step */
        max_p = 0.0;
        for (int i = 0; i < TWD_NSYM; i++)
            if (probs[i] > max_p) max_p = probs[i];
        sb = twd_shape_bin(max_p);
    }
}

/* ── Update ── */

static inline void tweedie_update(TweedieDenoiser *td, uint8_t true_symbol) {
    int og = td->cached_ord;
    int sb = td->cached_shape;
    int cb = td->cached_conf;

    for (int step = 0; step < TWD_STEPS; step++) {
        for (int level = 0; level < TWD_N_LEVELS; level++) {
            int block_size = TWD_NSYM >> level;
            int half       = block_size >> 1;

            int node_at_level = true_symbol / block_size;
            int start = node_at_level * block_size;
            int mid   = start + half;
            int went_right = (true_symbol >= mid) ? 1 : 0;

            int node_id = (1 << level) - 1 + node_at_level;
            int pbin = td->cached_prob_bin[step][node_id];
            int bctx = td->cached_bctx[step][node_id];

            TwdCalibEntry *e = &td->table[step][bctx][og][sb][cb][pbin];
            e->sum_pred += td->cached_p_right[step][node_id];
            e->total    += 1.0;
            if (went_right)
                e->hits += 1.0;
        }
    }
}

#endif /* TWEEDIE_H */
