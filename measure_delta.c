/*
 * Measure mean |delta| vs noise level (confidence) for the
 * delta-vs-gamma table in the paper.
 *
 * Runs the full pipeline (PPM+Match+Word+HCtx+Tweedie) and
 * instruments the Tweedie denoise to collect per-step, per-confidence
 * delta statistics.
 *
 * Usage: ./measure_delta <input_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fastmath.h"
#include "arith.h"
#include "ppm.h"
#include "match.h"
#include "word.h"
#include "highctx.h"

/* We need access to Tweedie internals, so include it but also
 * define instrumentation hooks */
#include "tweedie.h"

/* Accumulate |delta| by [step][conf_bin] */
#define N_CONF_REPORT 4
static double delta_sum[TWD_STEPS][N_CONF_REPORT];
static double delta_count[TWD_STEPS][N_CONF_REPORT];

/* Map raw confidence to our 4 reporting bins:
 *   bin 0: C ~ 128  (gamma ~ 0.500)
 *   bin 1: C ~ 512  (gamma ~ 0.200)
 *   bin 2: C ~ 2048 (gamma ~ 0.059)
 *   bin 3: C ~ 8192 (gamma ~ 0.015) */
static int conf_report_bin(double confidence) {
    if (confidence < 256.0)  return 0;
    if (confidence < 1024.0) return 1;
    if (confidence < 4096.0) return 2;
    return 3;
}

/* Instrumented denoise that collects delta stats */
static void tweedie_denoise_instrumented(TweedieDenoiser *td, double *probs,
                                          int ppm_order, double confidence) {
    int og = twd_order_group(ppm_order);
    int cb = twd_conf_bin(confidence);
    int crb = conf_report_bin(confidence);

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
        for (int i = 0; i < TWD_NSYM; i++)
            stree[TWD_NSYM + i] = probs[i];
        for (int i = TWD_NSYM - 1; i >= 1; i--)
            stree[i] = stree[2 * i] + stree[2 * i + 1];

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

                TwdCalibEntry *e = &td->table[step][bctx][og][sb][cb][pbin];
                double avg_pred = e->sum_pred / e->total;
                double emp_rate = e->hits / e->total;
                double delta = emp_rate - avg_pred;

                /* Apply same shrinkage as production code */
                double var_err = e->sum_sq_err / e->total;
                if (e->total > 10.0 && var_err > 1e-10) {
                    double snr = delta * delta * e->total / var_err;
                    double shrink = (snr > 4.0) ? 1.0 : snr / 4.0;
                    delta *= shrink;
                } else {
                    delta = 0.0;
                }

                /* Collect stats: weight by node probability mass */
                double weight = node_total / stree[1];
                delta_sum[step][crb]   += fabs(delta) * weight;
                delta_count[step][crb] += weight;

                double p_right_corr = p_right + delta;
                if (p_right_corr < 1e-8)       p_right_corr = 1e-8;
                if (p_right_corr > 1.0 - 1e-8) p_right_corr = 1.0 - 1e-8;

                double sl = (1.0 - p_right_corr) / (1.0 - p_right);
                double sr = p_right_corr / p_right;
                scale[2 * ni]     = scale[ni] * sl;
                scale[2 * ni + 1] = scale[ni] * sr;
            }
        }

        for (int i = 0; i < TWD_NSYM; i++)
            probs[i] *= scale[TWD_NSYM + i];

        double sum = 0.0;
        for (int i = 0; i < TWD_NSYM; i++) {
            if (probs[i] < 1e-10) probs[i] = 1e-10;
            sum += probs[i];
        }
        double inv = 1.0 / sum;
        for (int i = 0; i < TWD_NSYM; i++)
            probs[i] *= inv;

        max_p = 0.0;
        for (int i = 0; i < TWD_NSYM; i++)
            if (probs[i] > max_p) max_p = probs[i];
        sb = twd_shape_bin(max_p);
    }
}

static void my_clamp_normalize(double *p) {
    double sum = 0;
    for (int i = 0; i < 256; i++) {
        if (p[i] < 1e-10) p[i] = 1e-10;
        sum += p[i];
    }
    double inv = 1.0 / sum;
    for (int i = 0; i < 256; i++) p[i] *= inv;
}

static void my_blend_match(double *probs, int match_byte, double match_conf) {
    if (match_byte < 0 || match_conf < 0.01) return;
    double w = match_conf * 0.85;
    if (w > 0.95) w = 0.95;
    for (int i = 0; i < 256; i++)
        probs[i] *= (1.0 - w);
    probs[match_byte] += w;
}

static void my_blend_word(double *probs, double *wprobs, double wconf) {
    double w = wconf * 0.35;
    if (w > 0.45) w = 0.45;
    for (int i = 0; i < 256; i++)
        probs[i] = (1.0 - w) * probs[i] + w * wprobs[i];
}

static void my_blend_hctx(double *probs, double *hprobs, double hconf) {
    double w = hconf * 2.0;
    if (w > 0.60) w = 0.60;
    for (int i = 0; i < 256; i++)
        probs[i] = (1.0 - w) * probs[i] + w * hprobs[i];
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = malloc(len);
    fread(data, 1, len, f);
    fclose(f);

    PPMModel ppm;
    ppm_init(&ppm);

    MatchModel match;
    match_init(&match);

    WordModel word;
    word_init(&word);

    HighCtxModel hctx;
    highctx_init(&hctx);

    TweedieDenoiser *twd = malloc(sizeof(TweedieDenoiser));
    tweedie_init(twd);

    memset(delta_sum, 0, sizeof(delta_sum));
    memset(delta_count, 0, sizeof(delta_count));

    double probs[256], word_probs[256], hctx_probs[256];

    for (size_t i = 0; i < len; i++) {
        uint8_t byte = data[i];
        double confidence;
        int order;

        ppm_predict(&ppm, probs, &confidence, &order);
        my_clamp_normalize(probs);

        int match_byte;
        double match_conf;
        match_predict(&match, &match_byte, &match_conf);
        my_blend_match(probs, match_byte, match_conf);

        double w_conf;
        if (word_predict_cached(&word, word_probs, &w_conf))
            my_blend_word(probs, word_probs, w_conf);

        double hctx_conf;
        if (highctx_predict(&hctx, hctx_probs, &hctx_conf))
            my_blend_hctx(probs, hctx_probs, hctx_conf);

        tweedie_denoise_instrumented(twd, probs, order, confidence);
        my_clamp_normalize(probs);

        tweedie_update(twd, byte);
        match_update(&match, byte);
        word_update(&word, byte);
        highctx_update(&hctx, byte);
        ppm_update(&ppm, byte);

        if ((i + 1) % 50000 == 0)
            fprintf(stderr, "\r  %5.1f%%", (i + 1) * 100.0 / len);
    }
    fprintf(stderr, "\r                    \r");

    double gammas[] = {0.500, 0.200, 0.059, 0.015};
    int    c_repr[] = {128, 512, 2048, 8192};

    printf("File: %s (%zu bytes)\n\n", argv[1], len);
    printf("%-8s %-8s", "gamma", "C_repr");
    for (int s = 0; s < TWD_STEPS; s++)
        printf("  step_%d  ", s);
    printf("\n");

    for (int b = 0; b < N_CONF_REPORT; b++) {
        printf("%-8.3f %-8d", gammas[b], c_repr[b]);
        for (int s = 0; s < TWD_STEPS; s++) {
            if (delta_count[s][b] > 0)
                printf("  %.4f  ", delta_sum[s][b] / delta_count[s][b]);
            else
                printf("  ---     ");
        }
        printf("\n");
    }

    free(twd);
    free(data);
    return 0;
}
