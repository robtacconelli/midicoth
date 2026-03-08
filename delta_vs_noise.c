/*
 * delta_vs_noise.c — Experiment: |δ| vs noise level γ
 *
 * Compresses a file using the full pipeline, then dumps the
 * calibration table statistics showing mean |δ| per confidence bin.
 *
 * Usage: ./delta_vs_noise <input_file>
 *
 * Output: TSV table of (conf_bin, γ_approx, mean_|δ|, weighted_mean_|δ|, total_obs)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "fastmath.h"
#include "arith.h"
#include "ppm.h"
#include "tweedie.h"
#include "match.h"
#include "word.h"
#include "highctx.h"

#define SCALE (1 << 14)

static void probs_to_cumfreqs(const double *probs, int64_t *cumfreqs,
                               int64_t *out_total) {
    cumfreqs[0] = 0;
    for (int i = 0; i < 256; i++) {
        int64_t f = (int64_t)(probs[i] * SCALE + 0.5);
        if (f < 1) f = 1;
        cumfreqs[i + 1] = cumfreqs[i] + f;
    }
    *out_total = cumfreqs[256];
}

static void clamp_normalize(double *probs) {
    double sum = 0.0;
    for (int i = 0; i < 256; i++) {
        if (probs[i] < 1e-10) probs[i] = 1e-10;
        sum += probs[i];
    }
    double inv = 1.0 / sum;
    for (int i = 0; i < 256; i++)
        probs[i] *= inv;
}

/* Representative C value for each confidence bin.
 * twd_conf_bin uses: bin = (int)(ln(C) / 1.3863)
 * with C < 4 → bin 0.
 * Bin boundaries: 0:[0,4), 1:[4,e^1.39)≈[4,16), 2:[16,59), ... */
static double conf_bin_representative_C(int bin) {
    if (bin == 0) return 2.0;    /* midpoint of [0, 4) */
    /* bin = floor(ln(C) / 1.3863), so midpoint is exp((bin + 0.5) * 1.3863) */
    return exp((bin + 0.5) * 1.3862943611198906);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    FILE *fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 1; }
    fseek(fin, 0, SEEK_END);
    long file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc(file_size);
    if (fread(data, 1, file_size, fin) != (size_t)file_size) {
        fprintf(stderr, "Read error\n"); fclose(fin); return 1;
    }
    fclose(fin);

    fprintf(stderr, "Processing %s (%ld bytes)...\n", argv[1], file_size);

    /* Run the full pipeline to populate calibration tables */
    PPMModel ppm;       ppm_init(&ppm);
    MatchModel match;   match_init(&match);
    WordModel word;     word_init(&word);
    HighCtxModel hctx;  highctx_init(&hctx);
    ArithEncoder enc;   ae_init(&enc);
    TweedieDenoiser twd; tweedie_init(&twd);

    double probs[256], word_probs[256], hctx_probs[256];
    int64_t cumfreqs[257];
    int64_t total;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (long i = 0; i < file_size; i++) {
        uint8_t byte = data[i];

        double confidence;
        int order;
        ppm_predict(&ppm, probs, &confidence, &order);

        tweedie_denoise(&twd, probs, order, confidence);
        clamp_normalize(probs);

        int match_byte;
        double match_conf;
        match_predict(&match, &match_byte, &match_conf);
        blend_match(probs, match_byte, match_conf);

        double w_conf;
        if (word_predict_cached(&word, word_probs, &w_conf))
            blend_word_model(probs, word_probs, w_conf);

        double hctx_conf;
        if (highctx_predict(&hctx, hctx_probs, &hctx_conf))
            blend_highctx(probs, hctx_probs, hctx_conf);

        probs_to_cumfreqs(probs, cumfreqs, &total);
        ae_encode(&enc, cumfreqs, byte, total);

        tweedie_update(&twd, byte);
        match_update(&match, byte);
        word_update(&word, byte);
        highctx_update(&hctx, byte);
        ppm_update(&ppm, byte);

        if ((i + 1) % 50000 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
            fprintf(stderr, "\r  %5.1f%%  (%.0f B/s)",
                    (i + 1) * 100.0 / file_size, (i + 1) / elapsed);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "\r  Done in %.1fs                    \n", elapsed);

    /* ── Analyze calibration tables: mean |δ| per confidence bin ── */

    /* Aggregate across all steps, bit contexts, order groups, shapes, prob bins */
    double sum_abs_delta[TWD_N_CONF];
    double sum_weight[TWD_N_CONF];
    double sum_weighted_abs_delta[TWD_N_CONF];
    int    count[TWD_N_CONF];
    memset(sum_abs_delta, 0, sizeof(sum_abs_delta));
    memset(sum_weight, 0, sizeof(sum_weight));
    memset(sum_weighted_abs_delta, 0, sizeof(sum_weighted_abs_delta));
    memset(count, 0, sizeof(count));

    for (int t = 0; t < TWD_STEPS; t++)
        for (int b = 0; b < TWD_N_BCTX; b++)
            for (int o = 0; o < TWD_N_ORD; o++)
                for (int s = 0; s < TWD_N_SHAPE; s++)
                    for (int c = 0; c < TWD_N_CONF; c++)
                        for (int p = 0; p < TWD_N_PROB; p++) {
                            TwdCalibEntry *e = &twd.table[t][b][o][s][c][p];
                            double real_obs = e->total - TWD_PRIOR_WEIGHT;
                            if (real_obs < 1.0) continue;  /* skip bins with only prior */

                            double avg_pred = e->sum_pred / e->total;
                            double emp_rate = e->hits / e->total;
                            double delta = emp_rate - avg_pred;

                            sum_abs_delta[c] += fabs(delta);
                            sum_weighted_abs_delta[c] += fabs(delta) * real_obs;
                            sum_weight[c] += real_obs;
                            count[c]++;
                        }

    /* ── Output ── */
    printf("# Delta vs Noise Level — %s (%ld bytes)\n", argv[1], file_size);
    printf("# conf_bin\tC_repr\tgamma\tmean_abs_delta\tweighted_abs_delta\tactive_bins\ttotal_obs\n");

    for (int c = 0; c < TWD_N_CONF; c++) {
        double C_repr = conf_bin_representative_C(c);
        double gamma = 128.0 / (C_repr + 128.0);
        double mean_d = (count[c] > 0) ? sum_abs_delta[c] / count[c] : 0.0;
        double wmean_d = (sum_weight[c] > 0) ? sum_weighted_abs_delta[c] / sum_weight[c] : 0.0;

        printf("%d\t%.1f\t%.4f\t%.6f\t%.6f\t%d\t%.0f\n",
               c, C_repr, gamma, mean_d, wmean_d, count[c], sum_weight[c]);
    }

    /* ── Also output per-step breakdown ── */
    printf("\n# Per-step breakdown:\n");
    printf("# step\tconf_bin\tgamma\tweighted_abs_delta\ttotal_obs\n");

    for (int t = 0; t < TWD_STEPS; t++) {
        double step_sum_wd[TWD_N_CONF] = {0};
        double step_sum_w[TWD_N_CONF] = {0};

        for (int b = 0; b < TWD_N_BCTX; b++)
            for (int o = 0; o < TWD_N_ORD; o++)
                for (int s = 0; s < TWD_N_SHAPE; s++)
                    for (int c = 0; c < TWD_N_CONF; c++)
                        for (int p = 0; p < TWD_N_PROB; p++) {
                            TwdCalibEntry *e = &twd.table[t][b][o][s][c][p];
                            double real_obs = e->total - TWD_PRIOR_WEIGHT;
                            if (real_obs < 1.0) continue;

                            double avg_pred = e->sum_pred / e->total;
                            double emp_rate = e->hits / e->total;
                            double delta = emp_rate - avg_pred;

                            step_sum_wd[c] += fabs(delta) * real_obs;
                            step_sum_w[c] += real_obs;
                        }

        for (int c = 0; c < TWD_N_CONF; c++) {
            double C_repr = conf_bin_representative_C(c);
            double gamma = 128.0 / (C_repr + 128.0);
            double wmean_d = (step_sum_w[c] > 0) ? step_sum_wd[c] / step_sum_w[c] : 0.0;
            printf("%d\t%d\t%.4f\t%.6f\t%.0f\n",
                   t, c, gamma, wmean_d, step_sum_w[c]);
        }
    }

    free(data);
    ppm_free(&ppm);
    match_free(&match);
    word_free(&word);
    highctx_free(&hctx);
    ae_free(&enc);

    return 0;
}
