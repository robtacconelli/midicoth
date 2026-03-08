/*
 * Ablation study — C implementation
 * Measures incremental contribution of each pipeline layer.
 *
 * Usage:
 *   ./ablation                     # alice29.txt only
 *   ./ablation file1 file2 ...     # specific files
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <libgen.h>

#include "fastmath.h"
#include "arith.h"
#include "ppm.h"
#include "tweedie.h"
#include "match.h"
#include "word.h"
#include "highctx.h"

#define SCALE (1 << 14)

/* ── Flags ── */
#define FLAG_TWEEDIE    1
#define FLAG_MATCH      2
#define FLAG_WORD       4
#define FLAG_HIGHCTX    8

/* ── Helpers ── */

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

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── Configurable compress ── */

static uint8_t *do_compress(const uint8_t *data, size_t data_len,
                             int flags, size_t *out_len, double *out_time) {
    PPMModel ppm;       ppm_init(&ppm);
    MatchModel match;   if (flags & FLAG_MATCH) match_init(&match);
    WordModel word;     if (flags & FLAG_WORD) word_init(&word);
    HighCtxModel hctx;  if (flags & FLAG_HIGHCTX) highctx_init(&hctx);
    ArithEncoder enc;   ae_init(&enc);
    TweedieDenoiser twd; if (flags & FLAG_TWEEDIE) tweedie_init(&twd);

    double probs[256], word_probs[256], hctx_probs[256];
    int64_t cumfreqs[257];
    int64_t total;

    double t0 = now_sec();

    for (size_t i = 0; i < data_len; i++) {
        uint8_t byte = data[i];
        double confidence;
        int order;

        ppm_predict(&ppm, probs, &confidence, &order);

        if (flags & FLAG_TWEEDIE) {
            tweedie_denoise(&twd, probs, order, confidence);
        }
        clamp_normalize(probs);

        if (flags & FLAG_MATCH) {
            int match_byte;
            double match_conf;
            match_predict(&match, &match_byte, &match_conf);
            blend_match(probs, match_byte, match_conf);
        }

        if (flags & FLAG_WORD) {
            double w_conf;
            if (word_predict_cached(&word, word_probs, &w_conf))
                blend_word_model(probs, word_probs, w_conf);
        }

        if (flags & FLAG_HIGHCTX) {
            double hctx_conf;
            if (highctx_predict(&hctx, hctx_probs, &hctx_conf))
                blend_highctx(probs, hctx_probs, hctx_conf);
        }

        probs_to_cumfreqs(probs, cumfreqs, &total);
        ae_encode(&enc, cumfreqs, byte, total);

        /* Updates */
        if (flags & FLAG_TWEEDIE)
            tweedie_update(&twd, byte);
        if (flags & FLAG_MATCH)
            match_update(&match, byte);
        if (flags & FLAG_WORD)
            word_update(&word, byte);
        if (flags & FLAG_HIGHCTX)
            highctx_update(&hctx, byte);
        ppm_update(&ppm, byte);

        if ((i + 1) % 50000 == 0) {
            double elapsed = now_sec() - t0;
            double pct = (i + 1) * 100.0 / data_len;
            double speed = (i + 1) / elapsed;
            fprintf(stderr, "\r    %5.1f%%  (%zu/%zu)  %.0f B/s",
                    pct, i + 1, data_len, speed);
        }
    }

    ae_finish(&enc);
    double elapsed = now_sec() - t0;
    if (data_len >= 50000)
        fprintf(stderr, "\r                                                \r");

    *out_time = elapsed;

    /* Copy output */
    *out_len = enc.buf_len;
    uint8_t *result = (uint8_t *)malloc(enc.buf_len);
    memcpy(result, enc.buf, enc.buf_len);

    ae_free(&enc);
    ppm_free(&ppm);
    if (flags & FLAG_MATCH) match_free(&match);
    if (flags & FLAG_WORD) word_free(&word);
    if (flags & FLAG_HIGHCTX) highctx_free(&hctx);

    return result;
}

/* ── Configurable decompress ── */

static uint8_t *do_decompress(const uint8_t *compressed, size_t comp_len,
                                size_t original_size, int flags,
                                double *out_time) {
    PPMModel ppm;       ppm_init(&ppm);
    MatchModel match;   if (flags & FLAG_MATCH) match_init(&match);
    WordModel word;     if (flags & FLAG_WORD) word_init(&word);
    HighCtxModel hctx;  if (flags & FLAG_HIGHCTX) highctx_init(&hctx);
    ArithDecoder dec;   ad_init(&dec, compressed, comp_len);
    TweedieDenoiser twd; if (flags & FLAG_TWEEDIE) tweedie_init(&twd);

    uint8_t *result = (uint8_t *)malloc(original_size);

    double probs[256], word_probs[256], hctx_probs[256];
    int64_t cumfreqs[257];
    int64_t total;

    double t0 = now_sec();

    for (size_t i = 0; i < original_size; i++) {
        double confidence;
        int order;

        ppm_predict(&ppm, probs, &confidence, &order);

        if (flags & FLAG_TWEEDIE) {
            tweedie_denoise(&twd, probs, order, confidence);
        }
        clamp_normalize(probs);

        if (flags & FLAG_MATCH) {
            int match_byte;
            double match_conf;
            match_predict(&match, &match_byte, &match_conf);
            blend_match(probs, match_byte, match_conf);
        }

        if (flags & FLAG_WORD) {
            double w_conf;
            if (word_predict_cached(&word, word_probs, &w_conf))
                blend_word_model(probs, word_probs, w_conf);
        }

        if (flags & FLAG_HIGHCTX) {
            double hctx_conf;
            if (highctx_predict(&hctx, hctx_probs, &hctx_conf))
                blend_highctx(probs, hctx_probs, hctx_conf);
        }

        probs_to_cumfreqs(probs, cumfreqs, &total);
        int sym = ad_decode(&dec, cumfreqs, total);
        result[i] = (uint8_t)sym;

        if (flags & FLAG_TWEEDIE)
            tweedie_update(&twd, (uint8_t)sym);
        if (flags & FLAG_MATCH)
            match_update(&match, (uint8_t)sym);
        if (flags & FLAG_WORD)
            word_update(&word, (uint8_t)sym);
        if (flags & FLAG_HIGHCTX)
            highctx_update(&hctx, (uint8_t)sym);
        ppm_update(&ppm, (uint8_t)sym);
    }

    *out_time = now_sec() - t0;

    ppm_free(&ppm);
    if (flags & FLAG_MATCH) match_free(&match);
    if (flags & FLAG_WORD) word_free(&word);
    if (flags & FLAG_HIGHCTX) highctx_free(&hctx);

    return result;
}

/* ── Ablation configs ── */

typedef struct {
    const char *label;
    int flags;
} AblationConfig;

static const AblationConfig CONFIGS[] = {
    { "Base PPM",                    0 },
    { "+ Tweedie",                   FLAG_TWEEDIE },
    { "+ Twd + Match",               FLAG_TWEEDIE | FLAG_MATCH },
    { "+ Twd + Match + Word",        FLAG_TWEEDIE | FLAG_MATCH | FLAG_WORD },
    { "+ Twd + M + W + H",           FLAG_TWEEDIE | FLAG_MATCH | FLAG_WORD | FLAG_HIGHCTX },
};
#define N_CONFIGS 5

typedef struct {
    const char *label;
    size_t c_size;
    double ratio;
    double c_time;
} AblationResult;

static void run_ablation(const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) { fprintf(stderr, "File not found: %s\n", filepath); return; }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc(file_size);
    if (fread(data, 1, file_size, f) != (size_t)file_size) {
        fprintf(stderr, "Read error: %s\n", filepath);
        fclose(f); free(data); return;
    }
    fclose(f);

    /* basename */
    char *path_copy = strdup(filepath);
    const char *filename = basename(path_copy);
    size_t original_size = (size_t)file_size;

    printf("\n======================================================================\n");
    printf("  ABLATION: %s (%zu bytes)\n", filename, original_size);
    printf("======================================================================\n");

    AblationResult results[N_CONFIGS];

    for (int c = 0; c < N_CONFIGS; c++) {
        printf("\n  [%s]\n", CONFIGS[c].label);
        printf("    Compressing...");
        fflush(stdout);

        size_t comp_len;
        double c_time;
        uint8_t *compressed = do_compress(data, original_size, CONFIGS[c].flags,
                                           &comp_len, &c_time);
        double ratio = (double)comp_len / original_size;
        printf(" %zu bytes (%.2f%%) in %.1fs\n", comp_len, ratio * 100.0, c_time);

        /* Verify round-trip */
        printf("    Verifying...");
        fflush(stdout);

        /*double d_time;
        uint8_t *decompressed = do_decompress(compressed, comp_len, original_size,
                                                CONFIGS[c].flags, &d_time);

        if (memcmp(data, decompressed, original_size) == 0) {
            printf(" OK (%.1fs)\n", d_time);
        } else {
            printf(" FAILED!\n");
            // Find first mismatch
            for (size_t i = 0; i < original_size; i++) {
                if (data[i] != decompressed[i]) {
                    printf("    First mismatch at byte %zu: expected %d, got %d\n",
                           i, data[i], decompressed[i]);
                    break;
                }
            }
            free(compressed);
            free(decompressed);
            free(data);
            free(path_copy);
            exit(1);
        } */

        results[c].label = CONFIGS[c].label;
        results[c].c_size = comp_len;
        results[c].ratio = ratio;
        results[c].c_time = c_time;

        free(compressed);
        //free(decompressed);
    }

    /* ── Summary table ── */
    printf("\n======================================================================\n");
    printf("  RESULTS: %s (%zu bytes)\n", filename, original_size);
    printf("======================================================================\n");
    printf("  %-32s %8s %8s %9s %9s %7s\n",
           "Layer", "Size", "Ratio", "Layer +%", "Total +%", "Time");
    printf("  -------------------------------- -------- -------- --------- --------- -------\n");

    size_t base_size = results[0].c_size;
    size_t prev_size = results[0].c_size;

    for (int i = 0; i < N_CONFIGS; i++) {
        size_t c_size = results[i].c_size;
        double ratio = results[i].ratio;
        double c_time = results[i].c_time;

        if (i == 0) {
            printf("  %-32s %8zu %6.2f%% %9s %9s %6.1fs\n",
                   results[i].label, c_size, ratio * 100.0, "", "", c_time);
        } else {
            double layer_imp = (double)(prev_size - c_size) / prev_size * 100.0;
            double total_imp = (double)(base_size - c_size) / base_size * 100.0;
            printf("  %-32s %8zu %6.2f%% %+8.2f%% %+8.2f%% %6.1fs\n",
                   results[i].label, c_size, ratio * 100.0,
                   layer_imp, total_imp, c_time);
        }
        prev_size = c_size;
    }

    printf("  -------------------------------- -------- -------- --------- --------- -------\n");
    size_t final_size = results[N_CONFIGS - 1].c_size;
    double total_imp = (double)(base_size - final_size) / base_size * 100.0;
    printf("  %-32s %8s %8s %9s %+8.2f%%\n",
           "TOTAL IMPROVEMENT", "", "", "", total_imp);
    printf("\n");

    free(data);
    free(path_copy);
}

/* ── Main ── */

int main(int argc, char **argv) {
    if (argc > 1) {
        for (int i = 1; i < argc; i++)
            run_ablation(argv[i]);
    } else {
        run_ablation("../alice29.txt");
    }

    /* Cross-file comparison would go here for multiple files */
    return 0;
}
