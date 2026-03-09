/*
 * Midicoth Compressor — C implementation
 * Pipeline: PPM + Match + Word + HighCtx + Tweedie Denoising
 *
 * Usage:
 *   ./mdc compress   <input> <output>
 *   ./mdc decompress <input> <output>
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

#define MAGIC "MDC7"
#define SCALE (1 << 14)

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

/* ── Compress ── */

static int do_compress(const char *input_path, const char *output_path) {
    FILE *fin = fopen(input_path, "rb");
    if (!fin) { perror(input_path); return 1; }
    fseek(fin, 0, SEEK_END);
    long file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc(file_size);
    if (fread(data, 1, file_size, fin) != (size_t)file_size) {
        fprintf(stderr, "Read error\n"); fclose(fin); return 1;
    }
    fclose(fin);

    uint64_t original_size = (uint64_t)file_size;
    printf("  Input:  %s (%lu bytes)\n", input_path, (unsigned long)original_size);

    if (original_size == 0) {
        FILE *fout = fopen(output_path, "wb");
        fwrite(MAGIC, 1, 4, fout);
        uint64_t zero = 0;
        fwrite(&zero, 8, 1, fout);
        fclose(fout);
        printf("  Empty file -> 12 bytes\n");
        free(data);
        return 0;
    }

    PPMModel ppm;     ppm_init(&ppm);
    MatchModel match; match_init(&match);
    WordModel word;   word_init(&word);
    HighCtxModel hctx; highctx_init(&hctx);
    ArithEncoder enc; ae_init(&enc);
    TweedieDenoiser *twd = (TweedieDenoiser *)malloc(sizeof(TweedieDenoiser));
    tweedie_init(twd);

    double probs[256], word_probs[256], hctx_probs[256];
    int64_t cumfreqs[257];
    int64_t total;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (uint64_t i = 0; i < original_size; i++) {
        uint8_t byte = data[i];

        double confidence;
        int order;
        ppm_predict(&ppm, probs, &confidence, &order);
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

        tweedie_denoise(twd, probs, order, confidence);
        clamp_normalize(probs);

        probs_to_cumfreqs(probs, cumfreqs, &total);
        ae_encode(&enc, cumfreqs, byte, total);

        tweedie_update(twd, byte);
        match_update(&match, byte);
        word_update(&word, byte);
        highctx_update(&hctx, byte);
        ppm_update(&ppm, byte);

        if ((i + 1) % 50000 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
            double pct = (i + 1) * 100.0 / original_size;
            double speed = (i + 1) / elapsed;
            fprintf(stderr, "\r    %5.1f%%  (%lu/%lu)  %.0f B/s",
                    pct, (unsigned long)(i + 1), (unsigned long)original_size, speed);
        }
    }

    ae_finish(&enc);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "\r                                                \r");

    FILE *fout = fopen(output_path, "wb");
    if (!fout) { perror(output_path); return 1; }
    fwrite(MAGIC, 1, 4, fout);
    fwrite(&original_size, 8, 1, fout);
    fwrite(enc.buf, 1, enc.buf_len, fout);
    fclose(fout);

    uint64_t total_size = 4 + 8 + enc.buf_len;
    double ratio = (double)total_size / original_size;
    printf("  Output: %s (%lu bytes)\n", output_path, (unsigned long)total_size);
    printf("  Ratio:  %.4f (%.2f%%)\n", ratio, ratio * 100.0);
    printf("  Time:   %.1fs (%.0f B/s)\n", elapsed, original_size / elapsed);

    ae_free(&enc);
    ppm_free(&ppm);
    match_free(&match);
    word_free(&word);
    highctx_free(&hctx);
    free(twd);
    free(data);
    return 0;
}

/* ── Decompress ── */

static int do_decompress(const char *input_path, const char *output_path) {
    FILE *fin = fopen(input_path, "rb");
    if (!fin) { perror(input_path); return 1; }

    char magic[4];
    if (fread(magic, 1, 4, fin) != 4 || memcmp(magic, MAGIC, 4) != 0) {
        fprintf(stderr, "Error: not a MDC7 file\n");
        fclose(fin);
        return 1;
    }

    uint64_t original_size;
    if (fread(&original_size, 8, 1, fin) != 1) {
        fprintf(stderr, "Read error\n"); fclose(fin); return 1;
    }

    fseek(fin, 0, SEEK_END);
    long fsize = ftell(fin);
    fseek(fin, 12, SEEK_SET);
    size_t comp_len = (size_t)(fsize - 12);
    uint8_t *compressed = (uint8_t *)malloc(comp_len);
    if (fread(compressed, 1, comp_len, fin) != comp_len) {
        fprintf(stderr, "Read error\n"); fclose(fin); return 1;
    }
    fclose(fin);

    printf("  Input:  %s (%ld bytes)\n", input_path, fsize);
    printf("  Original size: %lu bytes\n", (unsigned long)original_size);

    if (original_size == 0) {
        FILE *fout = fopen(output_path, "wb");
        fclose(fout);
        printf("  Empty file\n");
        free(compressed);
        return 0;
    }

    PPMModel ppm;     ppm_init(&ppm);
    MatchModel match; match_init(&match);
    WordModel word;   word_init(&word);
    HighCtxModel hctx; highctx_init(&hctx);
    ArithDecoder dec; ad_init(&dec, compressed, comp_len);
    TweedieDenoiser *twd = (TweedieDenoiser *)malloc(sizeof(TweedieDenoiser));
    tweedie_init(twd);

    uint8_t *result = (uint8_t *)malloc(original_size);

    double probs[256], word_probs[256], hctx_probs[256];
    int64_t cumfreqs[257];
    int64_t total;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (uint64_t i = 0; i < original_size; i++) {
        double confidence;
        int order;
        ppm_predict(&ppm, probs, &confidence, &order);
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

        tweedie_denoise(twd, probs, order, confidence);
        clamp_normalize(probs);

        probs_to_cumfreqs(probs, cumfreqs, &total);
        int sym = ad_decode(&dec, cumfreqs, total);
        result[i] = (uint8_t)sym;

        tweedie_update(twd, (uint8_t)sym);
        match_update(&match, (uint8_t)sym);
        word_update(&word, (uint8_t)sym);
        highctx_update(&hctx, (uint8_t)sym);
        ppm_update(&ppm, (uint8_t)sym);

        if ((i + 1) % 50000 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
            double pct = (i + 1) * 100.0 / original_size;
            double speed = (i + 1) / elapsed;
            fprintf(stderr, "\r    %5.1f%%  (%lu/%lu)  %.0f B/s",
                    pct, (unsigned long)(i + 1), (unsigned long)original_size, speed);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "\r                                                \r");

    FILE *fout = fopen(output_path, "wb");
    fwrite(result, 1, original_size, fout);
    fclose(fout);

    printf("  Output: %s (%lu bytes)\n", output_path, (unsigned long)original_size);
    printf("  Time:   %.1fs (%.0f B/s)\n", elapsed, original_size / elapsed);

    ppm_free(&ppm);
    match_free(&match);
    word_free(&word);
    highctx_free(&hctx);
    free(twd);
    free(compressed);
    free(result);
    return 0;
}

/* ── Main ── */

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s compress|decompress <input> <output>\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "compress") == 0)
        return do_compress(argv[2], argv[3]);
    else if (strcmp(argv[1], "decompress") == 0)
        return do_decompress(argv[2], argv[3]);
    else {
        fprintf(stderr, "Unknown command: %s\n", argv[1]);
        return 1;
    }
}
