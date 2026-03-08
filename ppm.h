#ifndef PPM_H
#define PPM_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PPM_MAX_ORDER 4
#define PPM_NSYM 256
#define PPM_PRIOR 0.5

/*
 * Hash table entry: maps a 64-bit context hash to a count array.
 * counts[i] stores the (float) count for symbol i.
 * total caches sum(counts).
 * key == 0 means empty slot.
 */
typedef struct {
    uint64_t key;          /* context hash (0 = empty) */
    double counts[PPM_NSYM];
    double total;
} PPMEntry;

typedef struct {
    PPMEntry *entries;
    uint32_t capacity;     /* power of 2 */
    uint32_t used;
} PPMTable;

typedef struct {
    PPMTable tables[PPM_MAX_ORDER + 1];  /* order 0..4 */
    uint8_t *history;
    int hist_len;
    int hist_cap;
} PPMModel;

/* ── Hash helper ── */

static inline uint64_t ppm_hash_context(const uint8_t *ctx, int len) {
    /* We need a non-zero hash for all contexts including order-0 (empty).
     * Use FNV-1a style. Order-0 empty context gets a fixed hash. */
    if (len == 0) return 1;  /* special: order-0 empty context */
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) {
        h ^= ctx[i];
        h *= 1099511628211ULL;
    }
    if (h == 0) h = 1;  /* reserve 0 for empty slot */
    return h;
}

/* ── Table operations ── */

static inline void ppm_table_init(PPMTable *t, uint32_t capacity) {
    t->capacity = capacity;
    t->used = 0;
    t->entries = (PPMEntry *)calloc(capacity, sizeof(PPMEntry));
}

static inline void ppm_table_free(PPMTable *t) {
    free(t->entries);
    t->entries = NULL;
}

static inline void ppm_table_grow(PPMTable *t);

static inline PPMEntry *ppm_table_find(PPMTable *t, uint64_t key) {
    uint32_t mask = t->capacity - 1;
    uint32_t idx = (uint32_t)(key & mask);
    for (;;) {
        PPMEntry *e = &t->entries[idx];
        if (e->key == key) return e;
        if (e->key == 0) return NULL;
        idx = (idx + 1) & mask;
    }
}

static inline PPMEntry *ppm_table_insert(PPMTable *t, uint64_t key) {
    /* Grow if > 60% full */
    if (t->used * 5 > t->capacity * 3) {
        ppm_table_grow(t);
    }
    uint32_t mask = t->capacity - 1;
    uint32_t idx = (uint32_t)(key & mask);
    for (;;) {
        PPMEntry *e = &t->entries[idx];
        if (e->key == key) return e;  /* already exists */
        if (e->key == 0) {
            /* init new entry with prior */
            e->key = key;
            for (int i = 0; i < PPM_NSYM; i++)
                e->counts[i] = PPM_PRIOR;
            e->total = PPM_NSYM * PPM_PRIOR;
            t->used++;
            return e;
        }
        idx = (idx + 1) & mask;
    }
}

static inline void ppm_table_grow(PPMTable *t) {
    uint32_t old_cap = t->capacity;
    PPMEntry *old = t->entries;
    uint32_t new_cap = old_cap * 2;
    t->entries = (PPMEntry *)calloc(new_cap, sizeof(PPMEntry));
    t->capacity = new_cap;
    t->used = 0;
    for (uint32_t i = 0; i < old_cap; i++) {
        if (old[i].key != 0) {
            /* re-insert */
            PPMEntry *ne = ppm_table_insert(t, old[i].key);
            memcpy(ne->counts, old[i].counts, sizeof(old[i].counts));
            ne->total = old[i].total;
        }
    }
    free(old);
}

/* ── PPM Model ── */

static inline void ppm_init(PPMModel *m) {
    for (int o = 0; o <= PPM_MAX_ORDER; o++)
        ppm_table_init(&m->tables[o], 1024);
    m->hist_cap = 4096;
    m->hist_len = 0;
    m->history = (uint8_t *)malloc(m->hist_cap);
}

static inline void ppm_free(PPMModel *m) {
    for (int o = 0; o <= PPM_MAX_ORDER; o++)
        ppm_table_free(&m->tables[o]);
    free(m->history);
    m->history = NULL;
}

/*
 * predict_with_confidence: fills probs[256] and returns confidence + order.
 * Matches Python: fallback from max_order down to 0, first context with total > 1.
 * If nothing found, returns uniform.
 */
static inline void ppm_predict(PPMModel *m, double *probs,
                                double *out_confidence, int *out_order) {
    for (int order = PPM_MAX_ORDER; order >= 0; order--) {
        const uint8_t *ctx_start;
        int ctx_len = order;

        if (ctx_len > m->hist_len) continue;
        ctx_start = m->history + m->hist_len - ctx_len;

        uint64_t key = ppm_hash_context(ctx_start, ctx_len);
        PPMEntry *e = ppm_table_find(&m->tables[order], key);
        if (e == NULL) continue;
        if (e->total <= 1.0) continue;

        double inv_total = 1.0 / e->total;
        for (int i = 0; i < PPM_NSYM; i++)
            probs[i] = e->counts[i] * inv_total;

        *out_confidence = e->total;
        *out_order = order;
        return;
    }

    /* uniform fallback */
    double u = 1.0 / 256.0;
    for (int i = 0; i < PPM_NSYM; i++)
        probs[i] = u;
    *out_confidence = 0.0;
    *out_order = -1;
}

/*
 * update: add symbol count to all orders (0..4) where context is available.
 * Then append symbol to history.
 */
static inline void ppm_update(PPMModel *m, uint8_t symbol) {
    for (int order = 0; order <= PPM_MAX_ORDER; order++) {
        int ctx_len = order;
        if (ctx_len > m->hist_len) continue;

        const uint8_t *ctx_start = m->history + m->hist_len - ctx_len;
        uint64_t key = ppm_hash_context(ctx_start, ctx_len);

        PPMEntry *e = ppm_table_insert(&m->tables[order], key);
        e->counts[symbol] += 1.0;
        e->total += 1.0;
    }

    /* append to history */
    if (m->hist_len >= m->hist_cap) {
        m->hist_cap *= 2;
        m->history = (uint8_t *)realloc(m->history, m->hist_cap);
    }
    m->history[m->hist_len++] = symbol;
}

#endif /* PPM_H */
