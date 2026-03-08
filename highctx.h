#ifndef HIGHCTX_H
#define HIGHCTX_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * High-Order Context Model (orders 5-8)
 *
 * Extends effective context beyond PPM's order-4 limit without modifying PPM.
 * Uses hash tables mapping context_hash → count[256] for orders 5, 6, 7, 8.
 * Unlike the match model (which finds one position, predicts one byte),
 * this aggregates ALL matching positions into a full probability distribution.
 *
 * Blended after SSE in the pipeline, preserving diffusion's contribution.
 */

#define HCTX_NSYM    256
#define HCTX_N_ORDERS 4   /* orders 5, 6, 7, 8 */
#define HCTX_MIN_ORDER 5
#define HCTX_MAX_ENTRIES (1 << 20)  /* 1M entries per table, ~500MB total max */

/* Hash table entry: context hash → byte counts */
typedef struct {
    uint64_t key;
    uint16_t counts[HCTX_NSYM];
    uint32_t total;
} HCtxEntry;

typedef struct {
    HCtxEntry *entries;
    uint32_t capacity;
    uint32_t mask;
    uint32_t used;
} HCtxTable;

static inline void hctx_table_init(HCtxTable *t, uint32_t cap) {
    t->capacity = cap;
    t->mask = cap - 1;
    t->used = 0;
    t->entries = (HCtxEntry *)calloc(cap, sizeof(HCtxEntry));
}

static inline void hctx_table_free(HCtxTable *t) {
    free(t->entries);
    t->entries = NULL;
}

static inline void hctx_table_grow(HCtxTable *t) {
    uint32_t old_cap = t->capacity;
    HCtxEntry *old = t->entries;
    uint32_t new_cap = old_cap * 2;
    t->entries = (HCtxEntry *)calloc(new_cap, sizeof(HCtxEntry));
    t->capacity = new_cap;
    t->mask = new_cap - 1;
    t->used = 0;
    for (uint32_t i = 0; i < old_cap; i++) {
        if (old[i].key != 0) {
            uint32_t idx = (uint32_t)(old[i].key & t->mask);
            while (t->entries[idx].key != 0)
                idx = (idx + 1) & t->mask;
            t->entries[idx] = old[i];
            t->used++;
        }
    }
    free(old);
}

/* Find or create entry. Returns pointer to entry (NULL if full and create). */
static inline HCtxEntry *hctx_table_get(HCtxTable *t, uint64_t key, int create) {
    if (create && t->used * 5 > t->capacity * 3) {
        if (t->capacity < HCTX_MAX_ENTRIES)
            hctx_table_grow(t);
        else
            create = 0;  /* at max capacity, only look up existing */
    }
    uint32_t idx = (uint32_t)(key & t->mask);
    for (;;) {
        if (t->entries[idx].key == key)
            return &t->entries[idx];
        if (t->entries[idx].key == 0) {
            if (!create) return NULL;
            t->entries[idx].key = key;
            memset(t->entries[idx].counts, 0, sizeof(t->entries[idx].counts));
            t->entries[idx].total = 0;
            t->used++;
            return &t->entries[idx];
        }
        idx = (idx + 1) & t->mask;
    }
}

/* FNV-1a hash for context bytes */
static inline uint64_t hctx_hash(const uint8_t *data, int len) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    if (h == 0) h = 1;
    return h;
}

typedef struct {
    HCtxTable tables[HCTX_N_ORDERS];  /* orders 5, 6, 7, 8 */
    uint8_t *history;
    int hist_len;
    int hist_cap;
} HighCtxModel;

static inline void highctx_init(HighCtxModel *m) {
    for (int i = 0; i < HCTX_N_ORDERS; i++)
        hctx_table_init(&m->tables[i], 8192);
    m->hist_cap = 4096;
    m->hist_len = 0;
    m->history = (uint8_t *)malloc(m->hist_cap);
}

static inline void highctx_free(HighCtxModel *m) {
    for (int i = 0; i < HCTX_N_ORDERS; i++)
        hctx_table_free(&m->tables[i]);
    free(m->history);
    m->history = NULL;
}

/*
 * Predict: try highest order first (8, 7, 6, 5).
 * Use the highest order that has a context with total >= min_count.
 * Returns 1 if prediction available, fills probs[256] and *out_conf.
 */
static inline int highctx_predict(HighCtxModel *m, double *probs, double *out_conf) {
    int n = m->hist_len;

    for (int oidx = HCTX_N_ORDERS - 1; oidx >= 0; oidx--) {
        int order = HCTX_MIN_ORDER + oidx;  /* 8, 7, 6, 5 */
        if (n < order) continue;

        uint64_t key = hctx_hash(m->history + n - order, order);
        HCtxEntry *e = hctx_table_get(&m->tables[oidx], key, 0);
        if (!e || e->total < 4) continue;

        /* Build distribution: sparse smoothing to avoid zero probs */
        double smooth = 1e-4;
        double total_smooth = e->total + smooth * HCTX_NSYM;
        double inv = 1.0 / total_smooth;
        for (int s = 0; s < HCTX_NSYM; s++)
            probs[s] = (e->counts[s] + smooth) * inv;

        /* Confidence: ramps slowly, requires real data */
        double count_conf = (e->total - 4.0) / (e->total + 8.0);  /* 0 at total=4, ~0.7 at 20 */
        if (count_conf < 0) count_conf = 0;
        double order_factor = 0.4 + (order - HCTX_MIN_ORDER) * 0.1;  /* 0.4 for o5, 0.7 for o8 */
        *out_conf = count_conf * order_factor;

        return 1;
    }

    *out_conf = 0.0;
    return 0;
}

/*
 * Update: increment counts for all available orders.
 */
static inline void highctx_update(HighCtxModel *m, uint8_t byte) {
    int n = m->hist_len;

    for (int oidx = 0; oidx < HCTX_N_ORDERS; oidx++) {
        int order = HCTX_MIN_ORDER + oidx;
        if (n >= order) {
            uint64_t key = hctx_hash(m->history + n - order, order);
            HCtxEntry *e = hctx_table_get(&m->tables[oidx], key, 1);
            if (e) {
                e->counts[byte]++;
                e->total++;
            }
        }
    }

    /* Append to history */
    if (m->hist_len >= m->hist_cap) {
        m->hist_cap *= 2;
        m->history = (uint8_t *)realloc(m->history, m->hist_cap);
    }
    m->history[m->hist_len++] = byte;
}

/*
 * Blend high-context prediction into existing probability distribution.
 */
static inline void blend_highctx(double *probs, const double *hctx_probs,
                                   double hctx_conf) {
    if (hctx_conf < 0.01) return;
    double weight = hctx_conf * 2.0;
    if (weight > 0.60) weight = 0.60;
    double sum = 0.0;
    for (int i = 0; i < HCTX_NSYM; i++) {
        probs[i] = probs[i] * (1.0 - weight) + hctx_probs[i] * weight;
        if (probs[i] < 1e-10) probs[i] = 1e-10;
        sum += probs[i];
    }
    double inv = 1.0 / sum;
    for (int i = 0; i < HCTX_NSYM; i++)
        probs[i] *= inv;
}

#endif /* HIGHCTX_H */
