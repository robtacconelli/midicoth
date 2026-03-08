#ifndef MATCH_H
#define MATCH_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MATCH_NSYM 256
#define MATCH_N_CTX 5  /* context lengths: 4, 6, 8, 12, 16 */

/* Hash table entry: context hash → position in history */
typedef struct {
    uint64_t key;
    uint32_t pos;
} MatchHTEntry;

typedef struct {
    MatchHTEntry *entries;
    uint32_t capacity;
    uint32_t mask;
} MatchHT;

static inline void mht_init(MatchHT *t, uint32_t cap) {
    t->capacity = cap;
    t->mask = cap - 1;
    t->entries = (MatchHTEntry *)calloc(cap, sizeof(MatchHTEntry));
}

static inline void mht_free(MatchHT *t) {
    free(t->entries);
}

static inline void mht_grow(MatchHT *t) {
    uint32_t old_cap = t->capacity;
    MatchHTEntry *old = t->entries;
    uint32_t new_cap = old_cap * 2;
    t->entries = (MatchHTEntry *)calloc(new_cap, sizeof(MatchHTEntry));
    t->capacity = new_cap;
    t->mask = new_cap - 1;
    for (uint32_t i = 0; i < old_cap; i++) {
        if (old[i].key != 0) {
            uint32_t idx = (uint32_t)(old[i].key & t->mask);
            while (t->entries[idx].key != 0)
                idx = (idx + 1) & t->mask;
            t->entries[idx] = old[i];
        }
    }
    free(old);
}

static inline void mht_set(MatchHT *t, uint64_t key, uint32_t pos,
                             uint32_t *used) {
    if (*used * 5 > t->capacity * 3) mht_grow(t);
    uint32_t idx = (uint32_t)(key & t->mask);
    for (;;) {
        if (t->entries[idx].key == key || t->entries[idx].key == 0) {
            if (t->entries[idx].key == 0) (*used)++;
            t->entries[idx].key = key;
            t->entries[idx].pos = pos;
            return;
        }
        idx = (idx + 1) & t->mask;
    }
}

static inline int mht_get(MatchHT *t, uint64_t key, uint32_t *out_pos) {
    uint32_t idx = (uint32_t)(key & t->mask);
    for (;;) {
        if (t->entries[idx].key == key) {
            *out_pos = t->entries[idx].pos;
            return 1;
        }
        if (t->entries[idx].key == 0) return 0;
        idx = (idx + 1) & t->mask;
    }
}

typedef struct {
    int ctx_lens[MATCH_N_CTX];
    MatchHT tables[MATCH_N_CTX];
    uint32_t table_used[MATCH_N_CTX];
    uint8_t *history;
    int hist_len;
    int hist_cap;

    /* active match state */
    int match_read_pos;
    int match_active;
    int match_streak;

    /* adaptive accuracy */
    double hits;
    double total;
} MatchModel;

static inline uint64_t match_ctx_hash(const uint8_t *data, int len) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    if (h == 0) h = 1;
    return h;
}

static inline void match_init(MatchModel *m) {
    m->ctx_lens[0] = 4;
    m->ctx_lens[1] = 6;
    m->ctx_lens[2] = 8;
    m->ctx_lens[3] = 12;
    m->ctx_lens[4] = 16;
    for (int i = 0; i < MATCH_N_CTX; i++) {
        mht_init(&m->tables[i], 4096);
        m->table_used[i] = 0;
    }
    m->hist_cap = 4096;
    m->hist_len = 0;
    m->history = (uint8_t *)malloc(m->hist_cap);
    m->match_read_pos = -1;
    m->match_active = 0;
    m->match_streak = 0;
    m->hits = 1.0;
    m->total = 2.0;
}

static inline void match_free(MatchModel *m) {
    for (int i = 0; i < MATCH_N_CTX; i++)
        mht_free(&m->tables[i]);
    free(m->history);
    m->history = NULL;
}

/*
 * predict: returns predicted byte via *out_byte, confidence via *out_conf.
 * Returns 1 if prediction available, 0 otherwise.
 */
static inline int match_predict(MatchModel *m, int *out_byte, double *out_conf) {
    /* 1. Continue active match */
    if (m->match_active && m->match_read_pos >= 0
        && m->match_read_pos < m->hist_len) {
        *out_byte = m->history[m->match_read_pos];
        double base = m->hits / m->total;
        double conf = base * (0.65 + m->match_streak * 0.04);
        if (conf > 0.96) conf = 0.96;
        *out_conf = conf;
        return 1;
    }

    m->match_active = 0;

    /* 2. Try new match (longest context first) */
    for (int idx = MATCH_N_CTX - 1; idx >= 0; idx--) {
        int ctx_len = m->ctx_lens[idx];
        int n = m->hist_len;
        if (n < ctx_len) continue;

        uint64_t key = match_ctx_hash(m->history + n - ctx_len, ctx_len);
        uint32_t pos;
        if (mht_get(&m->tables[idx], key, &pos) && pos < (uint32_t)n) {
            *out_byte = m->history[pos];
            m->match_active = 1;
            m->match_read_pos = (int)pos;
            m->match_streak = 0;
            double base = m->hits / m->total;
            double conf = base * (ctx_len / 6.0);
            if (conf > base * 0.9) conf = base * 0.9;
            *out_conf = conf;
            return 1;
        }
    }

    *out_byte = -1;
    *out_conf = 0.0;
    return 0;
}

static inline void match_update(MatchModel *m, uint8_t actual_byte) {
    /* track accuracy of active match */
    if (m->match_active && m->match_read_pos >= 0
        && m->match_read_pos < m->hist_len) {
        int predicted = m->history[m->match_read_pos];
        m->total += 1.0;
        if (predicted == actual_byte) {
            m->hits += 1.0;
            m->match_streak++;
            m->match_read_pos++;
        } else {
            m->match_active = 0;
            m->match_streak = 0;
        }
        if (m->total > 500.0) {
            m->hits *= 0.99;
            m->total *= 0.99;
        }
    }

    /* store context → position */
    int n = m->hist_len;
    for (int tidx = 0; tidx < MATCH_N_CTX; tidx++) {
        int ctx_len = m->ctx_lens[tidx];
        if (n >= ctx_len) {
            uint64_t key = match_ctx_hash(m->history + n - ctx_len, ctx_len);
            mht_set(&m->tables[tidx], key, (uint32_t)n,
                     &m->table_used[tidx]);
        }
    }

    /* append to history */
    if (m->hist_len >= m->hist_cap) {
        m->hist_cap *= 2;
        m->history = (uint8_t *)realloc(m->history, m->hist_cap);
    }
    m->history[m->hist_len++] = actual_byte;
}

static inline void blend_match(double *probs, int match_byte,
                                double match_confidence) {
    if (match_byte < 0) return;
    double weight = match_confidence * 0.85;
    if (weight > 0.95) weight = 0.95;
    for (int i = 0; i < MATCH_NSYM; i++)
        probs[i] *= (1.0 - weight);
    probs[match_byte] += weight;
    double sum = 0.0;
    for (int i = 0; i < MATCH_NSYM; i++) {
        if (probs[i] < 1e-10) probs[i] = 1e-10;
        sum += probs[i];
    }
    double inv = 1.0 / sum;
    for (int i = 0; i < MATCH_NSYM; i++)
        probs[i] *= inv;
}

#endif /* MATCH_H */
