#ifndef WORD_H
#define WORD_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define WORD_NSYM 256

/* ── Word character set ── */
static inline int is_word_char(int c) {
    if (c >= 'a' && c <= 'z') return 1;
    if (c >= 'A' && c <= 'Z') return 1;
    if (c >= '0' && c <= '9') return 1;
    if (c == '\'' || c == '-') return 1;
    return 0;
}

/* ── Trie node ── */
typedef struct TrieNode {
    /* continuations: next_byte → count */
    int cont_keys[64];
    int cont_vals[64];
    int cont_count;

    /* children: byte → child node */
    struct TrieNode *children[256];
} TrieNode;

static inline TrieNode *trie_new(void) {
    TrieNode *n = (TrieNode *)calloc(1, sizeof(TrieNode));
    return n;
}

static inline void trie_free(TrieNode *n) {
    if (!n) return;
    for (int i = 0; i < 256; i++)
        trie_free(n->children[i]);
    free(n);
}

static inline void trie_add_cont(TrieNode *n, int byte_val) {
    for (int i = 0; i < n->cont_count; i++) {
        if (n->cont_keys[i] == byte_val) {
            n->cont_vals[i]++;
            return;
        }
    }
    if (n->cont_count < 64) {
        n->cont_keys[n->cont_count] = byte_val;
        n->cont_vals[n->cont_count] = 1;
        n->cont_count++;
    }
}

/* ── Word counts hash table ── */
typedef struct {
    uint64_t key;
    int count;
} WordCountEntry;

typedef struct {
    WordCountEntry *entries;
    uint32_t capacity;
    uint32_t mask;
    uint32_t used;
} WordCountHT;

static inline void wcht_init(WordCountHT *t, uint32_t cap) {
    t->capacity = cap;
    t->mask = cap - 1;
    t->used = 0;
    t->entries = (WordCountEntry *)calloc(cap, sizeof(WordCountEntry));
}

static inline void wcht_free(WordCountHT *t) {
    free(t->entries);
}

static inline uint64_t word_hash(const uint8_t *w, int len) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) {
        h ^= w[i];
        h *= 1099511628211ULL;
    }
    if (h == 0) h = 1;
    return h;
}

static inline void wcht_grow(WordCountHT *t) {
    uint32_t old_cap = t->capacity;
    WordCountEntry *old = t->entries;
    t->capacity *= 2;
    t->mask = t->capacity - 1;
    t->entries = (WordCountEntry *)calloc(t->capacity, sizeof(WordCountEntry));
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

static inline int wcht_get(WordCountHT *t, uint64_t key) {
    uint32_t idx = (uint32_t)(key & t->mask);
    for (;;) {
        if (t->entries[idx].key == key) return t->entries[idx].count;
        if (t->entries[idx].key == 0) return 0;
        idx = (idx + 1) & t->mask;
    }
}

static inline void wcht_add(WordCountHT *t, uint64_t key, int delta) {
    if (t->used * 5 > t->capacity * 3) wcht_grow(t);
    uint32_t idx = (uint32_t)(key & t->mask);
    for (;;) {
        if (t->entries[idx].key == key) {
            t->entries[idx].count += delta;
            return;
        }
        if (t->entries[idx].key == 0) {
            t->entries[idx].key = key;
            t->entries[idx].count = delta;
            t->used++;
            return;
        }
        idx = (idx + 1) & t->mask;
    }
}

/* ── Bigram table: word_hash → { byte → count } ── */
typedef struct {
    uint64_t key;
    int counts[256];
    int total;
} BigramEntry;

typedef struct {
    BigramEntry *entries;
    uint32_t capacity;
    uint32_t mask;
    uint32_t used;
} BigramHT;

static inline void bht_init(BigramHT *t, uint32_t cap) {
    t->capacity = cap;
    t->mask = cap - 1;
    t->used = 0;
    t->entries = (BigramEntry *)calloc(cap, sizeof(BigramEntry));
}

static inline void bht_free(BigramHT *t) {
    free(t->entries);
}

static inline void bht_grow(BigramHT *t) {
    uint32_t old_cap = t->capacity;
    BigramEntry *old = t->entries;
    t->capacity *= 2;
    t->mask = t->capacity - 1;
    t->entries = (BigramEntry *)calloc(t->capacity, sizeof(BigramEntry));
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

static inline BigramEntry *bht_get_or_create(BigramHT *t, uint64_t key) {
    if (t->used * 5 > t->capacity * 3) bht_grow(t);
    uint32_t idx = (uint32_t)(key & t->mask);
    for (;;) {
        if (t->entries[idx].key == key) return &t->entries[idx];
        if (t->entries[idx].key == 0) {
            t->entries[idx].key = key;
            memset(t->entries[idx].counts, 0, sizeof(t->entries[idx].counts));
            t->entries[idx].total = 0;
            t->used++;
            return &t->entries[idx];
        }
        idx = (idx + 1) & t->mask;
    }
}

/* ── Word Model ── */

typedef struct {
    TrieNode *trie;
    WordCountHT word_counts;
    BigramHT bigrams;

    uint8_t current_word[256];
    int current_word_len;

    uint8_t last_word[256];
    int last_word_len;
    int has_last_word;

    int in_word;
    double hits;
    double attempts;

    /* prediction cache to avoid double trie traversal */
    double cached_probs[WORD_NSYM];
    double cached_conf;
    int cache_valid;
} WordModel;

static inline void word_init(WordModel *w) {
    w->trie = trie_new();
    wcht_init(&w->word_counts, 4096);
    bht_init(&w->bigrams, 2048);
    w->current_word_len = 0;
    w->last_word_len = 0;
    w->has_last_word = 0;
    w->in_word = 0;
    w->hits = 1.0;
    w->attempts = 2.0;
    w->cache_valid = 0;
}

static inline void word_free(WordModel *w) {
    trie_free(w->trie);
    wcht_free(&w->word_counts);
    bht_free(&w->bigrams);
}

static inline void word_add_to_trie(WordModel *w, const uint8_t *word, int len) {
    uint64_t wh = word_hash(word, len);
    wcht_add(&w->word_counts, wh, 1);

    TrieNode *node = w->trie;
    for (int i = 0; i < len; i++) {
        int b = word[i];
        if (!node->children[b])
            node->children[b] = trie_new();
        TrieNode *entry = node->children[b];
        if (i + 1 < len)
            trie_add_cont(entry, word[i + 1]);
        node = entry;
    }
}

/* Get continuations for a prefix. Returns count of distinct continuations.
   Fills keys[] and vals[] arrays. */
static inline int word_get_continuations(WordModel *w, const uint8_t *prefix,
                                          int prefix_len, int *keys, int *vals) {
    if (prefix_len == 0) return 0;
    TrieNode *node = w->trie;
    for (int i = 0; i < prefix_len; i++) {
        int b = prefix[i];
        if (!node->children[b]) return 0;
        TrieNode *entry = node->children[b];
        if (i == prefix_len - 1) {
            int n = entry->cont_count;
            for (int j = 0; j < n; j++) {
                keys[j] = entry->cont_keys[j];
                vals[j] = entry->cont_vals[j];
            }
            return n;
        }
        node = entry;
    }
    return 0;
}

/*
 * predict: fills probs[256] if prediction available.
 * Returns 1 with confidence in *out_conf, or 0 if no prediction.
 */
static inline int word_predict(WordModel *w, double *probs, double *out_conf) {
    static const int boundary_chars[] = {32, 10, 13, 44, 46, 59, 58, 33, 63, 41, 93};
    static const int n_boundary = 11;

    if (w->in_word && w->current_word_len >= 1) {
        int keys[64], vals[64];
        int nc = word_get_continuations(w, w->current_word,
                                         w->current_word_len, keys, vals);
        if (nc > 0) {
            memset(probs, 0, WORD_NSYM * sizeof(double));
            int total = 0;
            for (int i = 0; i < nc; i++) total += vals[i];
            double inv_total = 1.0 / total;
            for (int i = 0; i < nc; i++)
                probs[keys[i]] += vals[i] * inv_total;

            /* word boundary probability */
            uint64_t wh = word_hash(w->current_word, w->current_word_len);
            int wcount = wcht_get(&w->word_counts, wh);
            if (wcount > 0) {
                double bw = (double)wcount / (wcount + total);
                for (int i = 0; i < WORD_NSYM; i++)
                    probs[i] *= (1.0 - bw);
                for (int i = 0; i < n_boundary; i++)
                    probs[boundary_chars[i]] += bw / n_boundary;
            }

            int plen = w->current_word_len;
            double confidence = (plen / 3.0 < 1.0 ? plen / 3.0 : 1.0);
            double cont_factor = nc * 0.5;
            if (cont_factor > 1.0) cont_factor = 1.0;
            confidence *= cont_factor;
            confidence *= (w->hits / w->attempts);

            double sum = 0.0;
            for (int i = 0; i < WORD_NSYM; i++) sum += probs[i];
            if (sum > 0.0) {
                double inv = 1.0 / sum;
                for (int i = 0; i < WORD_NSYM; i++) probs[i] *= inv;
                *out_conf = confidence;
                return 1;
            }
        }
    } else if (!w->in_word && w->has_last_word) {
        uint64_t wh = word_hash(w->last_word, w->last_word_len);
        BigramEntry *be = NULL;
        /* look up without creating */
        uint32_t idx = (uint32_t)(wh & w->bigrams.mask);
        for (;;) {
            if (w->bigrams.entries[idx].key == wh) {
                be = &w->bigrams.entries[idx];
                break;
            }
            if (w->bigrams.entries[idx].key == 0) break;
            idx = (idx + 1) & w->bigrams.mask;
        }
        if (be && be->total > 0) {
            memset(probs, 0, WORD_NSYM * sizeof(double));
            double inv = 1.0 / be->total;
            for (int i = 0; i < WORD_NSYM; i++)
                if (be->counts[i] > 0)
                    probs[i] = be->counts[i] * inv;

            double confidence = (be->total / 5.0 < 1.0 ? be->total / 5.0 : 1.0);
            confidence *= 0.3 * (w->hits / w->attempts);

            double sum = 0.0;
            for (int i = 0; i < WORD_NSYM; i++) sum += probs[i];
            if (sum > 0.0) {
                double inv2 = 1.0 / sum;
                for (int i = 0; i < WORD_NSYM; i++) probs[i] *= inv2;
                *out_conf = confidence;
                return 1;
            }
        }
    }

    *out_conf = 0.0;
    return 0;
}

/* Predict with caching: compute once, reuse in update */
static inline int word_predict_cached(WordModel *w, double *probs, double *out_conf) {
    if (w->cache_valid) {
        memcpy(probs, w->cached_probs, sizeof(w->cached_probs));
        *out_conf = w->cached_conf;
        return (*out_conf > 0.0) ? 1 : 0;
    }
    int ret = word_predict(w, probs, out_conf);
    if (ret) {
        memcpy(w->cached_probs, probs, sizeof(w->cached_probs));
        w->cached_conf = *out_conf;
    } else {
        w->cached_conf = 0.0;
    }
    w->cache_valid = 1;
    return ret;
}

static inline void word_update(WordModel *w, uint8_t byte_val) {
    /* track accuracy using cached prediction */
    double pred_conf = w->cached_conf;
    int has_pred = w->cache_valid && pred_conf > 0.01;
    if (has_pred) {
        w->attempts += 1.0;
        if (w->cached_probs[byte_val] > 0.05)
            w->hits += 1.0;
        if (w->attempts > 500.0) {
            w->hits *= 0.99;
            w->attempts *= 0.99;
        }
    }

    int is_wc = is_word_char(byte_val);
    if (is_wc) {
        if (!w->in_word) {
            w->current_word_len = 0;
            w->in_word = 1;
            /* bigram: last_word → first byte of new word */
            if (w->has_last_word) {
                uint64_t wh = word_hash(w->last_word, w->last_word_len);
                BigramEntry *be = bht_get_or_create(&w->bigrams, wh);
                be->counts[byte_val]++;
                be->total++;
            }
        }
        if (w->current_word_len < 255)
            w->current_word[w->current_word_len++] = byte_val;
    } else {
        if (w->in_word && w->current_word_len >= 2) {
            word_add_to_trie(w, w->current_word, w->current_word_len);
            w->last_word_len = w->current_word_len;
            memcpy(w->last_word, w->current_word, w->current_word_len);
            w->has_last_word = 1;
        } else if (w->in_word) {
            w->last_word_len = w->current_word_len;
            memcpy(w->last_word, w->current_word, w->current_word_len);
            w->has_last_word = 1;
        }
        w->in_word = 0;
        w->current_word_len = 0;
    }
    w->cache_valid = 0; /* invalidate cache after state change */
}

static inline void blend_word_model(double *probs, const double *word_probs,
                                     double word_confidence) {
    if (word_confidence < 0.01) return;
    double weight = word_confidence * 0.35;
    if (weight > 0.45) weight = 0.45;
    double sum = 0.0;
    for (int i = 0; i < WORD_NSYM; i++) {
        probs[i] = probs[i] * (1.0 - weight) + word_probs[i] * weight;
        if (probs[i] < 1e-10) probs[i] = 1e-10;
        sum += probs[i];
    }
    double inv = 1.0 / sum;
    for (int i = 0; i < WORD_NSYM; i++)
        probs[i] *= inv;
}

#endif /* WORD_H */
