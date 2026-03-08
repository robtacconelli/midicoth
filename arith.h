#ifndef ARITH_H
#define ARITH_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Arithmetic encoder */
typedef struct {
    uint32_t low;
    uint32_t high;
    int pending;
    uint8_t *buf;      /* output byte buffer */
    size_t buf_len;
    size_t buf_cap;
    int bit_buf;       /* accumulates 8 bits before flushing a byte */
    int bit_count;     /* bits in bit_buf (0..7) */
} ArithEncoder;

/* Arithmetic decoder */
typedef struct {
    const uint8_t *data;
    size_t data_len;
    size_t bit_pos;
    uint32_t low;
    uint32_t high;
    uint32_t value;
} ArithDecoder;

/* ── Encoder ── */

static inline void ae_init(ArithEncoder *e) {
    e->low = 0;
    e->high = 0xFFFFFFFF;
    e->pending = 0;
    e->buf_cap = 4096;
    e->buf_len = 0;
    e->buf = (uint8_t *)malloc(e->buf_cap);
    e->bit_buf = 0;
    e->bit_count = 0;
}

static inline void ae_flush_byte(ArithEncoder *e) {
    if (e->buf_len >= e->buf_cap) {
        e->buf_cap *= 2;
        e->buf = (uint8_t *)realloc(e->buf, e->buf_cap);
    }
    e->buf[e->buf_len++] = (uint8_t)e->bit_buf;
    e->bit_buf = 0;
    e->bit_count = 0;
}

static inline void ae_output_bit(ArithEncoder *e, int bit) {
    e->bit_buf = (e->bit_buf << 1) | bit;
    e->bit_count++;
    if (e->bit_count == 8) ae_flush_byte(e);

    int inv = 1 - bit;
    while (e->pending > 0) {
        e->bit_buf = (e->bit_buf << 1) | inv;
        e->bit_count++;
        if (e->bit_count == 8) ae_flush_byte(e);
        e->pending--;
    }
}

static inline void ae_encode(ArithEncoder *e, const int64_t *cumfreqs,
                              int symbol, int64_t total) {
    uint64_t rng = (uint64_t)e->high - e->low + 1;
    e->high = e->low + (uint32_t)((rng * cumfreqs[symbol + 1]) / total) - 1;
    e->low  = e->low + (uint32_t)((rng * cumfreqs[symbol]) / total);

    for (;;) {
        if (e->high < 0x80000000u) {
            ae_output_bit(e, 0);
        } else if (e->low >= 0x80000000u) {
            ae_output_bit(e, 1);
            e->low  -= 0x80000000u;
            e->high -= 0x80000000u;
        } else if (e->low >= 0x40000000u && e->high < 0xC0000000u) {
            e->pending++;
            e->low  -= 0x40000000u;
            e->high -= 0x40000000u;
        } else {
            break;
        }
        e->low  = (e->low << 1) & 0xFFFFFFFF;
        e->high = ((e->high << 1) | 1) & 0xFFFFFFFF;
    }
}

static inline void ae_finish(ArithEncoder *e) {
    e->pending++;
    if (e->low < 0x40000000u)
        ae_output_bit(e, 0);
    else
        ae_output_bit(e, 1);

    /* pad remaining bits in the last byte */
    if (e->bit_count > 0) {
        e->bit_buf <<= (8 - e->bit_count);
        ae_flush_byte(e);
    }
}

static inline void ae_free(ArithEncoder *e) {
    free(e->buf);
    e->buf = NULL;
}

/* ── Decoder ── */

static inline int ad_read_bit(ArithDecoder *d) {
    size_t byte_idx = d->bit_pos / 8;
    if (byte_idx >= d->data_len) {
        d->bit_pos++;
        return 0;
    }
    int bit = (d->data[byte_idx] >> (7 - (d->bit_pos % 8))) & 1;
    d->bit_pos++;
    return bit;
}

static inline void ad_init(ArithDecoder *d, const uint8_t *data, size_t len) {
    d->data = data;
    d->data_len = len;
    d->bit_pos = 0;
    d->low = 0;
    d->high = 0xFFFFFFFF;
    d->value = 0;
    for (int i = 0; i < 32; i++)
        d->value = (d->value << 1) | ad_read_bit(d);
}

static inline int ad_decode(ArithDecoder *d, const int64_t *cumfreqs,
                             int64_t total) {
    uint64_t rng = (uint64_t)d->high - d->low + 1;
    int64_t scaled = (int64_t)(((uint64_t)(d->value - d->low + 1) * total - 1) / rng);

    /* linear search (matches Python behavior) */
    int sym = 0;
    for (sym = 0; sym < 256; sym++) {
        if (cumfreqs[sym + 1] > scaled)
            break;
    }

    d->high = d->low + (uint32_t)((rng * cumfreqs[sym + 1]) / total) - 1;
    d->low  = d->low + (uint32_t)((rng * cumfreqs[sym]) / total);

    for (;;) {
        if (d->high < 0x80000000u) {
            /* nothing */
        } else if (d->low >= 0x80000000u) {
            d->low   -= 0x80000000u;
            d->high  -= 0x80000000u;
            d->value -= 0x80000000u;
        } else if (d->low >= 0x40000000u && d->high < 0xC0000000u) {
            d->low   -= 0x40000000u;
            d->high  -= 0x40000000u;
            d->value -= 0x40000000u;
        } else {
            break;
        }
        d->low   = (d->low << 1) & 0xFFFFFFFF;
        d->high  = ((d->high << 1) | 1) & 0xFFFFFFFF;
        d->value = ((d->value << 1) | ad_read_bit(d)) & 0xFFFFFFFF;
    }
    return sym;
}

#endif /* ARITH_H */
