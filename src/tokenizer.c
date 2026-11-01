// Byte-level BPE, as used by Qwen. Text is split into words with the Qwen
// pre-tokenizer rules, each word starts as single bytes, and adjacent pairs
// are merged in rank order until no merge applies.
#include "abacus.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static int cmp_merge(const void *a, const void *b) {
    uint64_t x = ((const Merge *)a)->key, y = ((const Merge *)b)->key;
    return (x > y) - (x < y);
}

// Reads the tokenizer section of the model file; returns the first byte after it.
const uint8_t *tokenizer_load(Tokenizer *t, const uint8_t *p) {
    int32_t n[3];
    memcpy(n, p, sizeof n);
    p += sizeof n;
    t->n_tokens = n[0], t->n_added = n[1], t->n_merges = n[2];
    t->piece = calloc(t->n_tokens, sizeof(char *));
    t->len = calloc(t->n_tokens, sizeof(int));
    for (int i = 0; i < t->n_tokens; i++) {
        uint16_t len;
        memcpy(&len, p, 2);
        t->len[i] = len;
        t->piece[i] = malloc(len + 1);
        memcpy(t->piece[i], p + 2, len);
        t->piece[i][len] = 0;
        if (len == 1) t->byte_token[(uint8_t)p[2]] = i;
        p += 2 + len;
    }
    t->added = malloc(t->n_added * sizeof(int));
    memcpy(t->added, p, t->n_added * sizeof(int));
    p += t->n_added * sizeof(int);
    t->merges = malloc(t->n_merges * sizeof(Merge));
    for (int i = 0; i < t->n_merges; i++, p += 12) {
        int32_t lrr[3];
        memcpy(lrr, p, 12);
        t->merges[i] = (Merge){(uint64_t)lrr[0] << 32 | (uint32_t)lrr[1], i, lrr[2]};
    }
    qsort(t->merges, t->n_merges, sizeof(Merge), cmp_merge);
    return p;
}

const char *tokenizer_piece(const Tokenizer *t, int id, int *len) {
    if (id < 0 || id >= t->n_tokens) {
        *len = 0;
        return "";
    }
    *len = t->len[id];
    return t->piece[id];
}

int tokenizer_is_special(const Tokenizer *t, int id) {
    for (int i = 0; i < t->n_added; i++)
        if (t->added[i] == id) return 1;
    return 0;
}

// ---------------------------------------------------------------- pre-tokenizer
// Qwen splits text with the regex
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// next_word() follows it alternative by alternative. Letters and digits are
// exact for ASCII; other code points count as letters except common punctuation,
// symbol and emoji blocks.

enum { LETTER, DIGIT, SPACE, OTHER };

static int char_len(const uint8_t *s) {
    return *s < 0x80 ? 1 : *s < 0xE0 ? 2 : *s < 0xF0 ? 3 : 4;
}

static int char_class(const uint8_t *s) {
    uint32_t c = *s;
    if (c < 0x80) {
        if ((c | 32) >= 'a' && (c | 32) <= 'z') return LETTER;
        if (c >= '0' && c <= '9') return DIGIT;
        if (c == ' ' || (c >= '\t' && c <= '\r')) return SPACE;
        return OTHER;
    }
    int n = char_len(s);
    c = n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
    for (int i = 1; i < n; i++) c = c << 6 | (s[i] & 0x3F);
    if ((c >= 0xA0 && c <= 0xBF) || c == 0xD7 || c == 0xF7) return OTHER;   // Latin-1 symbols
    if ((c >= 0x2000 && c <= 0x2BFF) || (c >= 0x3000 && c <= 0x303F)) return OTHER; // punctuation, symbols
    if (c >= 0x1F000) return OTHER;                                         // emoji
    return LETTER;
}

static int is_newline(uint8_t c) { return c == '\r' || c == '\n'; }

// Returns the end of the word starting at s[i] (s has length n).
static int next_word(const uint8_t *s, int i, int n) {
    int k = char_class(s + i);
    // 's 't 're 've 'm 'll 'd
    if (s[i] == '\'' && i + 1 < n) {
        int a = s[i + 1] | 32, b = i + 2 < n ? s[i + 2] | 32 : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    int j = i;
    if (k != LETTER && k != DIGIT && !is_newline(s[i]) && i + char_len(s + i) < n &&
        char_class(s + i + char_len(s + i)) == LETTER)
        j += char_len(s + i);
    if (char_class(s + j) == LETTER) {
        while (j < n && char_class(s + j) == LETTER) j += char_len(s + j);
        return j;
    }
    // \p{N}
    if (k == DIGIT) return i + 1;
    // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
    j = i + (s[i] == ' ' && i + 1 < n && char_class(s + i + 1) == OTHER);
    if (char_class(s + j) == OTHER) {
        while (j < n && char_class(s + j) == OTHER) j += char_len(s + j);
        while (j < n && is_newline(s[j])) j++;
        return j;
    }
    // whitespace run [i, end)
    int end = i;
    while (end < n && char_class(s + end) == SPACE) end++;
    // \s*[\r\n]+ : up to the last newline in the run
    for (int e = end - 1; e >= i; e--)
        if (is_newline(s[e])) return e + 1;
    // \s+(?!\S) : leave the last space to the following word
    if (end < n && end - i > 1) return end - 1;
    return end; // \s+
}

// ---------------------------------------------------------------- BPE

static const Merge *find_merge(const Tokenizer *t, int left, int right) {
    Merge key = {(uint64_t)left << 32 | (uint32_t)right, 0, 0};
    return bsearch(&key, t->merges, t->n_merges, sizeof(Merge), cmp_merge);
}

static int bpe_word(const Tokenizer *t, const uint8_t *s, int n, int *out) {
    for (int i = 0; i < n; i++) out[i] = t->byte_token[s[i]];
    while (n > 1) {
        int best = -1, best_rank = INT_MAX, merged = 0;
        for (int i = 0; i + 1 < n; i++) {
            const Merge *mg = find_merge(t, out[i], out[i + 1]);
            if (mg && mg->rank < best_rank) best = i, best_rank = mg->rank, merged = mg->result;
        }
        if (best < 0) break;
        out[best] = merged;
        memmove(out + best + 1, out + best + 2, (n - best - 2) * sizeof(int));
        n--;
    }
    return n;
}

static int match_added(const Tokenizer *t, const char *s) {
    for (int i = 0; i < t->n_added; i++) {
        int id = t->added[i];
        if (strncmp(s, t->piece[id], t->len[id]) == 0) return id;
    }
    return -1;
}

// Encodes text into at most max tokens; returns the count.
int tokenizer_encode(const Tokenizer *t, const char *text, int *out, int max) {
    const uint8_t *s = (const uint8_t *)text;
    int n = (int)strlen(text), count = 0;
    int *word = malloc((n + 1) * sizeof(int));
    for (int i = 0; i < n;) {
        int special = match_added(t, text + i);
        if (special >= 0) {
            if (count < max) out[count++] = special;
            i += t->len[special];
            continue;
        }
        // stop the word before the next special token
        int stop = i + 1;
        while (stop < n && match_added(t, text + stop) < 0) stop++;
        int end = next_word(s, i, stop);
        int k = bpe_word(t, s + i, end - i, word);
        for (int j = 0; j < k && count < max; j++) out[count++] = word[j];
        i = end;
    }
    free(word);
    return count;
}
