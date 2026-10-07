#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#ifdef ACCELERATE
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

#define DP    "duration_predictor.tts.dp"
#define TE    "text_encoder.tts.ttl"
#define FIELD "vector_estimator.vector_estimator.tts.ttl.vector_field"
#define MASK  "vector_estimator.vector_estimator.tts.ttl.uncond_masker"
#define AE    "vocoder.tts.ae"

enum { name_capacity = 192, text_capacity = 1024, arena_floats = 1 << 26,
       sample_rate = 44100, style_heads = 2, field_groups = 4,
       chunk_limit = 300, korean_chunk_limit = 120, span_capacity = 256,
       arena_align = 32, widen_floats = 1 << 18, vocoder_slice = 256,
       q4_block = 32, q4_group = 8, q4_head = 20 };

enum { module_duration, module_text, module_field, module_vocoder,
       module_other, module_count };

static const char * const module_names[module_count] = {
    "duration", "text", "field", "vocoder", "other" };

static const int32_t encoder_dilations[] = { 1, 1, 2, 2, 4, 4 };
static const int32_t field_dilations[]   = { 1, 2, 4, 8 };
static const int32_t vocoder_dilations[] = { 1, 2, 4, 1, 2, 4, 1, 1, 1, 1 };
static const float   layer_norm_epsilon  = 1e-6f;
static const float   batch_norm_epsilon  = 1e-5f;
static const float   time_scale          = 1000.0f;
static const float   guidance            = 3.0f;
static const double  gate_tolerance      = 1e-3;
static const double  gate_cosine         = 0.99999;
static const double  chunk_silence       = 0.3;

struct record {
    char    name[120];
    int32_t bits;
    int32_t spare;
    int32_t shape[4];
    int64_t count;
    int64_t offset;
};

struct tensor {
    const char *  name;
    const float * data;
    int64_t       count;
    int32_t       shape[4];
    int32_t       bits;
};

struct pack {
    struct tensor * tensors;
    int64_t         count;
};

struct span {
    const char * kernel;
    int32_t      module;
    int32_t      shape[4];
    int64_t      calls;
    double       seconds;
};

struct profile {
    struct span spans[span_capacity];
    int32_t     count;
    int32_t     module;
    double      entered;
    double      walls[module_count];
};

struct tts {
    struct pack      weights;
    struct pack      golden;
    struct profile * profile;
    float *          arena;
    size_t           used;
    int32_t          taps;
    int32_t          failures;
};

struct condition {
    const float * text;
    int32_t       length;
    const float * keys;
    const float * values;
    int32_t       tokens;
    const char *  tag;
};

struct voice {
    const float * ttl;
    const float * dp;
    int32_t       tokens;
};

struct audio {
    float * samples;
    int32_t count;
    float   seconds;
};

[[noreturn]] static void fatal(const char * format, ...) {
    va_list arguments;
    va_start(arguments, format);
    fprintf(stderr, "tts: ");
    vfprintf(stderr, format, arguments);
    fprintf(stderr, "\n");
    va_end(arguments);
    exit(1);
}

static double seconds_now(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

struct mt19937 {
    uint32_t key[624];
    int32_t  pos;
    bool     has_gauss;
    double   gauss;
};

static void mt_seed(struct mt19937 * mt, uint32_t seed) {
    mt->key[0] = seed;
    for (int32_t i = 1; i < 624; i++) {
        uint32_t prev = mt->key[i - 1];
        mt->key[i] = 1812433253u * (prev ^ (prev >> 30)) + (uint32_t)i;
    }
    mt->pos       = 624;
    mt->has_gauss = false;
    mt->gauss     = 0.0;
}

static void mt_twist(struct mt19937 * mt) {
    for (int32_t i = 0; i < 624; i++) {
        uint32_t y = (mt->key[i] & 0x80000000u) |
                     (mt->key[(i + 1) % 624] & 0x7fffffffu);
        uint32_t mag = (y & 1u) != 0 ? 0x9908b0dfu : 0u;
        mt->key[i] = mt->key[(i + 397) % 624] ^ (y >> 1) ^ mag;
    }
    mt->pos = 0;
}

static uint32_t mt_next(struct mt19937 * mt) {
    if (mt->pos >= 624) { mt_twist(mt); }
    uint32_t y = mt->key[mt->pos];
    mt->pos++;
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

static double mt_double(struct mt19937 * mt) {
    uint32_t a = mt_next(mt) >> 5;
    uint32_t b = mt_next(mt) >> 6;
    return ((double)a * 67108864.0 + (double)b) / 9007199254740992.0;
}

static double mt_gauss(struct mt19937 * mt) {
    double result = mt->gauss;
    if (mt->has_gauss) {
        mt->has_gauss = false;
        mt->gauss     = 0.0;
    } else {
        double x1 = 0.0;
        double x2 = 0.0;
        double r2 = 0.0;
        do {
            x1 = 2.0 * mt_double(mt) - 1.0;
            x2 = 2.0 * mt_double(mt) - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);
        double f = sqrt(-2.0 * log(r2) / r2);
        mt->gauss     = f * x1;
        mt->has_gauss = true;
        result        = f * x2;
    }
    return result;
}

struct text {
    uint32_t * cp;
    uint32_t * spare;
    int32_t    count;
};

static int32_t text_decode(const char * utf8, uint32_t * cp) {
    const uint8_t * s     = (const uint8_t *)utf8;
    int32_t         count = 0;
    while (*s != 0) {
        int32_t  extra = *s < 0x80 ? 0 : *s < 0xE0 ? 1 : *s < 0xF0 ? 2 : 3;
        uint32_t value = extra == 0 ? *s : *s & (0x3Fu >> extra);
        s++;
        for (int32_t i = 0; i < extra && (*s & 0xC0) == 0x80; i++) {
            value = value << 6 | (*s & 0x3Fu);
            s++;
        }
        cp[count++] = value;
    }
    return count;
}

static int32_t text_decompose(const uint32_t * raw, int32_t count,
                              const int32_t * offsets,
                              const int32_t * parts, uint32_t * cp) {
    int32_t made = 0;
    for (int32_t i = 0; i < count; i++) {
        const bool    basic = raw[i] <= 0xFFFF;
        const int32_t from  = basic ? offsets[raw[i]] : 0;
        const int32_t size  = basic ? offsets[raw[i] + 1] - from : 0;
        if (cp != NULL && size == 0) { cp[made] = raw[i]; }
        for (int32_t k = 0; cp != NULL && k < size; k++) {
            cp[made + k] = (uint32_t)parts[from + k];
        }
        made += size > 0 ? size : 1;
    }
    return made;
}

static int32_t text_class(const int32_t * classes, uint32_t cp) {
    return cp <= 0xFFFF ? classes[cp] : 0;
}

static void text_reorder(struct text * t, const int32_t * classes) {
    for (int32_t i = 1; i < t->count; i++) {
        const uint32_t cp   = t->cp[i];
        const int32_t  rank = text_class(classes, cp);
        int32_t        j    = i;
        while (j > 0 && rank > 0 && text_class(classes, t->cp[j - 1]) > rank) {
            t->cp[j] = t->cp[j - 1];
            j--;
        }
        t->cp[j] = cp;
    }
}

static bool text_among(uint32_t cp, const uint32_t * set, int32_t count) {
    int32_t i = 0;
    while (i < count && set[i] != cp) { i++; }
    return i < count;
}

static bool text_is_emoji(uint32_t cp) {
    return (0x1F300 <= cp && cp <= 0x1F64F) ||
           (0x1F680 <= cp && cp <= 0x1F8FF) ||
           (0x1F900 <= cp && cp <= 0x1FAFF) ||
           (0x1F1E6 <= cp && cp <= 0x1F1FF) ||
           (0x2600 <= cp && cp <= 0x27BF);
}

static bool text_is_space(uint32_t cp) {
    return (0x09 <= cp && cp <= 0x0D) || (0x1C <= cp && cp <= 0x20) ||
           (0x2000 <= cp && cp <= 0x200A) || cp == 0x85 || cp == 0xA0 ||
           cp == 0x1680 || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

static int32_t text_spaces(const uint32_t * cp, int32_t left) {
    int32_t i = 0;
    while (i < left && text_is_space(cp[i])) { i++; }
    return i;
}

static uint32_t text_symbol(uint32_t cp) {
    static const uint32_t map[][2] = {
        { 0x2013, '-'  }, { 0x2011, '-'  }, { 0x2014, '-'  },
        { 0x201C, '"'  }, { 0x201D, '"'  }, { 0x2018, '\'' },
        { 0x2019, '\'' }, { 0x00B4, '\'' }, { '`',    '\'' },
        { 0x00AF, ' '  }, { '_',    ' '  }, { '[',    ' '  },
        { ']',    ' '  }, { '|',    ' '  }, { '/',    ' '  },
        { '#',    ' '  }, { 0x2192, ' '  }, { 0x2190, ' '  } };
    const int32_t count = (int32_t)(sizeof(map) / sizeof(map[0]));
    int32_t       i     = 0;
    while (i < count && map[i][0] != cp) { i++; }
    return i < count ? map[i][1] : cp;
}

static void text_clean(struct text * t) {
    int32_t kept = 0;
    for (int32_t i = 0; i < t->count; i++) {
        uint32_t cp = text_symbol(t->cp[i]);
        if (!text_is_emoji(cp) && cp != 0xA9 && cp != '\\') {
            t->cp[kept++] = cp;
        }
    }
    t->count = kept;
}

static bool text_starts(const uint32_t * cp, int32_t left,
                        const char * prefix) {
    int32_t i = 0;
    while (prefix[i] != 0 && i < left && cp[i] == (uint8_t)prefix[i]) { i++; }
    return prefix[i] == 0;
}

static bool text_ends(const uint32_t * cp, int32_t count,
                      const char * suffix) {
    const int32_t size = (int32_t)strlen(suffix);
    return count >= size && text_starts(cp + count - size, size, suffix);
}

static void text_replace(struct text * t, const char * from,
                         const char * to) {
    const int32_t skip = (int32_t)strlen(from);
    int32_t       made = 0;
    int32_t       i    = 0;
    while (i < t->count) {
        if (text_starts(t->cp + i, t->count - i, from)) {
            for (int32_t k = 0; to[k] != 0; k++) {
                t->spare[made++] = (uint8_t)to[k];
            }
            i += skip;
        } else {
            t->spare[made++] = t->cp[i++];
        }
    }
    uint32_t * swapped = t->cp;
    t->cp    = t->spare;
    t->spare = swapped;
    t->count = made;
}

static void text_rewrite(struct text * t) {
    static const char * const pairs[][2] = {
        { "@", " at " }, { "e.g.,", "for example, " },
        { "i.e.,", "that is, " }, { " ,", "," }, { " .", "." },
        { " !", "!" }, { " ?", "?" }, { " ;", ";" }, { " :", ":" },
        { " '", "'" } };
    const int32_t count = (int32_t)(sizeof(pairs) / sizeof(pairs[0]));
    for (int32_t i = 0; i < count; i++) {
        text_replace(t, pairs[i][0], pairs[i][1]);
    }
}

static void text_squeeze(struct text * t) {
    int32_t  kept     = 0;
    uint32_t previous = 0;
    for (int32_t i = 0; i < t->count; i++) {
        uint32_t cp     = t->cp[i];
        bool     repeat = (cp == '"' || cp == '\'') && cp == previous;
        if (!text_is_space(cp)) {
            if (!repeat) { t->cp[kept++] = cp; }
        } else if (kept > 0 && t->cp[kept - 1] != ' ') {
            t->cp[kept++] = ' ';
        }
        previous = cp;
    }
    if (kept > 0 && t->cp[kept - 1] == ' ') { kept--; }
    t->count = kept;
}

static void text_end_sentence(struct text * t) {
    static const uint32_t ending[] = {
        '.', '!', '?', ';', ':', ',', '\'', '"', ')', ']', '}', 0x2026,
        0x3002, 0x300D, 0x300F, 0x3011, 0x3009, 0x300B, 0x203A, 0x00BB };
    const int32_t count = (int32_t)(sizeof(ending) / sizeof(ending[0]));
    if (t->count == 0 || !text_among(t->cp[t->count - 1], ending, count)) {
        t->cp[t->count++] = '.';
    }
}

static void text_wrap(struct text * t, const char * lang) {
    char open[16];
    char close[16];
    snprintf(open, sizeof(open), "<%s>", lang);
    snprintf(close, sizeof(close), "</%s>", lang);
    const int32_t head = (int32_t)strlen(open);
    const int32_t tail = (int32_t)strlen(close);
    memmove(t->cp + head, t->cp, (size_t)t->count * sizeof(uint32_t));
    for (int32_t i = 0; i < head; i++) { t->cp[i] = (uint8_t)open[i]; }
    for (int32_t i = 0; i < tail; i++) {
        t->cp[head + t->count + i] = (uint8_t)close[i];
    }
    t->count += head + tail;
}

static void text_lookup(const struct text * t, const int32_t * indexer,
                        int32_t * ids, int32_t capacity) {
    if (t->count > capacity) {
        fatal("text of %d characters exceeds %d", t->count, capacity);
    }
    for (int32_t i = 0; i < t->count; i++) {
        uint32_t cp = t->cp[i];
        ids[i] = cp <= 0xFFFF ? indexer[cp] : -1;
        if (ids[i] < 0) { fatal("unsupported character U+%04X", cp); }
    }
}

static int tensor_order(const void * left, const void * right) {
    const struct tensor * a = left;
    const struct tensor * b = right;
    return strcmp(a->name, b->name);
}

static struct pack pack_open(const char * path) {
    const int file = open(path, O_RDONLY);
    struct stat status;
    if (file < 0 || fstat(file, &status) != 0) {
        fatal("cannot open %s", path);
    }
    const char * map = mmap(NULL, (size_t)status.st_size, PROT_READ,
                            MAP_PRIVATE, file, 0);
    int64_t header = -1;
    if (map != MAP_FAILED) { memcpy(&header, map, sizeof header); }
    if (header < 0 || header > status.st_size - 24 ||
        memcmp(map + 8 + header, "SUPERTON", 8) != 0) {
        fatal("%s is not a pack", path);
    }
    close(file);
    const char * index = map + 8 + header;
    struct pack pack = { 0 };
    memcpy(&pack.count, index + 8, sizeof pack.count);
    const struct record * records = (const void *)(index + 16);
    pack.tensors = calloc((size_t)pack.count, sizeof(struct tensor));
    for (int64_t i = 0; i < pack.count; i++) {
        pack.tensors[i] = (struct tensor){
            .name = records[i].name, .count = records[i].count,
            .bits = records[i].bits,
            .data = (const void *)(map + records[i].offset) };
        memcpy(pack.tensors[i].shape, records[i].shape,
               sizeof records[i].shape);
    }
    return pack;
}

static const struct tensor * pack_find(const struct pack * pack,
                                       const char * format,
                                       va_list arguments) {
    char name[name_capacity];
    vsnprintf(name, sizeof name, format, arguments);
    const struct tensor key = { .name = name };
    return bsearch(&key, pack->tensors, (size_t)pack->count,
                   sizeof(struct tensor), tensor_order);
}

static bool has(const struct pack * pack, const char * format, ...) {
    va_list arguments;
    va_start(arguments, format);
    const struct tensor * tensor = pack_find(pack, format, arguments);
    va_end(arguments);
    return tensor != NULL;
}

static const struct tensor * weight(const struct pack * pack,
                                    const char * format, ...) {
    va_list arguments;
    va_start(arguments, format);
    const struct tensor * tensor = pack_find(pack, format, arguments);
    va_end(arguments);
    if (tensor == NULL) { fatal("no tensor named like %s", format); }
    return tensor;
}

static int32_t text_ids(const uint32_t * raw, int32_t count,
                        const char * lang, const struct pack * w,
                        int32_t * ids, int32_t capacity) {
    const int32_t * offsets = (const void *)weight(w, "nfkd.offsets")->data;
    const int32_t * parts   = (const void *)weight(w, "nfkd.data")->data;
    const size_t    room    = 4 * (size_t)text_decompose(raw, count, offsets,
                                                         parts, NULL) +
                              2 * strlen(lang) + 8;
    struct text     t       = { malloc(room * sizeof(uint32_t)),
                                malloc(room * sizeof(uint32_t)), 0 };
    assert(t.cp != NULL && t.spare != NULL && strlen(lang) < 12);
    t.count = text_decompose(raw, count, offsets, parts, t.cp);
    text_reorder(&t, (const void *)weight(w, "nfkd.class")->data);
    text_clean(&t);
    text_rewrite(&t);
    text_squeeze(&t);
    text_end_sentence(&t);
    text_wrap(&t, lang);
    text_lookup(&t, (const void *)weight(w, "indexer")->data, ids, capacity);
    free(t.cp);
    free(t.spare);
    return t.count;
}

struct chunks {
    uint32_t *            raw;
    uint32_t *            cp;
    const struct tensor * edges;
    int32_t               count;
    int32_t               limit;
    int32_t               at;
    int32_t               length;
};

static struct chunks chunk_open(const struct pack * w, const char * utf8,
                                int32_t limit) {
    const size_t  bytes = (strlen(utf8) + 1) * sizeof(uint32_t);
    struct chunks c     = { .raw = malloc(bytes), .cp = malloc(bytes),
                            .edges = weight(w, "word.edges"),
                            .limit = limit };
    assert(c.raw != NULL && c.cp != NULL);
    c.count = text_decode(utf8, c.raw);
    c.at    = text_spaces(c.raw, c.count);
    return c;
}

static bool chunk_is_word(const struct chunks * c, uint32_t cp) {
    const int32_t * edges = (const void *)c->edges->data;
    int64_t         i     = 0;
    while (i < c->edges->count && (uint32_t)edges[i] <= cp) { i++; }
    return i % 2 == 1;
}

static bool chunk_abbreviated(const struct chunks * c, int32_t at) {
    static const char * const known[] = {
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Ph.D.", "etc.",
        "e.g.", "i.e.", "vs.", "Inc.", "Ltd.", "Co.", "Corp.", "St.", "Ave.",
        "Blvd." };
    const int32_t    count = (int32_t)(sizeof(known) / sizeof(known[0]));
    const uint32_t * cp    = c->raw;
    int32_t          i     = 0;
    while (i < count && !text_ends(cp, at, known[i])) { i++; }
    const bool initial = at >= 2 && cp[at - 1] == '.' && 'A' <= cp[at - 2] &&
                         cp[at - 2] <= 'Z' &&
                         (at == 2 || !chunk_is_word(c, cp[at - 3]));
    return i < count || initial;
}

static bool chunk_paragraph(const struct chunks * c, int32_t from,
                            int32_t to) {
    int32_t lines = 0;
    for (int32_t i = from; i < to; i++) { lines += c->raw[i] == '\n'; }
    return lines >= 2;
}

static bool chunk_is_tag(const struct chunks * c, int32_t at) {
    static const char * const tags[] = {
        "<laugh>", "<breath>", "<surprise>", "<sigh>", "<scream>",
        "<throatclear>", "<sad>", "<angry>", "<cough>", "<yawn>" };
    const int32_t    count = (int32_t)(sizeof(tags) / sizeof(tags[0]));
    const uint32_t * cp    = c->raw + at;
    const int32_t    left  = c->count - at;
    int32_t          i     = 0;
    while (i < count && !text_starts(cp, left, tags[i])) { i++; }
    const int32_t size = i < count ? (int32_t)strlen(tags[i]) : 0;
    return i < count && (size == left || text_is_space(cp[size]));
}

static int32_t chunk_sentence(const struct chunks * c, int32_t at) {
    static const uint32_t stops[] = { '.', '!', '?' };
    const uint32_t *      cp      = c->raw;
    const bool            tag     = chunk_is_tag(c, at);
    int32_t               end     = at;
    int32_t               next    = at;
    do {
        end = next;
        while (end < c->count && !text_is_space(cp[end])) { end++; }
        next = end + text_spaces(cp + end, c->count - end);
    } while (next < c->count && !chunk_paragraph(c, end, next) &&
             chunk_is_tag(c, next) == tag &&
             (tag || !text_among(cp[end - 1], stops, 3) ||
              chunk_abbreviated(c, end)));
    return end;
}

static bool chunk_next(struct chunks * c) {
    const bool alone = chunk_is_tag(c, c->at);
    int32_t    from  = c->at;
    int32_t    end   = chunk_sentence(c, c->at);
    c->length = 0;
    while (c->at < c->count &&
           (c->length == 0 || (!alone && !chunk_is_tag(c, c->at) &&
                               !chunk_paragraph(c, from, c->at) &&
                               c->length + end - c->at + 1 <= c->limit))) {
        const int32_t size = end - c->at;
        if (c->length > 0) { c->cp[c->length++] = ' '; }
        memcpy(c->cp + c->length, c->raw + c->at,
               (size_t)size * sizeof(uint32_t));
        c->length += size;
        from  = end;
        c->at = end + text_spaces(c->raw + end, c->count - end);
        end   = chunk_sentence(c, c->at);
    }
    return c->length > 0;
}

static float * floats(struct tts * tts, int64_t count) {
    tts->used = (tts->used + arena_align - 1) / arena_align * arena_align;
    if (tts->used + (size_t)count > (size_t)arena_floats) {
        fatal("the arena of %d floats is full", arena_floats);
    }
    float * data = tts->arena + tts->used;
    tts->used += (size_t)count;
    return data;
}

static void profile_add(struct tts * tts, double started,
                        const char * kernel, int32_t out, int32_t in,
                        int32_t taps, int32_t n) {
    struct profile * p = tts->profile;
    if (p != NULL) {
        const struct span key = { .kernel = kernel, .module = p->module,
                                  .shape = { out, in, taps, n } };
        int32_t i = 0;
        while (i < p->count && (strcmp(p->spans[i].kernel, kernel) != 0 ||
                                p->spans[i].module != key.module ||
                                memcmp(p->spans[i].shape, key.shape,
                                       sizeof key.shape) != 0)) {
            i++;
        }
        if (i == p->count) {
            assert(p->count < span_capacity);
            p->spans[p->count++] = key;
        }
        p->spans[i].calls++;
        p->spans[i].seconds += seconds_now() - started;
    }
}

static void profile_module(struct tts * tts, int32_t module) {
    struct profile * p = tts->profile;
    if (p != NULL) {
        const double now = seconds_now();
        p->walls[p->module] += now - p->entered;
        p->module  = module;
        p->entered = now;
    }
}

static void compare(struct tts * tts, const char * name, const float * mine,
                    int64_t count, const struct tensor * want) {
    const bool comparable = want != NULL && want->count == count;
    double worst = 0, peak = 0, dot = 0, left = 0, right = 0;
    for (int64_t i = 0; comparable && i < count; i++) {
        const double a = mine[i];
        const double b = want->data[i];
        worst = fmax(worst, fabs(a - b));
        peak  = fmax(peak, fabs(b));
        dot   += a * b;
        left  += a * a;
        right += b * b;
    }
    const double cosine = dot / (sqrt(left) * sqrt(right) + 1e-30);
    const bool close = worst <= gate_tolerance * fmax(1.0, peak) &&
                       (count == 1 || cosine >= gate_cosine);
    const bool ok = comparable && close;
    printf("  %s %-18s n=%-7lld max|d|=%.3e peak=%.3e cos=%.8f\n",
           ok ? "ok  " : comparable ? "FAIL" : "NONE", name,
           (long long)count, worst, peak, cosine);
    tts->taps++;
    tts->failures += !ok;
}

static void tap(struct tts * tts, const float * data, int64_t count,
                const char * format, ...) {
    if (tts->golden.tensors != NULL) {
        char name[name_capacity];
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(name, sizeof name, format, arguments);
        va_end(arguments);
        va_start(arguments, format);
        compare(tts, name, data, count,
                pack_find(&tts->golden, format, arguments));
        va_end(arguments);
    }
}

static void tap_bytes(struct tts * tts, const char * name, const void * data,
                      size_t bytes, size_t width) {
    if (tts->golden.tensors != NULL) {
        const struct tensor * want = weight(&tts->golden, "%s", name);
        const bool same = (size_t)want->count * width == bytes &&
                          memcmp(want->data, data, bytes) == 0;
        printf("  %s %-18s %zu bytes %s\n", same ? "ok  " : "FAIL", name,
               bytes, same ? "identical" : "differ");
        tts->taps++;
        tts->failures += !same;
    }
}

static void fill(float * y, float value, int64_t n) {
    for (int64_t i = 0; i < n; i++) { y[i] = value; }
}

static void axpy(float * y, float a, const float * x, int64_t n) {
    for (int64_t i = 0; i < n; i++) { y[i] += a * x[i]; }
}

static float dot(const float * a, int64_t a_stride, const float * b,
                 int64_t b_stride, int32_t n) {
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        sum += a[i * a_stride] * b[i * b_stride];
    }
    return sum;
}

#ifdef ACCELERATE

static void matmul(float * y, const float * x, int32_t n, const float * w,
                   int32_t co, int32_t ci, bool transposed) {
    cblas_sgemm(CblasRowMajor, transposed ? CblasTrans : CblasNoTrans,
                CblasNoTrans, co, n, ci, 1.0f, w, transposed ? co : ci, x, n,
                1.0f, y, n);
}

static void gelu_erf(float * e, const float * x, int32_t count) {
    for (int32_t i = 0; i < count; i++) {
        const float z = x[i] / 1.4142135f;
        e[i] = -z * z;
    }
    vvexpf(e, e, &count);
    for (int32_t i = 0; i < count; i++) {
        const float t = 1.0f / (1.0f + 0.3275911f * fabsf(x[i] / 1.4142135f));
        const float tail = ((((1.061405429f * t - 1.453152027f) * t +
                              1.421413741f) * t - 0.284496736f) * t +
                            0.254829592f) * t * e[i];
        e[i] = copysignf(1.0f - tail, x[i]);
    }
}

#else

static void gelu_erf(float * e, const float * x, int32_t count) {
    for (int32_t i = 0; i < count; i++) { e[i] = erff(x[i] / 1.4142135f); }
}

static void axpy4(float * y, const float * a, int64_t step, const float * x,
                  int64_t n) {
    const float a0 = a[0];
    const float a1 = a[step];
    const float a2 = a[2 * step];
    const float a3 = a[3 * step];
    for (int64_t t = 0; t < n; t++) {
        y[t] = y[t] + a0 * x[t] + a1 * x[n + t] + a2 * x[2 * n + t] +
               a3 * x[3 * n + t];
    }
}

static void matmul(float * y, const float * x, int32_t n, const float * w,
                   int32_t co, int32_t ci, bool transposed) {
    const int64_t step = transposed ? co : 1;
    for (int32_t o = 0; o < co; o++) {
        const float * a = w + (transposed ? o : (int64_t)o * ci);
        float * row = y + (int64_t)o * n;
        for (int32_t i = 0; i + 4 <= ci; i += 4) {
            axpy4(row, a + i * step, step, x + (int64_t)i * n, n);
        }
        for (int32_t i = ci - ci % 4; i < ci; i++) {
            axpy(row, a[i * step], x + (int64_t)i * n, n);
        }
    }
}

#endif

static void unpacked(float * wide, const uint8_t * slice, int64_t blocks) {
    const uint8_t * codes = slice + (blocks + q4_group - 1) / q4_group *
                                    q4_head;
    for (int64_t i = 0; i < blocks * (q4_block / 2); i++) {
        wide[2 * i]     = (float)(codes[i] & 15);
        wide[2 * i + 1] = (float)(codes[i] >> 4);
    }
    for (int64_t b = 0; b < blocks; b++) {
        const uint8_t * head = slice + b / q4_group * q4_head;
        _Float16 unit[2];
        memcpy(unit, head, sizeof unit);
        const float scale = (float)unit[0] * (float)head[4 + b % q4_group];
        const float least = (float)unit[1] * (float)head[12 + b % q4_group];
        float * out = wide + b * q4_block;
        for (int64_t i = 0; i < q4_block; i++) {
            out[i] = scale * out[i] - least;
        }
    }
}

static const float * widened(struct tts * tts, const struct tensor * w,
                             int32_t from, int32_t size) {
    const int64_t slice = w->count / w->shape[0];
    const float * result = w->data + from * slice;
    if (w->bits == 16) {
        const _Float16 * half = (const void *)w->data;
        float * wide = floats(tts, size * slice);
        for (int64_t i = 0; i < size * slice; i++) {
            wide[i] = (float)half[from * slice + i];
        }
        result = wide;
    } else if (w->bits == 8) {
        const int8_t * codes = (const void *)w->data;
        const float * scales = weight(&tts->weights, "%s.scale", w->name)
                                   ->data;
        float * wide = floats(tts, size * slice);
        for (int64_t row = 0; row < size; row++) {
            const float    scale = scales[from + row];
            const int8_t * code = codes + (from + row) * slice;
            for (int64_t i = 0; i < slice; i++) {
                wide[row * slice + i] = scale * (float)code[i];
            }
        }
        result = wide;
    } else if (w->bits == 4) {
        const int64_t blocks = (slice + q4_block - 1) / q4_block;
        const int64_t bytes = (blocks + q4_group - 1) / q4_group * q4_head +
                              blocks * (q4_block / 2);
        const uint8_t * packed = (const void *)w->data;
        float * wide = floats(tts, size * slice + q4_block);
        for (int64_t row = 0; row < size; row++) {
            unpacked(wide + row * slice, packed + (from + row) * bytes,
                     blocks);
        }
        result = wide;
    }
    return result;
}

static void product(struct tts * tts, float * y, const float * x, int32_t n,
                    const struct tensor * w, const struct tensor * bias,
                    bool transposed) {
    const int32_t rows = w->shape[0];
    const int32_t slice = (int32_t)(w->count / rows);
    const int32_t co = transposed ? slice : rows;
    const int32_t fit = widen_floats / slice / 4 * 4;
    const int32_t block = w->bits == 0 ? rows : fit < 4 ? 4 : fit;
    const size_t  mark = tts->used;
    for (int32_t o = 0; o < co; o++) {
        fill(y + (int64_t)o * n, bias != NULL ? bias->data[o] : 0.0f, n);
    }
    for (int32_t from = 0; from < rows; from += block) {
        const int32_t size = rows - from < block ? rows - from : block;
        const float * part = widened(tts, w, from, size);
        if (transposed) {
            matmul(y, x + (int64_t)from * n, n, part, co, size, true);
        } else {
            matmul(y + (int64_t)from * n, x, n, part, size, slice, false);
        }
        tts->used = mark;
    }
}

static const float * unfolded(struct tts * tts, const float * x, int32_t ci,
                              int32_t n, int32_t k, int32_t dilation,
                              bool causal) {
    const int32_t span = dilation * (k - 1);
    const int32_t left = causal ? span : span / 2;
    float * taps = floats(tts, (int64_t)ci * k * n);
    for (int64_t row = 0; row < (int64_t)ci * k; row++) {
        const float * channel = x + row / k * n;
        const int64_t shift = row % k * dilation - left;
        const int64_t head = shift > 0 ? 0 : -shift < n ? -shift : n;
        const int64_t tail = shift < 0 ? 0 : shift < n ? shift : n;
        float * out = taps + row * n;
        fill(out, channel[0], head);
        for (int64_t t = head; t < n - tail; t++) {
            out[t] = channel[t + shift];
        }
        fill(out + n - tail, channel[n - 1], tail);
    }
    return taps;
}

static void conv(struct tts * tts, float * y, const float * x, int32_t ci,
                 int32_t n, const struct tensor * w,
                 const struct tensor * bias, int32_t dilation, bool causal) {
    const int32_t co = w->shape[0];
    const int32_t k = w->shape[2];
    const bool    depthwise = w->shape[1] == 1 && ci > 1;
    const size_t  mark = tts->used;
    const double  started = seconds_now();
    const float * taps = k == 1 ? x : unfolded(tts, x, ci, n, k, dilation,
                                               causal);
    if (depthwise) {
        for (int64_t i = 0; i < (int64_t)co * k; i++) {
            float * row = y + i / k * n;
            if (i % k == 0) { fill(row, bias->data[i / k], n); }
            axpy(row, w->data[i], taps + i * n, n);
        }
    } else {
        product(tts, y, taps, n, w, bias, false);
    }
    tts->used = mark;
    profile_add(tts, started,
                depthwise ? "depthwise" : k > 1 ? "conv k" : "conv 1x1", co,
                depthwise ? 1 : ci, k, n);
}

static void project(struct tts * tts, float * y, const float * x, int32_t n,
                    const struct tensor * w, const struct tensor * bias) {
    const double started = seconds_now();
    product(tts, y, x, n, w, bias, true);
    profile_add(tts, started, "project", w->shape[1], w->shape[0], 1, n);
}

static void layer_norm(struct tts * tts, float * x, int32_t c, int32_t n,
                       const struct tensor * gain,
                       const struct tensor * bias) {
    const size_t mark = tts->used;
    const double started = seconds_now();
    float * mean = floats(tts, n);
    float * scale = floats(tts, n);
    fill(mean, 0.0f, n);
    fill(scale, 0.0f, n);
    for (int32_t i = 0; i < c; i++) { axpy(mean, 1.0f, x + (int64_t)i * n, n); }
    for (int32_t t = 0; t < n; t++) { mean[t] /= (float)c; }
    for (int32_t i = 0; i < c; i++) {
        for (int32_t t = 0; t < n; t++) {
            const float d = x[(int64_t)i * n + t] - mean[t];
            scale[t] += d * d;
        }
    }
    for (int32_t t = 0; t < n; t++) {
        scale[t] = 1.0f / sqrtf(scale[t] / (float)c + layer_norm_epsilon);
    }
    for (int32_t i = 0; i < c; i++) {
        float * row = x + (int64_t)i * n;
        for (int32_t t = 0; t < n; t++) {
            row[t] = (row[t] - mean[t]) * scale[t] * gain->data[i] +
                     bias->data[i];
        }
    }
    tts->used = mark;
    profile_add(tts, started, "layer norm", c, 1, 1, n);
}

static void gelu(struct tts * tts, float * x, int32_t c, int32_t n) {
    const size_t mark = tts->used;
    const double started = seconds_now();
    float * e = floats(tts, (int64_t)c * n);
    gelu_erf(e, x, c * n);
    for (int64_t i = 0; i < (int64_t)c * n; i++) {
        x[i] = x[i] * (e[i] + 1.0f) * 0.5f;
    }
    tts->used = mark;
    profile_add(tts, started, "gelu", c, 1, 1, n);
}

static void prelu(float * x, int64_t n, float slope) {
    for (int64_t i = 0; i < n; i++) {
        x[i] = x[i] < 0.0f ? x[i] * slope : x[i];
    }
}

static void softmax(float * x, int32_t n) {
    float top = x[0];
    float sum = 0.0f;
    for (int32_t i = 1; i < n; i++) { top = fmaxf(top, x[i]); }
    for (int32_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - top);
        sum += x[i];
    }
    for (int32_t i = 0; i < n; i++) { x[i] /= sum; }
}

static float * transposed(struct tts * tts, const float * rows, int32_t m,
                          int32_t c) {
    float * columns = floats(tts, (int64_t)m * c);
    for (int32_t i = 0; i < m; i++) {
        for (int32_t j = 0; j < c; j++) {
            columns[(int64_t)j * m + i] = rows[(int64_t)i * c + j];
        }
    }
    return columns;
}

static float * convnext(struct tts * tts, const float * x, int32_t c,
                        int32_t n, int32_t dilation, bool causal,
                        const char * at) {
    const struct pack *   w = &tts->weights;
    const char *          dw = causal ? "dwconv.net" : "dwconv";
    const struct tensor * w1 = weight(w, "%s.pwconv1.weight", at);
    const struct tensor * gamma = weight(w, "%s.gamma", at);
    const int32_t         wide = w1->shape[0];
    float * y = floats(tts, (int64_t)c * n);
    const size_t mark = tts->used;
    float * h = floats(tts, (int64_t)c * n);
    float * z = floats(tts, (int64_t)wide * n);
    conv(tts, h, x, c, n, weight(w, "%s.%s.weight", at, dw),
         weight(w, "%s.%s.bias", at, dw), dilation, causal);
    layer_norm(tts, h, c, n, weight(w, "%s.norm.norm.weight", at),
               weight(w, "%s.norm.norm.bias", at));
    conv(tts, z, h, c, n, w1, weight(w, "%s.pwconv1.bias", at), 1, false);
    gelu(tts, z, wide, n);
    conv(tts, y, z, wide, n, weight(w, "%s.pwconv2.weight", at),
         weight(w, "%s.pwconv2.bias", at), 1, false);
    for (int64_t i = 0; i < (int64_t)c * n; i++) {
        y[i] = x[i] + gamma->data[i / n] * y[i];
    }
    tts->used = mark;
    return y;
}

static float * stack(struct tts * tts, float * x, int32_t c, int32_t n,
                     const int32_t * dilations, bool causal,
                     const char * tag, const char * at) {
    for (int32_t i = 0; has(&tts->weights, "%s.convnext.%d.gamma", at, i);
         i++) {
        char layer[name_capacity];
        snprintf(layer, sizeof layer, "%s.convnext.%d", at, i);
        x = convnext(tts, x, c, n, dilations != NULL ? dilations[i] : 1,
                     causal, layer);
        if (tag != NULL) { tap(tts, x, (int64_t)c * n, "%s.%d", tag, i); }
    }
    return x;
}

static void relative_row(float * o, float * p, const float * q,
                         const float * k, const float * v,
                         const struct tensor * rel_k,
                         const struct tensor * rel_v, int32_t n, int32_t i) {
    const int32_t d = rel_k->shape[2];
    const int32_t window = rel_k->shape[1] / 2;
    for (int32_t j = 0; j < n; j++) {
        const int32_t r = j - i + window;
        p[j] = dot(q + i, n, k + j, n, d);
        if (r >= 0 && r <= 2 * window) {
            p[j] += dot(q + i, n, rel_k->data + (int64_t)r * d, 1, d);
        }
    }
    softmax(p, n);
    for (int32_t e = 0; e < d; e++) {
        float relative = 0.0f;
        for (int32_t r = 0; r <= 2 * window; r++) {
            const int32_t j = i + r - window;
            if (j >= 0 && j < n) {
                relative += p[j] * rel_v->data[(int64_t)r * d + e];
            }
        }
        o[(int64_t)e * n + i] = dot(p, 1, v + (int64_t)e * n, 1, n) +
                                relative;
    }
}

static void relative_attention(struct tts * tts, float * y, const float * x,
                               int32_t c, int32_t n, const char * at) {
    const struct pack *   w = &tts->weights;
    const struct tensor * rel_k = weight(w, "%s.emb_rel_k", at);
    const struct tensor * rel_v = weight(w, "%s.emb_rel_v", at);
    const int32_t         d = rel_k->shape[2];
    const int64_t         size = (int64_t)c * n;
    const size_t          mark = tts->used;
    float * q = floats(tts, size);
    float * k = floats(tts, size);
    float * v = floats(tts, size);
    float * o = floats(tts, size);
    float * p = floats(tts, n);
    conv(tts, q, x, c, n, weight(w, "%s.conv_q.weight", at),
         weight(w, "%s.conv_q.bias", at), 1, false);
    conv(tts, k, x, c, n, weight(w, "%s.conv_k.weight", at),
         weight(w, "%s.conv_k.bias", at), 1, false);
    conv(tts, v, x, c, n, weight(w, "%s.conv_v.weight", at),
         weight(w, "%s.conv_v.bias", at), 1, false);
    for (int64_t i = 0; i < size; i++) { q[i] /= sqrtf((float)d); }
    const double started = seconds_now();
    for (int32_t h = 0; h < c / d; h++) {
        const int64_t base = (int64_t)h * d * n;
        for (int32_t i = 0; i < n; i++) {
            relative_row(o + base, p, q + base, k + base, v + base, rel_k,
                         rel_v, n, i);
        }
    }
    profile_add(tts, started, "relative", c, n, 2, n);
    conv(tts, y, o, c, n, weight(w, "%s.conv_o.weight", at),
         weight(w, "%s.conv_o.bias", at), 1, false);
    tts->used = mark;
}

static void feed_forward(struct tts * tts, float * y, const float * x,
                         int32_t c, int32_t n, const char * at) {
    const struct pack *   w = &tts->weights;
    const struct tensor * w1 = weight(w, "%s.conv_1.weight", at);
    const int32_t         wide = w1->shape[0];
    const size_t          mark = tts->used;
    float * h = floats(tts, (int64_t)wide * n);
    conv(tts, h, x, c, n, w1, weight(w, "%s.conv_1.bias", at), 1, false);
    prelu(h, (int64_t)wide * n, 0.0f);
    conv(tts, y, h, wide, n, weight(w, "%s.conv_2.weight", at),
         weight(w, "%s.conv_2.bias", at), 1, false);
    tts->used = mark;
}

static float * encoder(struct tts * tts, float * x, int32_t c, int32_t n,
                       const int32_t * dilations, const char * tag,
                       const char * at) {
    const struct pack * w = &tts->weights;
    const int64_t       size = (int64_t)c * n;
    char name[name_capacity];
    snprintf(name, sizeof name, "%s.convnext", tag);
    char blocks[name_capacity];
    snprintf(blocks, sizeof blocks, "%s.convnext", at);
    x = stack(tts, x, c, n, dilations, false, name, blocks);
    float * h = floats(tts, size);
    float * y = floats(tts, size);
    memcpy(h, x, (size_t)size * sizeof(float));
    for (int32_t i = 0;
         has(w, "%s.attn_encoder.attn_layers.%d.emb_rel_k", at, i); i++) {
        snprintf(name, sizeof name, "%s.attn_encoder.attn_layers.%d", at, i);
        relative_attention(tts, y, h, c, n, name);
        axpy(h, 1.0f, y, size);
        snprintf(name, sizeof name, "%s.attn_encoder.norm_layers_1.%d.norm",
                 at, i);
        layer_norm(tts, h, c, n, weight(w, "%s.weight", name),
                   weight(w, "%s.bias", name));
        tap(tts, h, size, "%s.attn.%d.mid", tag, i);
        snprintf(name, sizeof name, "%s.attn_encoder.ffn_layers.%d", at, i);
        feed_forward(tts, y, h, c, n, name);
        axpy(h, 1.0f, y, size);
        snprintf(name, sizeof name, "%s.attn_encoder.norm_layers_2.%d.norm",
                 at, i);
        layer_norm(tts, h, c, n, weight(w, "%s.weight", name),
                   weight(w, "%s.bias", name));
        tap(tts, h, size, "%s.attn.%d", tag, i);
    }
    axpy(h, 1.0f, x, size);
    return h;
}

static void embed(float * y, int32_t stride, const struct tensor * table,
                  const int32_t * ids, int32_t count) {
    const int32_t c = table->shape[1];
    for (int32_t i = 0; i < c; i++) {
        for (int32_t t = 0; t < count; t++) {
            y[(int64_t)i * stride + t] = table->data[(int64_t)ids[t] * c + i];
        }
    }
}

static float duration(struct tts * tts, const int32_t * ids, int32_t count,
                      const float * style) {
    const struct pack *   w = &tts->weights;
    const struct tensor * token = weight(w, DP ".sentence_encoder"
                                            ".sentence_token");
    const struct tensor * table = weight(w, DP ".sentence_encoder"
                                            ".text_embedder.char_embedder"
                                            ".weight");
    const struct tensor * first = weight(w, DP ".predictor.layers.0.weight");
    const int32_t         c = table->shape[1];
    const int32_t         n = count + 1;
    const size_t          mark = tts->used;
    float * x = floats(tts, (int64_t)c * n);
    float * in = floats(tts, first->shape[1]);
    float * hidden = floats(tts, first->shape[0]);
    float   out = 0.0f;
    for (int32_t i = 0; i < c; i++) { x[(int64_t)i * n] = token->data[i]; }
    embed(x + 1, n, table, ids, count);
    tap(tts, x, (int64_t)c * n, "dp.embed");
    float * h = encoder(tts, x, c, n, NULL, "dp", DP ".sentence_encoder");
    for (int32_t i = 0; i < c; i++) { hidden[i] = h[(int64_t)i * n]; }
    conv(tts, in, hidden, c, 1,
         weight(w, DP ".sentence_encoder.proj_out.net.weight"), NULL, 1,
         false);
    tap(tts, in, c, "dp.sentence");
    memcpy(in + c, style, (size_t)(first->shape[1] - c) * sizeof(float));
    conv(tts, hidden, in, first->shape[1], 1, first,
         weight(w, DP ".predictor.layers.0.bias"), 1, false);
    prelu(hidden, first->shape[0],
          weight(w, DP ".predictor.activation.weight")->data[0]);
    conv(tts, &out, hidden, first->shape[0], 1,
         weight(w, DP ".predictor.layers.1.weight"),
         weight(w, DP ".predictor.layers.1.bias"), 1, false);
    tts->used = mark;
    return expf(out);
}

static void attend(struct tts * tts, float * y, const float * q, int32_t n,
                   const float * k, const float * v, int32_t m, int32_t c,
                   int32_t heads, float scale) {
    const int32_t d = c / heads;
    const size_t  mark = tts->used;
    const double  started = seconds_now();
    float * p = floats(tts, m);
    float * o = floats(tts, d);
    const float * columns = transposed(tts, v, c, m);
    for (int32_t h = 0; h < heads; h++) {
        const int64_t base = (int64_t)h * d;
        for (int32_t i = 0; i < n; i++) {
            fill(p, 0.0f, m);
            fill(o, 0.0f, d);
            for (int32_t e = 0; e < d; e++) {
                axpy(p, q[(base + e) * n + i], k + (base + e) * m, m);
            }
            for (int32_t j = 0; j < m; j++) { p[j] /= scale; }
            softmax(p, m);
            for (int32_t j = 0; j < m; j++) {
                axpy(o, p[j], columns + (int64_t)j * c + base, d);
            }
            for (int32_t e = 0; e < d; e++) { y[(base + e) * n + i] = o[e]; }
        }
    }
    tts->used = mark;
    profile_add(tts, started, "attend", c, m, 2, n);
}

static void style_attention(struct tts * tts, float * y, const float * x,
                            int32_t n, const struct condition * condition,
                            const char * at) {
    const struct pack *   w = &tts->weights;
    const struct tensor * wq = weight(w, "%s.W_query.linear.weight", at);
    const struct tensor * wk = weight(w, "%s.W_key.linear.weight", at);
    const int32_t         c = wq->shape[1];
    const int32_t         m = condition->tokens;
    const size_t          mark = tts->used;
    float * q = floats(tts, (int64_t)c * n);
    float * k = floats(tts, (int64_t)c * m);
    float * v = floats(tts, (int64_t)c * m);
    float * o = floats(tts, (int64_t)c * n);
    project(tts, q,x, n, wq, weight(w, "%s.W_query.linear.bias", at));
    project(tts, k,condition->keys, m, wk,
            weight(w, "%s.W_key.linear.bias", at));
    project(tts, v,condition->values, m,
            weight(w, "%s.W_value.linear.weight", at),
            weight(w, "%s.W_value.linear.bias", at));
    for (int64_t i = 0; i < (int64_t)c * m; i++) { k[i] = tanhf(k[i]); }
    attend(tts, o, q, n, k, v, m, c, style_heads,
           sqrtf((float)wk->shape[0]));
    project(tts, y, o, n,weight(w, "%s.out_fc.linear.weight", at),
            weight(w, "%s.out_fc.linear.bias", at));
    tts->used = mark;
}

static float * text_encoder(struct tts * tts, const int32_t * ids, int32_t n,
                            const struct condition * condition) {
    const struct pack *   w = &tts->weights;
    const struct tensor * table = weight(w, TE ".text_encoder.text_embedder"
                                            ".char_embedder.weight");
    const int32_t         c = table->shape[1];
    const int64_t         size = (int64_t)c * n;
    float * emb = floats(tts, size);
    const size_t mark = tts->used;
    float * x = floats(tts, size);
    float * y = floats(tts, size);
    embed(x, n, table, ids, n);
    float * e = encoder(tts, x, c, n, encoder_dilations, "te",
                        TE ".text_encoder");
    tap(tts, e, size, "te.encoded");
    style_attention(tts, y, e, n, condition,
                    TE ".speech_prompted_text_encoder.attention1");
    axpy(y, 1.0f, e, size);
    tap(tts, y, size, "te.style.1");
    style_attention(tts, emb, y, n, condition,
                    TE ".speech_prompted_text_encoder.attention2");
    axpy(emb, 1.0f, e, size);
    layer_norm(tts, emb, c, n,
               weight(w, TE ".speech_prompted_text_encoder.norm.norm.weight"),
               weight(w, TE ".speech_prompted_text_encoder.norm.norm.bias"));
    tts->used = mark;
    return emb;
}

static void time_embedding(struct tts * tts, float * out, float t) {
    const struct pack *   w = &tts->weights;
    const struct tensor * w0 = weight(w, FIELD ".time_encoder.mlp.0.linear"
                                         ".weight");
    const int32_t         half = w0->shape[1] / 2;
    const int32_t         wide = w0->shape[0];
    const float *         freqs = weight(w, "time.freqs")->data;
    const size_t          mark = tts->used;
    float * waves = floats(tts, 2 * half);
    float * h = floats(tts, wide);
    for (int32_t i = 0; i < half; i++) {
        __sincosf(t * time_scale * freqs[i], &waves[i], &waves[half + i]);
    }
    conv(tts, h, waves, 2 * half, 1, w0,
         weight(w, FIELD ".time_encoder.mlp.0.linear.bias"), 1, false);
    for (int32_t i = 0; i < wide; i++) {
        h[i] = h[i] * tanhf(log1pf(expf(h[i])));
    }
    conv(tts, out, h, wide, 1,
         weight(w, FIELD ".time_encoder.mlp.2.linear.weight"),
         weight(w, FIELD ".time_encoder.mlp.2.linear.bias"), 1, false);
    tts->used = mark;
}

static void rotate(struct tts * tts, float * x, int32_t c, int32_t n,
                   const struct tensor * theta) {
    const int32_t half = (int32_t)theta->count;
    const double  started = seconds_now();
    for (int64_t i = 0; i < (int64_t)c / 2 * n; i++) {
        const int64_t pair = i / n;
        const int32_t t = (int32_t)(i % n);
        float * a = x + (pair / half * 2 * half + pair % half) * n + t;
        float * b = a + (int64_t)half * n;
        const float angle = (float)t / (float)n * theta->data[pair % half];
        float sine = 0.0f;
        float cosine = 0.0f;
        __sincosf(angle, &sine, &cosine);
        const float real = *a * cosine - *b * sine;
        *b = *a * sine + *b * cosine;
        *a = real;
    }
    profile_add(tts, started, "rotate", c, 1, 1, n);
}

static void text_condition(struct tts * tts, float * x, int32_t c, int32_t n,
                           const struct condition * condition,
                           const char * at) {
    const struct pack *   w = &tts->weights;
    const struct tensor * theta = weight(w, FIELD ".main_blocks.3.attn"
                                            ".theta");
    const struct tensor * wk = weight(w, "%s.attn.W_key.linear.weight", at);
    const int32_t         m = condition->length;
    const int32_t         heads = c / (2 * (int32_t)theta->count);
    const size_t          mark = tts->used;
    float * q = floats(tts, (int64_t)c * n);
    float * k = floats(tts, (int64_t)c * m);
    float * v = floats(tts, (int64_t)c * m);
    float * o = floats(tts, (int64_t)c * n);
    project(tts, q,x, n, weight(w, "%s.attn.W_query.linear.weight", at),
            weight(w, "%s.attn.W_query.linear.bias", at));
    project(tts, k,condition->text, m, wk,
            weight(w, "%s.attn.W_key.linear.bias", at));
    project(tts, v,condition->text, m,
            weight(w, "%s.attn.W_value.linear.weight", at),
            weight(w, "%s.attn.W_value.linear.bias", at));
    rotate(tts, q, c, n, theta);
    rotate(tts, k, c, m, theta);
    attend(tts, o, q, n, k, v, m, c, heads, sqrtf((float)wk->shape[0]));
    project(tts, q,o, n, weight(w, "%s.attn.out_fc.linear.weight", at),
            weight(w, "%s.attn.out_fc.linear.bias", at));
    axpy(x, 1.0f, q, (int64_t)c * n);
    tts->used = mark;
}

static float * field_block(struct tts * tts, float * x, int32_t c, int32_t n,
                           int32_t block, const float * time,
                           const struct condition * condition) {
    const struct pack * w = &tts->weights;
    const size_t        mark = tts->used;
    char at[name_capacity];
    snprintf(at, sizeof at, FIELD ".main_blocks.%d", block);
    float * y = x;
    if (block % 2 == 0) {
        y = stack(tts, x, c, n, block % 6 == 0 ? field_dilations : NULL,
                  false, NULL, at);
    } else if (block % 6 == 1) {
        float * shift = floats(tts, c);
        project(tts, shift, time, 1,
                weight(w, "%s.linear.linear.weight", at),
                weight(w, "%s.linear.linear.bias", at));
        for (int64_t i = 0; i < (int64_t)c * n; i++) { x[i] += shift[i / n]; }
        tts->used = mark;
    } else {
        if (block % 6 == 3) {
            text_condition(tts, x, c, n, condition, at);
        } else {
            float * styled = floats(tts, (int64_t)c * n);
            strlcat(at, ".attention", sizeof at);
            style_attention(tts, styled, x, n, condition, at);
            axpy(x, 1.0f, styled, (int64_t)c * n);
            tts->used = mark;
        }
        snprintf(at, sizeof at, FIELD ".main_blocks.%d", block);
        layer_norm(tts, x, c, n, weight(w, "%s.norm.norm.weight", at),
                   weight(w, "%s.norm.norm.bias", at));
    }
    return y;
}

static float * velocity(struct tts * tts, const float * latent, int32_t n,
                        const float * time,
                        const struct condition * condition) {
    const struct pack *   w = &tts->weights;
    const struct tensor * in = weight(w, FIELD ".proj_in.net.weight");
    const struct tensor * out = weight(w, FIELD ".proj_out.net.weight");
    const int32_t         c = in->shape[0];
    const int32_t         blocks = 6 * field_groups;
    const char *          tag = condition->tag;
    float * v = floats(tts, (int64_t)out->shape[0] * n);
    const size_t mark = tts->used;
    float * x = floats(tts, (int64_t)c * n);
    conv(tts, x, latent, in->shape[1], n, in, NULL, 1, false);
    tap(tts, x, (int64_t)c * n, "ve.%s.proj_in", tag);
    for (int32_t block = 0; block < blocks; block++) {
        x = field_block(tts, x, c, n, block, time, condition);
        tap(tts, x, (int64_t)c * n, "ve.%s.block.%d", tag, block);
    }
    x = stack(tts, x, c, n, NULL, false, NULL, FIELD ".last_convnext");
    tap(tts, x, (int64_t)c * n, "ve.%s.last", tag);
    conv(tts, v, x, c, n, out, NULL, 1, false);
    tap(tts, v, (int64_t)out->shape[0] * n, "ve.%s.velocity", tag);
    tts->used = mark;
    return v;
}

static void flow(struct tts * tts, float * x, int64_t size, int32_t n,
                 struct condition * spoken, struct condition * silent,
                 int32_t steps) {
    const struct tensor * out = weight(&tts->weights,
                                       FIELD ".time_encoder.mlp.2.linear"
                                             ".weight");
    char tags[2][16];
    spoken->tag = tags[0];
    silent->tag = tags[1];
    for (int32_t step = 0; step < steps; step++) {
        const size_t mark = tts->used;
        float * time = floats(tts, out->shape[0]);
        snprintf(tags[0], sizeof tags[0], "%d.c", step);
        snprintf(tags[1], sizeof tags[1], "%d.u", step);
        time_embedding(tts, time, (float)step / (float)steps);
        tap(tts, time, out->shape[0], "ve.%d.time", step);
        const float * with = velocity(tts, x, n, time, spoken);
        const float * without = velocity(tts, x, n, time, silent);
        for (int64_t i = 0; i < size; i++) {
            x[i] += 1.0f / (float)steps * (with[i] * (1.0f + guidance) -
                                            without[i] * guidance);
        }
        tap(tts, x, size, "ve.step.%d", step);
        tts->used = mark;
    }
}

static const float * vocoder_head(struct tts * tts, float * x, int32_t c,
                                  int32_t n, bool tapped) {
    const struct pack *   w = &tts->weights;
    const struct tensor * head = weight(w, AE ".decoder.head.layer1.net"
                                           ".weight");
    const struct tensor * last = weight(w, AE ".decoder.head.layer2.weight");
    const float * shift = weight(w, AE ".decoder.final_norm.norm"
                                    ".running_mean")->data;
    const float * spread = weight(w, AE ".decoder.final_norm.norm"
                                     ".running_var")->data;
    const float * gain = weight(w, AE ".decoder.final_norm.norm.weight")->data;
    const float * bias = weight(w, AE ".decoder.final_norm.norm.bias")->data;
    const int32_t wide = head->shape[0];
    float * y = floats(tts, (int64_t)last->shape[0] * n);
    const size_t mark = tts->used;
    float * h = floats(tts, (int64_t)wide * n);
    for (int64_t i = 0; i < (int64_t)c * n; i++) {
        const int64_t at = i / n;
        x[i] = (x[i] - shift[at]) / sqrtf(spread[at] + batch_norm_epsilon) *
               gain[at] + bias[at];
    }
    if (tapped) { tap(tts, x, (int64_t)c * n, "voc.norm"); }
    conv(tts, h, x, c, n, head, weight(w, AE ".decoder.head.layer1.net.bias"),
         1, true);
    prelu(h, (int64_t)wide * n,
          weight(w, AE ".decoder.head.act.weight")->data[0]);
    conv(tts, y, h, wide, n, last, NULL, 1, false);
    tts->used = mark;
    return y;
}

static void vocoder_span(struct tts * tts, float * wav, const float * latent,
                         int32_t frames, int32_t from, int32_t to,
                         bool tapped) {
    const struct pack *   w = &tts->weights;
    const struct tensor * first = weight(w, AE ".decoder.embed.net.weight");
    const struct tensor * head = weight(w, AE ".decoder.head.layer1.net"
                                           ".weight");
    const struct tensor * last = weight(w, AE ".decoder.head.layer2.weight");
    const struct tensor * wing = weight(w, AE ".decoder.convnext.0.dwconv.net"
                                           ".weight");
    const float * mean = weight(w, AE ".latent_mean")->data;
    const float * std = weight(w, AE ".latent_std")->data;
    const float   scale = weight(w, "vocoder.tts.ttl.normalizer.scale")
                              ->data[0];
    const int32_t ld = first->shape[1];
    const int32_t c = first->shape[0];
    const int32_t hop = last->shape[0];
    const int32_t factor = weight(w, FIELD ".proj_in.net.weight")->shape[1] /
                           ld;
    int32_t context = first->shape[2] - 1 + head->shape[2] - 1;
    for (size_t i = 0; i < sizeof vocoder_dilations / sizeof(int32_t); i++) {
        context += vocoder_dilations[i] * (wing->shape[2] - 1);
    }
    const int32_t start = from > context ? from - context : 0;
    const int32_t n = to - start;
    const size_t  mark = tts->used;
    float * z = floats(tts, (int64_t)ld * n);
    float * x = floats(tts, (int64_t)c * n);
    for (int64_t i = 0; i < (int64_t)ld * n; i++) {
        const int64_t t = start + i % n;
        z[i] = latent[(i / n * factor + t % factor) * frames + t / factor] /
               scale * std[i / n] + mean[i / n];
    }
    if (tapped) { tap(tts, z, (int64_t)ld * n, "voc.latent"); }
    conv(tts, x, z, ld, n, first, weight(w, AE ".decoder.embed.net.bias"), 1,
         true);
    if (tapped) { tap(tts, x, (int64_t)c * n, "voc.embed"); }
    x = stack(tts, x, c, n, vocoder_dilations, true,
              tapped ? "voc.convnext" : NULL, AE ".decoder");
    const float * y = vocoder_head(tts, x, c, n, tapped);
    for (int64_t i = (int64_t)from * hop; i < (int64_t)to * hop; i++) {
        wav[i] = y[i % hop * n + i / hop - start];
    }
    tts->used = mark;
}

static float * vocoder(struct tts * tts, const float * latent,
                       int32_t frames, int32_t * samples) {
    const struct pack * w = &tts->weights;
    const int32_t hop = weight(w, AE ".decoder.head.layer2.weight")->shape[0];
    const int32_t n = frames *
                      weight(w, FIELD ".proj_in.net.weight")->shape[1] /
                      weight(w, AE ".decoder.embed.net.weight")->shape[1];
    float * wav = floats(tts, (int64_t)n * hop);
    if (tts->golden.tensors != NULL) {
        vocoder_span(tts, wav, latent, frames, 0, n, true);
        tap(tts, wav, (int64_t)n * hop, "wav");
    }
    for (int32_t from = 0; from < n; from += vocoder_slice) {
        vocoder_span(tts, wav, latent, frames, from,
                     n - from < vocoder_slice ? n : from + vocoder_slice,
                     false);
    }
    *samples = n * hop;
    return wav;
}

static struct voice voice_open(struct tts * tts, const char * name) {
    const struct pack *   w = &tts->weights;
    const struct tensor * ttl = weight(w, "voice.%s.ttl", name);
    const struct voice    voice = {
        .dp = weight(w, "voice.%s.dp", name)->data, .tokens = ttl->shape[0],
        .ttl = transposed(tts, ttl->data, ttl->shape[0], ttl->shape[1]) };
    return voice;
}

static struct condition unconditional(struct tts * tts, int32_t n,
                                      int32_t m, int32_t c) {
    const struct pack *   w = &tts->weights;
    const struct tensor * blank = weight(w, MASK ".text_special_token");
    float * nothing = floats(tts, (int64_t)blank->shape[1] * n);
    for (int64_t i = 0; i < (int64_t)blank->shape[1] * n; i++) {
        nothing[i] = blank->data[i / n];
    }
    const struct condition silent = {
        .text = nothing, .length = n, .tokens = m,
        .keys = transposed(tts, weight(
            w, MASK ".style_key_special_token")->data, m, c),
        .values = transposed(tts, weight(
            w, MASK ".style_value_special_token")->data, m, c) };
    return silent;
}

static struct audio synthesize(struct tts * tts, const uint32_t * text,
                               int32_t count, const char * lang,
                               const struct voice * voice,
                               struct mt19937 * mt, int32_t steps,
                               float speed) {
    const struct pack *   w = &tts->weights;
    const struct tensor * keys = weight(w, TE ".style_encoder"
                                           ".style_token_layer.style_key");
    const struct tensor * blank = weight(w, MASK ".text_special_token");
    const int32_t rows = weight(w, FIELD ".proj_in.net.weight")->shape[1];
    const int32_t chunk =
        weight(w, AE ".decoder.head.layer2.weight")->shape[0] * rows /
        weight(w, AE ".decoder.embed.net.weight")->shape[1];
    const int32_t m = voice->tokens;
    int32_t ids[text_capacity];
    const int32_t n = text_ids(text, count, lang, w, ids, text_capacity);
    tap_bytes(tts, "text.ids", ids, (size_t)n * sizeof(int32_t),
              sizeof(int32_t));
    struct condition spoken = {
        .length = n, .values = voice->ttl, .tokens = m,
        .keys = transposed(tts, keys->data, m, keys->shape[2]) };
    struct condition silent = unconditional(tts, n, m, keys->shape[2]);
    profile_module(tts, module_duration);
    float seconds = duration(tts, ids, n, voice->dp);
    tap(tts, &seconds, 1, "dp.duration");
    seconds /= speed;
    tap(tts, &seconds, 1, "duration");
    profile_module(tts, module_text);
    spoken.text = text_encoder(tts, ids, n, &spoken);
    profile_module(tts, module_other);
    tap(tts, spoken.text, (int64_t)blank->shape[1] * n, "te.text_emb");
    const int32_t frames = (int32_t)(
        (seconds * (float)sample_rate + (float)chunk - 1.0f) / (float)chunk);
    float * x = floats(tts, (int64_t)rows * frames);
    for (int64_t i = 0; i < (int64_t)rows * frames; i++) {
        x[i] = (float)mt_gauss(mt);
    }
    tap_bytes(tts, "noise", x, (size_t)rows * (size_t)frames * sizeof(float),
              sizeof(float));
    profile_module(tts, module_field);
    flow(tts, x, (int64_t)rows * frames, frames, &spoken, &silent, steps);
    struct audio audio = { .seconds = seconds };
    profile_module(tts, module_vocoder);
    audio.samples = vocoder(tts, x, frames, &audio.count);
    profile_module(tts, module_other);
    tap(tts, audio.samples, audio.count, "wav");
    return audio;
}

static int16_t pcm16(float sample) {
    const float scaled = sample * 2147483648.0f;
    int16_t value = 0;
    if ((double)scaled >= 2147483647.0) {
        value = INT16_MAX;
    } else if (scaled <= -2147483648.0f) {
        value = INT16_MIN;
    } else {
        value = (int16_t)(lrintf(scaled) >> 16);
    }
    return value;
}

static uint8_t * wave(const float * samples, int32_t count, int32_t rate,
                      size_t * bytes) {
    const uint32_t data = (uint32_t)count * 2;
    const uint32_t header[11] = {
        0x46464952, 36 + data, 0x45564157, 0x20746d66, 16, 0x00010001,
        (uint32_t)rate, (uint32_t)rate * 2, 0x00100002, 0x61746164, data };
    uint8_t * file = malloc(sizeof header + data);
    memcpy(file, header, sizeof header);
    for (int32_t i = 0; i < count; i++) {
        const int16_t value = pcm16(samples[i]);
        memcpy(file + sizeof header + (size_t)i * 2, &value, sizeof value);
    }
    *bytes = sizeof header + data;
    return file;
}

static struct tts tts_open(const char * path) {
    struct tts tts = { .weights = pack_open(path) };
    tts.arena = malloc((size_t)arena_floats * sizeof(float));
    if (tts.arena == NULL) { fatal("no memory for the arena"); }
    return tts;
}

static void wave_gate(struct tts * tts, const struct audio * audio) {
    const struct tensor * file = weight(&tts->golden, "wav.file");
    const struct tensor * wav = weight(&tts->golden, "wav");
    const uint8_t * theirs = (const void *)file->data;
    size_t bytes = 0;
    uint8_t * rule = wave(wav->data, (int32_t)wav->count, sample_rate,
                          &bytes);
    tap_bytes(tts, "wav.file", rule, bytes, 1);
    free(rule);
    uint8_t * mine = wave(audio->samples, audio->count, sample_rate,
                          &bytes);
    int32_t differing = 0;
    int32_t worst = 0;
    for (size_t i = 44; i + 1 < bytes && bytes == (size_t)file->count;
         i += 2) {
        int16_t a = 0;
        int16_t b = 0;
        memcpy(&a, mine + i, sizeof a);
        memcpy(&b, theirs + i, sizeof b);
        differing += a != b;
        worst = abs(a - b) > worst ? abs(a - b) : worst;
    }
    printf("  wav: %d of %d samples differ from the SDK's file, at most "
           "%d LSB\n", differing, audio->count, worst);
    free(mine);
}

static int32_t gate(struct tts * tts, const char * path) {
    tts->golden = pack_open(path);
    const struct pack * g = &tts->golden;
    const char *   text = (const void *)weight(g, "text")->data;
    const char *   name = (const void *)weight(g, "voice")->data;
    const char *   lang = (const void *)weight(g, "lang")->data;
    const int32_t  steps = *(const int32_t *)weight(g, "steps")->data;
    const uint32_t seed = *(const uint32_t *)weight(g, "seed")->data;
    printf("gate: %s voice=%s seed=%u steps=%d text=\"%s\"\n", path, name,
           seed, steps, text);
    tts->used = 0;
    tts->taps = 0;
    tts->failures = 0;
    const struct voice voice = voice_open(tts, name);
    uint32_t * raw = malloc((strlen(text) + 1) * sizeof(uint32_t));
    struct mt19937 mt;
    mt_seed(&mt, seed);
    const struct audio audio = synthesize(tts, raw, text_decode(text, raw),
                                          lang, &voice, &mt, steps,
                                          weight(g, "speed")->data[0]);
    free(raw);
    wave_gate(tts, &audio);
    printf("gate: %s (%d taps, %d failed)\n",
           tts->failures == 0 ? "PASS" : "FAIL", tts->taps, tts->failures);
    return tts->failures;
}

static struct audio narrate(struct tts * tts, const char * text,
                            const char * lang, const struct voice * voice,
                            uint32_t seed, int32_t steps, float speed) {
    const int32_t limit = strcmp(lang, "ko") == 0 ? korean_chunk_limit :
                                                    chunk_limit;
    const int32_t pause = (int32_t)(chunk_silence * sample_rate);
    const size_t  mark  = tts->used;
    struct chunks c     = chunk_open(&tts->weights, text, limit);
    struct audio  whole = { 0 };
    struct mt19937 mt;
    mt_seed(&mt, seed);
    while (chunk_next(&c)) {
        const struct audio part = synthesize(tts, c.cp, c.length, lang, voice,
                                             &mt, steps, speed);
        const int32_t gap = whole.count > 0 ? pause : 0;
        whole.samples = realloc(whole.samples,
                                (size_t)(whole.count + gap + part.count) *
                                sizeof(float));
        if (whole.samples == NULL) { fatal("no memory for the audio"); }
        fill(whole.samples + whole.count, 0.0f, gap);
        memcpy(whole.samples + whole.count + gap, part.samples,
               (size_t)part.count * sizeof(float));
        whole.count   += gap + part.count;
        whole.seconds += part.seconds + (float)gap / (float)sample_rate;
        tts->used = mark;
    }
    if (whole.count == 0) { fatal("the text is empty"); }
    free(c.raw);
    free(c.cp);
    return whole;
}

static void speak(struct tts * tts, const char * text, const char * lang,
                  const char * name, uint32_t seed, int32_t steps,
                  float speed, const char * out) {
    const double started = seconds_now();
    const struct voice voice = voice_open(tts, name);
    const struct audio audio = narrate(tts, text, lang, &voice, seed, steps,
                                       speed);
    const double elapsed = seconds_now() - started;
    size_t bytes = 0;
    uint8_t * file = wave(audio.samples, audio.count, sample_rate, &bytes);
    FILE * sink = fopen(out, "wb");
    if (sink == NULL || fwrite(file, 1, bytes, sink) != bytes) {
        fatal("cannot write %s", out);
    }
    fclose(sink);
    free(file);
    free(audio.samples);
    printf("%s: audio=%.2fs wall=%.2fs rtf=%.1fx rate=%d samples=%d "
           "voice=%s\n", out, (double)audio.seconds, elapsed,
           (double)audio.seconds / elapsed, sample_rate, audio.count, name);
}

static int span_order(const void * left, const void * right) {
    const struct span * a = left;
    const struct span * b = right;
    return (a->seconds < b->seconds) - (a->seconds > b->seconds);
}

static void profile_print(struct tts * tts) {
    struct profile * p = tts->profile;
    double kernels[module_count] = { 0 };
    double whole = 0.0;
    profile_module(tts, module_other);
    for (int32_t m = 0; m < module_count; m++) { whole += p->walls[m]; }
    qsort(p->spans, (size_t)p->count, sizeof(struct span), span_order);
    printf("%-8s %-10s %5s %5s %2s %5s %6s %8s %6s %9s %6s\n", "module",
           "kernel", "out", "in", "k", "n", "calls", "seconds", "share",
           "madds", "GFLOPS");
    for (int32_t i = 0; i < p->count; i++) {
        const struct span * s = &p->spans[i];
        const double madds = (double)s->shape[0] * s->shape[1] * s->shape[2] *
                             s->shape[3] * (double)s->calls;
        printf("%-8s %-10s %5d %5d %2d %5d %6lld %8.3f %5.1f%% %9.2e %6.2f\n",
               module_names[s->module], s->kernel, s->shape[0], s->shape[1],
               s->shape[2], s->shape[3], (long long)s->calls, s->seconds,
               100.0 * s->seconds / whole, madds,
               2e-9 * madds / s->seconds);
        kernels[s->module] += s->seconds;
    }
    for (int32_t m = 0; m < module_count; m++) {
        printf("%-8s wall %8.3f s %5.1f%%, in the kernels above %8.3f s\n",
               module_names[m], p->walls[m], 100.0 * p->walls[m] / whole,
               kernels[m]);
    }
}

static const char * option(int argc, char ** argv, const char * key,
                           const char * fallback) {
    const char * value = fallback;
    for (int i = 2; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) { value = argv[i + 1]; }
    }
    return value;
}

static bool flag(int argc, char ** argv, const char * key) {
    int i = 2;
    while (i < argc && strcmp(argv[i], key) != 0) { i++; }
    return i < argc;
}

int main(int argc, char ** argv) {
    const char * pack = option(argc, argv, "--pack",
                               "models/supertonic-fp32.safetensors");
    const char * command = argc > 1 ? argv[1] : "help";
    int status = 0;
    if (strcmp(command, "speak") == 0) {
        static struct profile profile = { .module = module_other };
        struct tts tts = tts_open(pack);
        if (flag(argc, argv, "--profile")) {
            profile.entered = seconds_now();
            tts.profile = &profile;
        }
        speak(&tts,
              option(argc, argv, "--text", "A gentle breeze moved through "
                     "the open window while everyone listened to the "
                     "story."),
              option(argc, argv, "--lang", "en"),
              option(argc, argv, "--voice", "M1"),
              (uint32_t)strtoul(option(argc, argv, "--seed", "0"), NULL, 10),
              (int32_t)strtol(option(argc, argv, "--steps", "8"), NULL, 10),
              (float)strtod(option(argc, argv, "--speed", "1.05"), NULL),
              option(argc, argv, "--out", "tmp/tts.c.wav"));
        if (tts.profile != NULL) { profile_print(&tts); }
    } else if (strcmp(command, "gate") == 0 && argc > 2) {
        struct tts tts = tts_open(pack);
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--pack") == 0) {
                i++;
            } else {
                status |= gate(&tts, argv[i]) != 0;
            }
        }
    } else {
        printf("tts speak [--text T] [--voice M1] [--lang en] [--seed 0]\n"
               "          [--steps 8] [--speed 1.05] [--out tmp/tts.c.wav]\n"
               "          [--pack models/supertonic-fp32.safetensors]\n"
               "          [--profile]\n"
               "tts gate  [--pack models/supertonic-fp32.safetensors]\n"
               "          tmp/golden/NAME.safetensors ...\n");
        status = strcmp(command, "help") != 0;
    }
    return status;
}
