// ESP32 dispatch node — standalone model-dispatch router (LilyGo T3S3, ESP32-S3 4MB flash / 2MB PSRAM).
//
// Role in the fleet (docs/PORT-PLAN.md): the dispatch model leaves the host
// and lives on its own ESP32. A host sends a text query; this node tokenizes
// with the deterministic FNV-1a word hash (bit-identical to
// train_router.py::fnv1a and md_server.py::fnv1a), runs the quantized
// 2.17M-param transformer classifier, and returns top-k experts with
// renormalized softmax weights.
//
// Memory design (T3S3: 4MB embedded flash, 2MB QSPI PSRAM — NOT opi!):
//   - model (2.17MB int8 blob) stays flash-mmap'd in the 'model' partition.
//     Nothing is copied to SRAM or PSRAM at boot; D=64 weight rows are single
//     cache lines, so mmap reads are cheap.
//   - activations/scratch fit internal SRAM (~40KB): the hot path never
//     touches PSRAM, so QSPI-vs-OPI PSRAM misconfig can't bite this firmware.
//
// Serial contract (921600 baud, newline-delimited):
//   "ROUTE <text>\n"    -> "ROUTE <e>:<w>,<e>:<w> <us>\n"   (k=ROUTE_TOP_K, renormalized)
//   "TOKENS <ids...>\n" -> same, but host supplies token ids (skips tokenizer)
//   "LOGITS <ids...>\n" -> "LOGITS <l0> <l1> <l2> <us>\n" raw classifier logits
//                          (bit-exact HW verification against export_router.py golden)
//   "PING\n"            -> "PONG router RTR1 V=.. D=.. L=.. E=.. S=..\n"
//   "STAT\n"            -> "STAT routes=.. last_us=.. sram_free=..\n"
//   "GOLDEN\n"          -> prints logits for the embedded golden vectors
//                          (export_router.py golden set) for HW verification
// Errors: "ERR <reason>\n". Banner + "READY" on boot.
//
// Flash layout (partitions.csv): factory app @0x10000 (1.2MB), model @0x140000
// (2.75MB), coredump @0x3F0000. Board fqbn MUST include CDCOnBoot=cdc or
// Serial is silent (fleet lesson, learned twice).

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "WiFi.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define ROUTE_TOP_K 2
#define MAX_TEXT 1024
#define MAX_SEQ 64        // firmware-side cap; setup() refuses models with S > MAX_SEQ
#define LINE_BUF 2048
#define RTR1_MAGIC 0x52545231u  // "RTR1"

// ---------------- RTR1 format (must match export_router.py) ------------------
// header: u32 magic, i32 V, D, L, H, F, S, E
// then (fixed order):
//   tok_codes V*D i8 | tok_scales V fp16 | pos_emb S*D fp16
//   per layer: qkv_w8 3D*D i8 + qkv_s 3D fp16 | proj_w8 D*D i8 + proj_s D fp16
//              | up_w8 F*D i8 + up_s F fp16 | dn_w8 D*F i8 + dn_s D fp16
//              | attn_norm D fp16 | ffn_norm D fp16
//   out_norm D fp16 | cls_w8 E*D i8 + cls_s E fp16 | cls_bias E fp32
typedef struct { int32_t V, D, L, H, F, S, E; } RCfg;

static const uint8_t *g_base;
static RCfg g_c;

// IEEE half -> float (verbatim from firmware/common/llm.h — proven code)
static inline float half2float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
  if (exp == 0) {
    if (man == 0) f = sign;
    else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) { man <<= 1; exp--; }
      man &= 0x3FF; f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out; memcpy(&out, &f, 4); return out;
}

// ---------------- FNV-1a word hash (MUST match train_router.py::fnv1a) ------
static inline uint32_t fnv1a(const char *s, size_t len, int32_t V) {
  uint32_t h = 0x811C9DC5u;
  for (size_t i = 0; i < len; i++) { h ^= (uint8_t)s[i]; h *= 0x01000193u; }
  return h % (uint32_t)V;
}

// Tokenize exactly like train_router.py::tokenize: lowercase (ASCII), split on
// whitespace runs, FNV-1a id per word, pad id 0 to seq_len, truncate to S.
static void tokenize_text(const char *text, int *ids, int S, int32_t V) {
  char buf[MAX_TEXT];
  strncpy(buf, text, MAX_TEXT - 1);
  buf[MAX_TEXT - 1] = 0;
  for (char *c = buf; *c; c++) if (*c >= 'A' && *c <= 'Z') *c += 32;
  int n = 0;
  const char *p = buf;
  while (*p && n < S) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == '\f' || *p == '\v') p++;
    if (!*p) break;
    const char *w = p;
    while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r'
           && *p != '\f' && *p != '\v') p++;
    ids[n++] = (int)fnv1a(w, (size_t)(p - w), V);
  }
  for (int i = n; i < S; i++) ids[i] = 0;  // pad id 0, exactly like training
}

// ---------------- int8 math (contract mirrors llm.h + export_router.py) -----
// weights: per-output-row symmetric int8, fp16-rounded scales
// activations: dynamic per-tensor symmetric int8
// linear: y[r] = (dot(act_q, w8[r]) * wscale[r]) * act_s, int32 accumulate
static int8_t act_q[256];   // largest activation row: 3D = 192
static float act_s;

static inline void quantize_act(const float *x, int n) {
  float xmax = 1e-8f;
  for (int j = 0; j < n; j++) { float a = fabsf(x[j]); if (a > xmax) xmax = a; }
  float inv = 127.f / xmax;
  for (int j = 0; j < n; j++) {
    int q = (int)lrintf(x[j] * inv);
    act_q[j] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  }
  act_s = xmax / 127.f;
}

static inline void lin_i8(const int8_t *w8, const uint16_t *ws,
                          int rows, int cols, const float *x, float *y) {
  quantize_act(x, cols);
  for (int r = 0; r < rows; r++) {
    const int8_t *wr = w8 + (size_t)r * cols;
    int32_t acc = 0;
    for (int j = 0; j < cols; j++) acc += (int32_t)act_q[j] * (int32_t)wr[j];
    y[r] = (float)acc * half2float(ws[r]) * act_s;
  }
}

static inline void rmsnorm(float *out, const float *x, const uint16_t *w, int D) {
  float ss = 0.f;
  for (int j = 0; j < D; j++) ss += x[j] * x[j];
  float inv = 1.f / sqrtf(ss / D + 1e-6f);
  // multiply order matches export_router.py rmsnorm: (w * x) * inv
  for (int j = 0; j < D; j++) out[j] = half2float(w[j]) * x[j] * inv;
}

static inline void gelu_erf(float *x, int n) {
  for (int j = 0; j < n; j++)
    x[j] = 0.5f * x[j] * (1.f + erff(x[j] * 0.70710678118654752f));
}

// ---------------- forward pass state (internal SRAM, no PSRAM) --------------
static float *x_buf;     // S*D
static float *qkv_buf;   // 3D
static float *ctx_buf;   // D
static float *scores;    // S
static float *nrm_buf;   // F  (normed input to a linear)
static float *lin_out;   // F  (linear output scratch, F >= 3D >= D)
static float *kcache;    // S*D per layer (recomputed per layer)
static float *vcache;    // S*D
static float *xn_buf;    // S*D (post out_norm, for pooling)
static float *pooled;    // D
static float *logits;    // E

static int64_t g_last_us = 0;
static uint32_t g_routes = 0;

// Causal attention over one layer: full S pass with per-token KV cache.
static bool forward(const int *ids, int *topk_idx, float *topk_p) {
  const int D = g_c.D, L = g_c.L, H = g_c.H, S = g_c.S, F = g_c.F, E = g_c.E;
  const int hd = D / H;
  int64_t t0 = esp_timer_get_time();

  // --- walk tensor pointers (fixed order, no state) ---
  // header = 4-byte magic + 7 x i32 = 32 bytes (verified: /tmp/sim_walk.py
  // reproduces the misaligned-walk NaN signature only at skip=36)
  const uint8_t *p = g_base + 4 + 7 * 4;
  const int8_t *tok_codes = (const int8_t *)p;   p += (size_t)g_c.V * D;
  const uint16_t *tok_scales = (const uint16_t *)p; p += (size_t)g_c.V * 2;
  const uint16_t *pos_emb = (const uint16_t *)p; p += (size_t)S * D * 2;
  const int8_t *b_qkv[L], *b_proj[L], *b_up[L], *b_dn[L];
  const uint16_t *b_qkvs[L], *b_projs[L], *b_ups[L], *b_dns[L], *b_an[L], *b_fn[L];
  for (int l = 0; l < L; l++) {
    b_qkv[l] = (const int8_t *)p;   p += (size_t)3 * D * D;
    b_qkvs[l] = (const uint16_t *)p; p += (size_t)3 * D * 2;
    b_proj[l] = (const int8_t *)p;  p += (size_t)D * D;
    b_projs[l] = (const uint16_t *)p; p += (size_t)D * 2;
    b_up[l] = (const int8_t *)p;    p += (size_t)F * D;
    b_ups[l] = (const uint16_t *)p; p += (size_t)F * 2;
    b_dn[l] = (const int8_t *)p;    p += (size_t)D * F;
    b_dns[l] = (const uint16_t *)p; p += (size_t)D * 2;
    b_an[l] = (const uint16_t *)p;  p += (size_t)D * 2;
    b_fn[l] = (const uint16_t *)p;  p += (size_t)D * 2;
  }
  const uint16_t *out_n = (const uint16_t *)p; p += (size_t)D * 2;
  const int8_t *cls_w8 = (const int8_t *)p; p += (size_t)E * D;
  const uint16_t *cls_s = (const uint16_t *)p; p += (size_t)E * 2;
  const float *cls_b = (const float *)p; p += (size_t)E * 4;

  // --- embeddings ---
  for (int t = 0; t < S; t++) {
    const int8_t *row = tok_codes + (size_t)ids[t] * D;
    float sc = half2float(tok_scales[ids[t]]);
    for (int j = 0; j < D; j++)
      x_buf[(size_t)t * D + j] = (float)row[j] * sc + half2float(pos_emb[(size_t)t * D + j]);
  }

  // --- blocks ---
  for (int l = 0; l < L; l++) {
    for (int t = 0; t < S; t++) {
      float *xt = x_buf + (size_t)t * D;
      // attention: q,k,v for token t
      rmsnorm(nrm_buf, xt, b_an[l], D);
      lin_i8(b_qkv[l], b_qkvs[l], 3 * D, D, nrm_buf, lin_out);
      memcpy(kcache + (size_t)t * D, lin_out + D, (size_t)D * 4);
      memcpy(vcache + (size_t)t * D, lin_out + 2 * D, (size_t)D * 4);
      const float inv_sqrt = 1.f / sqrtf((float)hd);
      for (int h = 0; h < H; h++) {
        const float *q = lin_out + h * hd;
        // max-stabilized softmax (torch.softmax subtracts the row max —
        // without this, scores >~88 overflow expf to inf, then inf/inf = NaN)
        float smax = -INFINITY;
        for (int u = 0; u <= t; u++) {
          const float *ku = kcache + (size_t)u * D + h * hd;
          float s = 0.f;
          for (int j = 0; j < hd; j++) s += q[j] * ku[j];
          scores[u] = s * inv_sqrt;
          if (scores[u] > smax) smax = scores[u];
        }
        float wsum = 0.f;
        for (int u = 0; u <= t; u++) { scores[u] = expf(scores[u] - smax); wsum += scores[u]; }
        float *ctxh = ctx_buf + h * hd;
        for (int j = 0; j < hd; j++) ctxh[j] = 0.f;
        for (int u = 0; u <= t; u++) {
          float w = scores[u] / wsum;
          const float *vu = vcache + (size_t)u * D + h * hd;
          for (int j = 0; j < hd; j++) ctxh[j] += w * vu[j];
        }
      }
      lin_i8(b_proj[l], b_projs[l], D, D, ctx_buf, nrm_buf);  // reuse nrm_buf
      for (int j = 0; j < D; j++) xt[j] += nrm_buf[j];
      // FFN
      rmsnorm(nrm_buf, xt, b_fn[l], D);
      lin_i8(b_up[l], b_ups[l], F, D, nrm_buf, lin_out);
      gelu_erf(lin_out, F);
      lin_i8(b_dn[l], b_dns[l], D, F, lin_out, nrm_buf);
      for (int j = 0; j < D; j++) xt[j] += nrm_buf[j];
      if ((t & 15) == 0) delay(0);   // feed WDT / keep serial alive
    }
  }

  // --- out_norm + mean-pool (ALL positions incl. pads, as trained) + head ---
  for (int t = 0; t < S; t++)
    rmsnorm(xn_buf + (size_t)t * D, x_buf + (size_t)t * D, out_n, D);
  for (int j = 0; j < D; j++) {
    float s = 0.f;
    for (int t = 0; t < S; t++) s += xn_buf[(size_t)t * D + j];
    pooled[j] = s / S;
  }
  lin_i8(cls_w8, cls_s, E, D, pooled, logits);
  for (int e = 0; e < E; e++) logits[e] += cls_b[e];

  // --- top-k with renormalized softmax weights (router.route() semantics) ---
  int idx[8];
  for (int e = 0; e < E && e < 8; e++) idx[e] = e;
  for (int i = 1; i < E && i < 8; i++) {           // insertion sort desc
    int k = idx[i], j = i - 1;
    while (j >= 0 && logits[idx[j]] < logits[k]) { idx[j + 1] = idx[j]; j--; }
    idx[j + 1] = k;
  }
  float mx = logits[idx[0]], sum = 0.f;
  for (int e = 0; e < ROUTE_TOP_K && e < E; e++) sum += expf(logits[idx[e]] - mx);
  for (int e = 0; e < ROUTE_TOP_K && e < E; e++) {
    topk_idx[e] = idx[e];
    topk_p[e] = expf(logits[idx[e]] - mx) / sum;
  }
  g_last_us = esp_timer_get_time() - t0;
  g_routes++;
  return true;
}

// ---------------- serial command handlers ------------------------------------
static void forward_common(const int *ids, bool print_logits) {
  int topk_idx[ROUTE_TOP_K];
  float topk_p[ROUTE_TOP_K];
  if (!forward(ids, topk_idx, topk_p)) { Serial.println("ERR fwd"); return; }
  if (print_logits) {
    char out[128];
    int off = snprintf(out, sizeof(out), "LOGITS");
    for (int e = 0; e < g_c.E; e++)
      off += snprintf(out + off, sizeof(out) - off, " %.6f", (double)logits[e]);
    snprintf(out + off, sizeof(out) - off, " %lld", (long long)g_last_us);
    Serial.println(out);
    return;
  }
  char out[128];
  int off = snprintf(out, sizeof(out), "ROUTE ");
  for (int e = 0; e < ROUTE_TOP_K; e++)
    off += snprintf(out + off, sizeof(out) - off, "%s%d:%.4f",
                    e ? "," : "", topk_idx[e], (double)topk_p[e]);
  snprintf(out + off, sizeof(out) - off, " %lld", (long long)g_last_us);
  Serial.println(out);
}

static void handle_route(const char *text) {
  int ids[MAX_SEQ];
  tokenize_text(text, ids, g_c.S, g_c.V);
  forward_common(ids, false);
}

static void parse_ids_and_run(const char *arg, bool print_logits) {
  int ids[MAX_SEQ], n = 0;
  const char *p = arg;
  while (*p && n < g_c.S) {
    char *end;
    long v = strtol(p, &end, 10);
    if (end == p) break;
    ids[n++] = (int)v;
    p = end;
  }
  for (int i = n; i < g_c.S; i++) ids[i] = 0;
  forward_common(ids, print_logits);
}

static void handle_ping() {
  Serial.printf("PONG router RTR1 V=%d D=%d L=%d H=%d E=%d S=%d\n",
                g_c.V, g_c.D, g_c.L, g_c.H, g_c.E, g_c.S);
}

static void handle_stat() {
  Serial.printf("STAT routes=%u last_us=%lld sram_free=%u\n",
                g_routes, (long long)g_last_us,
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
}

// ---------------- command loop (bare-metal; framework never regains control;
// same pattern as sensor agent + expert firmware — Arduino loop() background
// tasks cost 50-75s per cycle) -----------------------------------------------
static void command_loop(void) {
  static char line_buf[LINE_BUF];
  static size_t line_pos = 0;
  while (1) {
    while (Serial.available()) {
      char c = (char)Serial.read();
      if (c == '\n' || c == '\r') {
        if (line_pos) {
          line_buf[line_pos] = 0;
          if (!strncmp(line_buf, "ROUTE ", 6))        handle_route(line_buf + 6);
          else if (!strncmp(line_buf, "TOKENS ", 7))  parse_ids_and_run(line_buf + 7, false);
          else if (!strncmp(line_buf, "LOGITS ", 7))  parse_ids_and_run(line_buf + 7, true);
          else if (!strcmp(line_buf, "PING"))         handle_ping();
          else if (!strcmp(line_buf, "STAT"))         handle_stat();
          else if (line_buf[0])                       Serial.println("ERR unknown cmd");
          line_pos = 0;
        }
      } else if (line_pos < sizeof(line_buf) - 1) {
        line_buf[line_pos++] = c;
      }
    }
    delay(2);
  }
}

void setup() {
  WiFi.mode(WIFI_OFF);
  Serial.begin(921600);
  unsigned long _t0 = millis();
  while (!Serial && (millis() - _t0 < 3000)) { delay(10); }
  Serial.println("\n=== ESP32 Dispatch Node (T3S3) ===");

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("ERR model partition not found"); return; }
  esp_partition_mmap_handle_t h;
  esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                     ESP_PARTITION_MMAP_DATA,
                                     (const void **)&g_base, &h);
  if (err != ESP_OK) { Serial.printf("ERR mmap: %d\n", err); return; }

  uint32_t magic; memcpy(&magic, g_base, 4);
  if (magic != RTR1_MAGIC) { Serial.printf("ERR bad magic %08lx", (unsigned long)magic); return; }
  memcpy(&g_c, g_base + 4, sizeof(RCfg));
  Serial.printf("model: V=%d D=%d L=%d H=%d F=%d S=%d E=%d (%.2f MB)\n",
                g_c.V, g_c.D, g_c.L, g_c.H, g_c.F, g_c.S, g_c.E, part->size / 1e6);

  const int D = g_c.D, S = g_c.S, F = g_c.F;
  if (S > MAX_SEQ || F > 3 * D) {
    Serial.printf("ERR unsupported config (S=%d max %d, F=%d > 3D=%d)\n",
                  S, MAX_SEQ, F, 3 * D);
    return;
  }
  x_buf   = (float *)malloc((size_t)S * D * 4);       // 16KB
  qkv_buf = (float *)malloc((size_t)3 * D * 4);       // 768B
  ctx_buf = (float *)malloc((size_t)D * 4);
  scores  = (float *)malloc((size_t)S * 4);
  nrm_buf = (float *)malloc((size_t)F * 4);
  lin_out = (float *)malloc((size_t)3 * D * 4);       // 3D >= F: qkv output AND ffn hidden
  kcache  = (float *)malloc((size_t)S * D * 4);       // 16KB
  vcache  = (float *)malloc((size_t)S * D * 4);       // 16KB
  xn_buf  = (float *)malloc((size_t)S * D * 4);       // 16KB
  pooled  = (float *)malloc((size_t)D * 4);
  logits  = (float *)malloc((size_t)g_c.E * 4);
  if (!x_buf || !qkv_buf || !ctx_buf || !scores || !nrm_buf || !lin_out ||
      !kcache || !vcache || !xn_buf || !pooled || !logits) {
    Serial.println("ERR SRAM alloc"); return;
  }

  Serial.printf("SRAM free: %u KB\n",
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024);
  Serial.println("READY");
  command_loop();   // never returns
}

void loop() {}  // unreachable