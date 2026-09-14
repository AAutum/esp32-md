// ESP32 dispatch expert — serial GEN-protocol inference node.
//
// Contract (matches src/md_server.py::ExpertDevice):
//   Recv: "GEN <token_ids space-separated> <max_tokens>\n"
//   Send: "<generated_token_ids space-separated>\n"
// Optional host-side liveness: "PING\n" -> "PONG <domain>\n"
//
// Same proven runtime as the sensor agent + esp32_llm: PLE TinyLM mmap'd from
// flash 'model' partition (llm.h, host-verified), int8-staged tied head on
// dual cores, PSRAM scratch. WiFi kept OFF — experts are compute nodes, not
// telemetry (the sensor agent owns the fleet_sink path).
//
// Domain identity comes from +platformio... no — plain Arduino env:
// pass -DEXPERT_DOMAIN='"code"' via build_opt or edit EXPERT_DOMAIN below.
// It is echoed in PONG so the orchestrator can verify what's plugged where.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "WiFi.h"
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "../common/llm.h"

#ifndef EXPERT_DOMAIN
#define EXPERT_DOMAIN "expert"
#endif

Model model;
Scratch s;

// ---- int8 output head (identical to esp32_llm/sensor agent) ----------------
static int8_t *head_w8 = NULL;
static float  *head_scale8 = NULL;
static int head_rows, head_cols;
static int8_t head_actq[1024];      // head input dim = D (up to 512 safe)
static float  head_acts;

static inline int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

static void head_rows_range(float *y, int r0, int r1) {
  for (int r = r0; r < r1; r++)
    y[r] = (float)dot_i8(head_actq, head_w8 + (size_t)r * head_cols, head_cols)
           * head_scale8[r] * head_acts;
}

static TaskHandle_t head_worker;
static TaskHandle_t inference_task;
static float *volatile head_job_y;
static volatile int head_job_split;

static void head_worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    head_rows_range(head_job_y, 0, head_job_split);
    xTaskNotifyGive(inference_task);
  }
}

static void head_matvec_int8(const QT *t, const float *x, float *y) {
  (void)t;
  quantize_act(x, head_cols, head_actq, &head_acts);
  head_job_y = y;
  head_job_split = head_rows / 2;
  xTaskNotifyGive(head_worker);
  head_rows_range(y, head_job_split, head_rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

static void *ps(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (!p) { Serial.printf("PSRAM alloc failed (%u bytes)\n", (unsigned)n); while (1) delay(1000); }
  return p;
}

static void stage_head_int8(QT *t) {
  head_rows = t->rows; head_cols = t->cols;
  head_w8 = (int8_t *)ps((size_t)head_rows * head_cols);
  head_scale8 = (float *)ps((size_t)head_rows * sizeof(float));
  for (int r = 0; r < head_rows; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    int8_t *dst = head_w8 + (size_t)r * head_cols;
    for (int j = 0; j < head_cols; j++) {
      uint8_t byte = row[j >> 1];
      int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
      dst[j] = (int8_t)(code - 8);
    }
    head_scale8[r] = half2float(t->scales[(size_t)r * t->n_groups]);  // D<=512: single group
  }
  Serial.printf("head staged int8: %.2f MB\n",
                ((size_t)head_rows * head_cols + (size_t)head_rows * 4) / 1e6);
}

// Bare-metal command loop — called from setup()'s while(1) so the Arduino
// framework never regains control (same pattern as the sensor agent; the
// framework's loop() background tasks would eat 50-75s per cycle).
static void command_loop(void) {
  static char   line_buf[2048];        // GEN 4096-vocab prompt: ids up to 5 chars + spaces
  static size_t line_pos = 0;
  while (1) {
    while (Serial.available()) {
      char c = (char)Serial.read();
      if (c == '\n' || c == '\r') {
        if (line_pos) {
          line_buf[line_pos] = 0;
          if (!strncmp(line_buf, "GEN ", 4))         handle_gen(line_buf);
          else if (!strcmp(line_buf, "PING"))        handle_ping();
          else if (!strcmp(line_buf, "STAT"))        handle_stat();
          else if (strcmp(line_buf, ""))             Serial.println("ERR unknown cmd");
          line_pos = 0;
        }
      } else if (line_pos < sizeof(line_buf) - 1) {
        line_buf[line_pos++] = c;
      }
    }
    delay(2);   // tiny yield; commands are bursty
  }
}

// ---- generation ------------------------------------------------------------
// Greedy. KV cache persists only within one GEN call; each GEN is a fresh
// sequence (the orchestrator sends full prompts, chat-style). Temperature
// reserved for later — greedy is deterministic and matches router training.
static void handle_gen(const char *line) {
  // parse: GEN <ids...> <max>
  const char *p = line + 3;          // skip "GEN"
  int prompt[256]; int n_prompt = 0;
  long max_tokens = 32;
  while (*p == ' ') p++;
  while (*p && n_prompt < 256) {
    char *end;
    long v = strtol(p, &end, 10);
    if (end == p) break;
    p = end;
    if (*p == '\0') { max_tokens = v; break; }   // last number = max_tokens
    prompt[n_prompt++] = (int)v;
    while (*p == ' ') p++;
  }
  if (n_prompt == 0) { Serial.println("ERR no prompt"); return; }
  if (max_tokens < 1) max_tokens = 1;
  if (max_tokens > 256) max_tokens = 256;
  if (n_prompt + max_tokens > model.c.seq_len) {
    max_tokens = model.c.seq_len - n_prompt;
    if (max_tokens < 1) { Serial.println("ERR seq_len"); return; }
  }

  int pos = 0;
  for (int i = 0; i < n_prompt; i++)
    llm_forward(&model, prompt[i], pos++, &s);

  int64_t t0 = esp_timer_get_time();
  // first token: argmax from the prompt-primed logits, print, then continue
  String out;
  out.reserve(max_tokens * 5);
  for (int step = 0; step < max_tokens && pos < model.c.seq_len; step++) {
    int best = 0; float bv = -1e30f;
    for (int v = 0; v < head_rows; v++)
      if (s.logits[v] > bv) { bv = s.logits[v]; best = v; }
    out += best; out += ' ';
    llm_forward(&model, best, pos++, &s);
    if ((step & 7) == 0) delay(0);   // feed WDT
  }
  int64_t dt = esp_timer_get_time() - t0;
  Serial.println(out);
  Serial.printf("// %ld tok, %.1f ms/tok, %.1f tok/s\n",
                (long)max_tokens, (float)dt / 1000.0f / max_tokens,
                max_tokens * 1e6f / dt);
}

static void handle_ping() {
  Serial.printf("PONG %s V=%d D=%d L=%d heap=%u\n", EXPERT_DOMAIN,
                model.c.vocab, model.c.dim, model.c.n_layers,
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

static void handle_stat() {
  Serial.printf("STAT domain=%s tok/s=NA profile_calls=%u\n", EXPERT_DOMAIN,
                s.profile.calls);
  float n = (float)s.profile.calls * 1000.f;
  if (s.profile.calls)
    Serial.printf("STAT profile ms/tok: in %.1f attn %.1f ffn %.1f ple %.1f head %.1f\n",
                  s.profile.input_us / n, s.profile.attn_us / n,
                  s.profile.ffn_us / n, s.profile.ple_us / n,
                  s.profile.head_us / n);
}

void setup() {
  WiFi.mode(WIFI_OFF);   // experts: no radio, no interference
  Serial.begin(921600);  // matches md_server.py baud
  unsigned long _t0 = millis();
  while (!Serial && (millis() - _t0 < 3000)) { delay(10); }
  Serial.println("\n=== ESP32-S3 Dispatch Expert ===");

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("ERR model partition not found"); return; }
  const void *base;
  esp_partition_mmap_handle_t h;
  esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                     ESP_PARTITION_MMAP_DATA, &base, &h);
  if (err != ESP_OK) { Serial.printf("ERR mmap: %d\n", err); return; }

  if (llm_load((const uint8_t *)base, &model)) { Serial.println("ERR bad model magic"); return; }
  Cfg *c = &model.c;
  Serial.printf("model: V=%d D=%d L=%d H=%d F=%d P=%d S=%d (%.1f MB)\n",
                c->vocab, c->dim, c->n_layers, c->n_heads, c->ffn, c->ple_dim,
                c->seq_len, part->size / 1e6);
  Serial.printf("domain: " EXPERT_DOMAIN "\n");

  // Stage the FULL vocab head (no VOCAB_N cap here — orchestrator speaks ids,
  // not text; every trained row must be scoreable).
  stage_head_int8(&model.tok_emb);
  model.head_matvec = head_matvec_int8;
  inference_task = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(head_worker_main, "head", 4096, NULL, 2,
                              &head_worker, 0) != pdPASS) {
    Serial.println("ERR head worker");
    return;
  }

  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn, V = c->vocab, S = c->seq_len;
  s.x = (float *)ps(D * 4);
  s.h = (float *)ps((F > D ? F : D) * 4);
  s.qkv = (float *)ps(3 * D * 4);
  s.att = (float *)ps(D * 4);
  s.g1 = (float *)ps(F * 4);
  s.g2 = (float *)ps((P > F ? P : F) * 4);
  s.ple = (float *)ps(L * P * 4);
  s.tmpP = (float *)ps(L * P * 4);
  s.trow = (float *)ps(L * P * 4);
  s.logits = (float *)ps(V * 4);
  s.scores = (float *)ps(S * 4);
  s.kcache = (float *)ps((size_t)L * S * D * 4);
  s.vcache = (float *)ps((size_t)L * S * D * 4);
  Serial.printf("PSRAM free: %u KB\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);
  Serial.println("READY");   // orchestrator sync point

  command_loop();            // never returns; framework never takes control
}

void loop() { }   // unreachable — command_loop owns the chip