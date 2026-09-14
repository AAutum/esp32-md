"""Export the trained dispatch model to an ESP32 flash binary + golden reference.

Why a separate format from the expert PLE models (PLE1/llm.h): the router is a
CLASSIFIER, not a generator — different architecture (embed → 2 transformer
blocks → mean-pool → linear head, no PLE tables, no KV-cache output head over
32k vocab). Reusing PLE1 would force dead fields; RTR1 is minimal and exact.

Binary layout (little-endian, "RTR1" magic, see firmware/md_router/):
  header:
    u32 magic 'RTR1' (0x52545231)
    i32 V, D, L, H, F, S, E            (vocab, d_model, n_layers, n_heads,
                                        ffn dim, seq_len, n_experts)
  tensors, fixed order (offsets derivable from header — C hard-codes order):
    tok_codes    V*D  int8    per-token-embedding int8 codes
    tok_scales   V    fp16    per-row dequant scales
    pos_emb      S*D  fp16    positional embeddings (kept fp16, tiny)
    per block (L times, in order):
      qkv_w8       D*3D int8    + qkv_scales 3D fp16
      proj_w8      D*D  int8    + proj_scales D fp16
      up_w8        D*F  int8    + up_scales F fp16
      dn_w8        F*D  int8    + dn_scales D fp16
      attn_norm_w  D    fp16    RMSNorm weight
      ffn_norm_w   D    fp16    RMSNorm weight
    out_norm_w   D    fp16
    cls_w8       D*E  int8    + cls_scales E fp16
    cls_bias     E    fp32

Quantization contract (MUST match firmware):
  - weights: per-output-row symmetric int8, scale = max|row|/127, fp16-ROUNDED
    BEFORE dividing the row (so golden == device bit-for-bit), codes ±127.
  - activations: dynamic per-tensor symmetric int8 (quantize_act semantics):
    xmax = max(|x|,1e-8); inv = 127/xmax; q = rint(x*inv) clamp ±127;
    xscale = xmax/127. Linear output y[r] = (dot(q, w8[r]) * wscale[r]) * xscale
    (left-assoc float mul, int32 accumulate).
  - tok_emb dequant: emb[v][j] = code * scale[v] (fp32) then + pos_emb fp16.
  - attention, residuals, RMSNorm, softmax, mean-pool: fp32 on device.
  - GELU: exact erf form 0.5*x*(1+erf(x/sqrt(2))) (matches nn.GELU default).

Golden reference: computed here with the SAME int8 math (float64 dot for exact
integer accumulation) so C-vs-golden isolates port bugs from quant error.
Also reports fp32-vs-int8 val accuracy — the deploy gate.
"""

import hashlib
import json
import math
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from router import RouterModel, RouterConfig  # noqa: E402
from train_router import tokenize  # deterministic FNV-1a word hash  # noqa: E402

WS = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(WS, "checkpoints")
DATA = os.path.join(WS, "data")
OUT = os.path.join(RUNS, "router-export")
MAGIC = 0x52545231  # "RTR1"


def q_row_i8(w: torch.Tensor):
    """Per-output-row symmetric int8, fp16-rounded scale (see module docstring)."""
    rows, cols = w.shape
    amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    sc = (amax / 127.0).half().float()          # fp16-round scale FIRST
    q = torch.clamp(torch.round(w / sc), -127, 127).to(torch.int8)
    return q.reshape(-1), sc.reshape(-1)


def quantize_act_t(x: torch.Tensor):
    """Dynamic per-tensor int8 activation quant, mirroring llm.h quantize_act."""
    xmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    inv = 127.0 / xmax
    q = torch.round(x * inv).clamp(-127, 127)
    xs = xmax / 127.0
    return q, xs


def lin_i8(x: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """y[r] = (dot(q_act, w8[r]) * wscale[r]) * act_scale — exact int dots in f64."""
    q, xs = quantize_act_t(x)
    dots = (q.to(torch.float64) @ wq.to(torch.float64).T).to(torch.float32)
    return (dots * ws) * xs


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def forward_int8(model: RouterModel, M: dict, ids: torch.Tensor) -> torch.Tensor:
    """int8-exact forward, mirroring the C device code 1:1 (see module docstring).

    Padding semantics MUST match training: pad token id 0 flows through the
    full seq_len pipeline (pos_emb, causal attention, mean-pool over ALL
    positions) — exactly what RouterBatcher/md_server do host-side.
    """
    cfg = model.cfg
    V, D, L, H, S, E = cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.seq_len, cfg.n_experts
    hd = D // H
    B = ids.shape[0]

    # embeddings: int8 tok + fp16 pos
    codes = M["tok_codes"].view(V, D).to(torch.float32)          # (V, D)
    scales = M["tok_scales"].to(torch.float32)                   # (V,)
    emb = codes * scales[:, None]                                # dequant
    pos = M["pos_emb"].view(S, D).to(torch.float32)
    x = emb[ids] + pos[None, :, :]                               # (B, S, D)

    for l in range(L):
        blk = M["blocks"][l]
        # --- causal self-attention ---
        y = rmsnorm(x, blk["attn_norm"])
        qkv = lin_i8(y, blk["qkv_w8"].view(3 * D, D), blk["qkv_scales"])  # (B,S,3D)
        q, k, v = qkv.split(D, dim=-1)
        ctx = torch.empty_like(q)
        for h in range(H):
            qh = q[:, :, h * hd:(h + 1) * hd]
            kh = k[:, :, h * hd:(h + 1) * hd]
            vh = v[:, :, h * hd:(h + 1) * hd]
            scores = qh @ kh.transpose(1, 2) / math.sqrt(hd)     # fp32
            mask = torch.full((S, S), float("-inf")).triu(1)     # causal
            scores = scores + mask
            attn = torch.softmax(scores, dim=-1)
            ctx[:, :, h * hd:(h + 1) * hd] = attn @ vh
        x = x + lin_i8(ctx, blk["proj_w8"].view(D, D), blk["proj_scales"])
        # --- FFN ---
        y2 = rmsnorm(x, blk["ffn_norm"])
        h1 = gelu(lin_i8(y2, blk["up_w8"].view(cfg.d_model * 2, D), blk["up_scales"]))
        x = x + lin_i8(h1, blk["dn_w8"].view(D, cfg.d_model * 2), blk["dn_scales"])

    xn = rmsnorm(x, M["out_norm"])
    pooled = xn.mean(dim=1)                                      # mean over ALL S
    logits = lin_i8(pooled, M["cls_w8"].view(E, D), M["cls_scales"]) + M["cls_bias"]
    return logits


@torch.no_grad()
def forward_fp32(model: RouterModel, ids: torch.Tensor) -> torch.Tensor:
    logits, _ = model(ids)
    return logits


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "router-3exp-s0"
    ckpt_path = os.path.join(RUNS, f"{tag}.pt")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = RouterConfig(**ck["cfg"])
    model = RouterModel(cfg)
    model.load_state_dict(ck["state"])
    model.eval()
    V, D, L, H, S, E = (cfg.vocab_size, cfg.d_model, cfg.n_layers,
                        cfg.n_heads, cfg.seq_len, cfg.n_experts)
    F = cfg.d_model * 2  # ffn hidden = 2*D per router.py RouterBlock

    sd = model.state_dict()

    # ---- quantize ----
    M = {"tok_codes": None, "tok_scales": None,
         "pos_emb": sd["pos_emb.weight"].half(),
         "blocks": [], "out_norm": sd["out_norm.weight"].half()}
    tok_q, tok_s = q_row_i8(sd["tok_emb.weight"])
    M["tok_codes"], M["tok_scales"] = tok_q.to(torch.int8), tok_s.half()
    for l in range(L):
        p = f"blocks.{l}."
        M["blocks"].append({
            "attn_norm": sd[p + "attn_norm.weight"].half(),
            "qkv_w8": None, "qkv_scales": None,
            "proj_w8": None, "proj_scales": None,
            "ffn_norm": sd[p + "ffn_norm.weight"].half(),
            "up_w8": None, "up_scales": None,
            "dn_w8": None, "dn_scales": None,
        })
        for name, wkey in (("qkv", p + "attn.qkv.weight"),
                           ("proj", p + "attn.proj.weight"),
                           ("up", p + "ffn.0.weight"),
                           ("dn", p + "ffn.2.weight")):
            q, s = q_row_i8(sd[wkey])
            M["blocks"][l][name + "_w8"] = q.to(torch.int8)
            M["blocks"][l][name + "_scales"] = s.half()
    cls_q, cls_s = q_row_i8(sd["classifier.weight"])
    M["cls_w8"] = cls_q.to(torch.int8)
    M["cls_scales"] = cls_s.half()
    M["cls_bias"] = sd["classifier.bias"].to(torch.float32)

    # ---- serialize (order fixed, see docstring) ----
    blobs = []
    hdr = struct.pack("<I7i", MAGIC, V, D, L, H, F, S, E)
    blobs.append(hdr)
    blobs.append(M["tok_codes"].numpy().astype(np.int8).tobytes())
    blobs.append(M["tok_scales"].numpy().astype(np.float16).tobytes())
    blobs.append(M["pos_emb"].numpy().astype(np.float16).tobytes())
    for blk in M["blocks"]:
        for name in ("qkv", "proj", "up", "dn"):
            blobs.append(blk[name + "_w8"].numpy().astype(np.int8).tobytes())
            blobs.append(blk[name + "_scales"].numpy().astype(np.float16).tobytes())
        blobs.append(blk["attn_norm"].numpy().astype(np.float16).tobytes())
        blobs.append(blk["ffn_norm"].numpy().astype(np.float16).tobytes())
    blobs.append(M["out_norm"].numpy().astype(np.float16).tobytes())
    blobs.append(M["cls_w8"].numpy().astype(np.int8).tobytes())
    blobs.append(M["cls_scales"].numpy().astype(np.float16).tobytes())
    blobs.append(M["cls_bias"].numpy().astype(np.float32).tobytes())
    blob = b"".join(blobs)
    os.makedirs(OUT, exist_ok=True)
    bin_path = os.path.join(OUT, "router_model.bin")
    with open(bin_path, "wb") as f:
        f.write(blob)

    # ---- golden: router val set (deterministic FNV-1a tokenization) + fuzz ----
    val = []
    with open(os.path.join(DATA, "router_val.jsonl")) as f:
        for line in f:
            ex = json.loads(line)
            val.append((tokenize(ex["text"], vocab_size=V, seq_len=S), ex["best_expert"]))
    rng = np.random.default_rng(7)
    fuzz = [(rng.integers(0, V, S).tolist(), -1) for _ in range(64)]
    all_ex = val + fuzz
    ids = torch.tensor([e[0] for e in all_ex], dtype=torch.long)
    labels = torch.tensor([e[1] for e in all_ex], dtype=torch.long)

    logits_i8 = forward_int8(model, M, ids)
    logits_f32 = forward_fp32(model, ids)
    probs_i8 = torch.softmax(logits_i8.float(), dim=-1)

    real = labels >= 0
    acc_i8 = (logits_i8.argmax(-1)[real] == labels[real]).float().mean().item()
    acc_f32 = (logits_f32.argmax(-1)[real] == labels[real]).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(
        logits_i8.float(), logits_f32.float(), dim=-1)[real]
    margin = (logits_i8.float().topk(2, dim=-1).values[:, 0]
              - logits_i8.float().topk(2, dim=-1).values[:, 1])

    np.savez(os.path.join(OUT, "router_golden.npz"),
             ids=ids.numpy(), labels=labels.numpy(),
             logits_int8=logits_i8.numpy(), logits_fp32=logits_f32.numpy(),
             probs_int8=probs_i8.numpy())

    md5 = hashlib.md5(blob).hexdigest()
    report = {
        "tag": tag, "params": sum(p.numel() for p in model.parameters()),
        "config": cfg.__dict__,
        "bin_bytes": len(blob), "bin_md5": md5, "bin_path": bin_path,
        "val_acc_int8": acc_i8, "val_acc_fp32": acc_f32,
        "logits_cos_int8_vs_fp32": {"mean": cos.mean().item(), "min": cos.min().item()},
        "top2_margin_int8": {"mean": margin.mean().item(), "min": margin.min().item()},
        "note": "margin is int8 logit top1-top2 gap; HW flip risk concentrates on min-margin rows",
    }
    with open(os.path.join(OUT, "export_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nWrote {bin_path} ({len(blob):,} bytes)")
    print(f"Gate: int8 val acc {acc_i8:.1%} vs fp32 {acc_f32:.1%} "
          f"({'PASS' if acc_i8 >= acc_f32 - 0.03 and acc_i8 >= 0.90 else 'REVIEW'} — "
          f"drop ≤3pt vs fp32 and ≥90% absolute)")


if __name__ == "__main__":
    main()