# TPU v4 investigation tools (Qwen3.5-family hybrid GDN models)

Found while serving Qwen3.8-27B (BF16 and W8A8 INT8, TP4) on one TPU v4 host (4 chips) with vLLM TPU
(tpu-inference main + vLLM `c8d7a7d`).

| Tool | What it does |
|---|---|
| `gdn_v4_probe.py` | Standalone `fused_conv1d_gdn` on one chip with the per-chip TP4 shapes (4 kq / 12 v heads). Knobs: `REJIT=1` (call inside an outer jit, like the model step), `DTILE`/`MTILE` (tile sizes), `STATE_DTYPE`/`CONV_DTYPE`, `READSPLIT=1` (align-mode read/write slots), `MAX_REQS`, `NBLK`. |
| `rpa_v4_probe.py` | Standalone RPA v3 with explicit block sizes (v4 has no default tiling), decode or mixed, optionally nested (`REJIT=1`). |
| `prefix_cache_gate.py` | Cold-vs-warm greedy check against a running server: cold = fresh `cache_salt`, warm = a salt seeded by an earlier request. Cases: exact prefix hit, multi-turn extend, forked prefixes; prompt lengths around block (256) and chunk (2048) boundaries; real source code as the prompt corpus. Flags CORRUPT (first token differs, or a confident prediction becomes uncertain), FAIL (divergence < 8 tokens), NOISE. `QUICK=1` runs 15 cases. Needs `--enable-prompt-tokens-details`. |
| `trace_breakdown.py`, `trace_diff.py` | Device-time breakdown and op-level diff of phased-profiler traces (`trace.json.gz`). |

## Findings
1. **GDN v3 core halt at ≥ 10 decode sequences on v4.** Operands landed in CMEM inside the model jit. Fixed on `fix/gdn-v3-tpu-v4-cmem`.
2. **GDN v3 prefill VMEM OOM with fp32 state on v4.** The 64-row tile does not fit in 16 MiB. Fixed on `fix/gdn-v3-tpu-v4-f32-state-vmem`.
3. **Per-step whole-pool conv-state work.** `astype(float32)` plus a relayout of the entire conv-state pool every layer: +1.4 ms/step going from a 65-slot to a 257-slot pool. `--mamba-cache-dtype float32` removes the cast; the relayout remains.
4. **Align-mode prefix caching is not output-preserving on this stack.** Warm (cache hit) greedy outputs diverge from cold, including CORRUPT cases at simple exact-prefix hits (seed prompt of 257 / 513 / 2049 tokens, hit at 256 / 512 / 2048, recompute 1 token: the first token changes).
   - It is history-dependent: the first gate pass on a fresh server is mostly clean, and a second pass on the same server corrupts.
   - Not fixed by PR #3634 (which does remove hits on never-checkpointed blocks), fp32 conv or recurrent state, `--no-async-scheduling`, or a 2.5× larger mamba pool.
   - The per-layer metadata lookup is correct: the GDN layers always find their own group's block table.
