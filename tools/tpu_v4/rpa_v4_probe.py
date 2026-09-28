"""Standalone RPA v3 probe on ONE TPU chip with the server's per-chip shapes (Qwen3.8-27B, TP4):
6 q heads, 1 kv head, head_dim 256, page 256, bf16 KV, max_num_seqs 16, pages_per_seq 1024 (262144 ctx).

python rpa_v4_probe.py NSEQ MODE [BOUNDS]   MODE: decode | mixed ; BOUNDS=1 enables kernel bounds checks
"""
import os, sys, time
import jax, jax.numpy as jnp, numpy as np
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (get_kv_cache_shape, ragged_paged_attention,
                                                                     ref_ragged_paged_attention)

nseq, mode = int(sys.argv[1]), sys.argv[2]
bounds = len(sys.argv) > 3 and sys.argv[3] == "1"
MAX_SEQ, PPS, PAGE, NPAGES, HQ, HKV, D = int(os.environ.get('MAX_SEQ', 16)), 1024, 256, 4200, 6, 1, 256
env = lambda k, d: tuple(int(x) for x in os.environ.get(k, d).split(","))
d_bs, p_bs, m_bs = env("RPA_D", "1,2048,1,2048"), env("RPA_P", "128,1024,64,512"), env("RPA_M", "128,1024,64,512")
rng = np.random.default_rng(0)
if mode == "decode":
    q_lens = [1] * nseq
else:  # one prefill chunk + (nseq-1) decodes, like the crashing staggered test
    q_lens = [1] * (nseq - 1) + [300]
kv_lens = [int(rng.integers(40, 900)) + q for q in q_lens]
T = 2048 if mode != "decode" else max(16, -(-nseq // 16) * 16)
q = jnp.asarray(rng.standard_normal((T, HQ, D), np.float32), jnp.bfloat16)
k = jnp.asarray(rng.standard_normal((T, HKV, D), np.float32), jnp.bfloat16)
v = jnp.asarray(rng.standard_normal((T, HKV, D), np.float32), jnp.bfloat16)
kv_cache = jnp.asarray(rng.standard_normal(get_kv_cache_shape(NPAGES, PAGE, HKV, D, jnp.bfloat16), np.float32),
                       jnp.bfloat16)
page_indices = np.zeros((MAX_SEQ, PPS), np.int32)
nxt = 1
for s, kl in enumerate(kv_lens):
    n = -(-kl // PAGE)
    page_indices[s, :n] = np.arange(nxt, nxt + n)
    nxt += n
cu = np.zeros(MAX_SEQ + 1, np.int32)
cu[1:nseq + 1] = np.cumsum(q_lens)
cu[nseq + 1:] = cu[nseq]
kv_l = np.zeros(MAX_SEQ, np.int32)
kv_l[:nseq] = kv_lens
n_dec = sum(1 for x in q_lens if x == 1)
dist = np.array([n_dec, n_dec, nseq], np.int32) if mode == "decode" else np.array([n_dec, n_dec, nseq], np.int32)
args = (q, k, v, kv_cache, jnp.asarray(kv_l), jnp.asarray(page_indices.reshape(-1)), jnp.asarray(cu), jnp.asarray(dist))
print(f"nseq={nseq} mode={mode} bounds_checks={bounds} q_lens={q_lens[:3]}..{q_lens[-1]} dist={dist.tolist()} "
      f"d={d_bs} p={p_bs} m={m_bs} device={jax.devices()[0].device_kind}", flush=True)
ref, _ = ref_ragged_paged_attention(*args, sm_scale=D ** -0.5)
ref = jax.block_until_ready(ref)
args = tuple(jnp.copy(a) for a in args)  # the kernel donates q/k/v/kv_cache
t = time.time()
call = lambda *a: ragged_paged_attention(*a, sm_scale=D ** -0.5, d_block_sizes=d_bs, p_block_sizes=p_bs,
                                        m_block_sizes=m_bs, disable_bounds_checks=not bounds)
if os.environ.get("REJIT") == "1":  # nested inside another jit, like the model step
    call = jax.jit(call)
out, _ = call(*args)
out = jax.block_until_ready(out)
n_tok = int(cu[nseq])
err = float(jnp.max(jnp.abs(out[:n_tok].astype(jnp.float32) - ref[:n_tok].astype(jnp.float32))))
print(f"OK nseq={nseq} mode={mode}: max|out-ref|={err:.4f} ({time.time() - t:.1f}s)", flush=True)
