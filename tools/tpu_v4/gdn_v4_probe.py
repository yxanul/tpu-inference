"""Standalone GDN v3 (fused conv1d + gated delta rule) probe on ONE TPU chip with the server's per-chip shapes
(Qwen3.8-27B, TP4): 4 k/q heads, 12 v heads, head dims 128, conv kernel 4, bf16 activations and states,
max_num_seqs 16 padding, mamba pool of 129 slots (align mode) - decode batch of NSEQ real sequences.

python gdn_v4_probe.py NSEQ [MODE]   MODE: decode (default) | mixed (NSEQ-1 decodes + one 300-token prefill)
"""
import os, sys, time
import jax, jax.numpy as jnp, numpy as np
from tpu_inference.kernels.gdn.v3 import wrapper

nseq = int(sys.argv[1]); mode = sys.argv[2] if len(sys.argv) > 2 else "decode"
MAX_REQS, NKQ, NV, D, K = int(os.environ.get('MAX_REQS', 16)), 4, 12, 128, 4
NBLK = int(os.environ.get('NBLK', 129))
dim = 2 * NKQ * D + NV * D
lens = [1] * nseq if mode == "decode" else [1] * (nseq - 1) + [int(os.environ.get("PLEN", 300))]
T = max(16, -(-nseq // 16) * 16) if mode == "decode" else 2048
q_loc = np.zeros(MAX_REQS + 1, np.int32); q_loc[1:nseq + 1] = np.cumsum(lens); q_loc[nseq + 1:] = q_loc[nseq]
n_dec = sum(1 for x in lens if x == 1)
dist = np.array([n_dec, nseq if mode == "decode" else n_dec, nseq], np.int32)
rng = np.random.default_rng(0)
state_idx = np.zeros(MAX_REQS, np.int32)
state_idx[:nseq] = (np.arange(1, nseq + 1) if os.environ.get('SLOTS') == 'seq' else rng.choice(np.arange(1, NBLK), nseq, replace=False))
read_idx = state_idx.copy()
if os.environ.get('READSPLIT') == '1':  # align-mode prefix caching: read one slot, write another
    free = np.setdiff1d(np.arange(1, NBLK), state_idx[:nseq])
    read_idx[:nseq] = rng.choice(free, nseq, replace=False)
seq_lens = np.zeros(MAX_REQS, np.int32); seq_lens[:nseq] = [100 + l for l in lens]  # has prior context
f = lambda *s: jnp.asarray(rng.standard_normal(s, np.float32), jnp.bfloat16)
kw = dict(qkv=f(T, dim), b=f(T, NV), a=f(T, NV), conv_state=(f(NBLK, K - 1, dim) * 0.1).astype(os.environ.get('CONV_DTYPE', 'bfloat16')),
          recurrent_state=(f(NBLK, NV, D, D) * 0.01).astype(os.environ.get('STATE_DTYPE', 'bfloat16')), conv_weight=f(dim, 1, K), conv_bias=None,
          a_log=jnp.asarray(rng.standard_normal(NV), jnp.float32), dt_bias=jnp.asarray(rng.standard_normal(NV), jnp.float32),
          query_start_loc=jnp.asarray(q_loc), state_indices=jnp.asarray(state_idx), distribution=jnp.asarray(dist),
          seq_lens=jnp.asarray(seq_lens), read_state_indices=jnp.asarray(read_idx),
          n_kq=NKQ, n_v=NV, d_k=D, d_v=D, kernel_size=K)
print(f"nseq={nseq} mode={mode} dist={dist.tolist()} slots={state_idx[:nseq].tolist()} device={jax.devices()[0].device_kind}",
      flush=True)
fn = (jax.jit(wrapper.fused_conv1d_gdn, static_argnames=['n_kq', 'n_v', 'd_k', 'd_v', 'kernel_size', 'decode_tile_size', 'mixed_tile_size'])
      if os.environ.get('REJIT') == '1' else wrapper.fused_conv1d_gdn)  # REJIT=1: nested jit (no donation), like a model step
t = time.time()
extra = {'decode_tile_size': int(os.environ['DTILE'])} if os.environ.get('DTILE') else {}
if os.environ.get('MTILE'): extra['mixed_tile_size'] = int(os.environ['MTILE'])
(new_conv, new_rec), out = fn(**kw, **extra)
out = jax.block_until_ready(out)
print(f"OK nseq={nseq} mode={mode}: out finite={bool(jnp.all(jnp.isfinite(out[:int(q_loc[nseq])])))} "
      f"state finite={bool(jnp.all(jnp.isfinite(new_rec)))} ({time.time() - t:.1f}s)", flush=True)

if "--ref" in sys.argv:
    sys.path.insert(0, __import__("os").path.expanduser("~/vllm-tpu/src/tpu-inference/tests/kernels"))
    from gdn_attention_v3_test import gdn_attention_ref
    ref_kw = {k: v for k, v in kw.items()}
    (rc, rr), rout = gdn_attention_ref(**{k: v for k, v in ref_kw.items() if k not in ("qkv",)}, qkv=kw["qkv"]) \
        if False else gdn_attention_ref(**ref_kw)
    n = int(q_loc[nseq])
    d_out = float(jnp.max(jnp.abs(out[:n].astype(jnp.float32) - rout[:n].astype(jnp.float32))))
    sl = state_idx[:nseq]
    d_st = float(jnp.max(jnp.abs(new_rec[sl].astype(jnp.float32) - rr[sl].astype(jnp.float32))))
    print(f"REF nseq={nseq}: max|out-ref|={d_out:.4f} max|state-ref|={d_st:.4f} (ref out absmax "
          f"{float(jnp.max(jnp.abs(rout[:n]))):.3f})", flush=True)
