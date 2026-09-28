"""Prefix-cache correctness gate for hybrid (GDN) models: warm (cache-hit) greedy output must match cold output.

Cold = same prompt with a fresh cache_salt (no reuse possible). Warm = a salt that an earlier request has already seeded.
Cases (block = 256 tokens, prefill chunk = 2048):
  exact     : seed prompt of length L, then the identical prompt again (hit on its own full blocks)
  extend    : seed L, then seed + its output + a new "user turn" (multi-turn agent continuation)
  fork      : seed a long prompt, then a prompt sharing only its first F tokens + a different suffix
              (F in the middle of a prefill chunk: the #3634 case, intermediate blocks never checkpointed)
Prints per case: cached tokens, whether the greedy outputs are identical, first divergence index, and the
top-1 logprob of the first generated token (cold vs warm) to tell numerical noise from state corruption.

python prefix_cache_gate.py [--model-path /mnt/llm/models/...] [--max-tokens 32]
"""
import argparse, json, random, sys, uuid
import requests
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="qwen3.8-27b")
ap.add_argument("--model-path", default="/mnt/llm/models/Qwen3.8-27B-Uncensored-INT8")
ap.add_argument("--max-tokens", type=int, default=32)
ap.add_argument("--out", default=None)
ap.add_argument("--corpus", default="~/vllm-tpu/src/tpu-inference/tpu_inference")
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.model_path)
WORDS = ("the agent reads a file then calls a tool and returns the result to the user while the cache keeps "
         "every prefix block so that later turns can resume from the saved recurrent state without recomputing").split()


_CORPUS = None


def filler(n, seed):
    """n tokens of real Python source (confident next-token predictions), starting at a seed-dependent offset."""
    global _CORPUS
    if _CORPUS is None:
        import glob, os
        files = sorted(glob.glob(os.path.expanduser(args.corpus) + "/**/*.py", recursive=True))
        _CORPUS = tok.encode("\n".join(open(f, errors="replace").read() for f in files), add_special_tokens=False)
    start = random.Random(seed).randrange(0, len(_CORPUS) - n - 1)
    return _CORPUS[start:start + n]


def gen(ids, salt):
    body = {"model": args.model, "prompt": ids, "max_tokens": args.max_tokens, "temperature": 0, "logprobs": 1,
            "cache_salt": salt}
    r = requests.post(f"{args.base}/v1/completions", json=body, timeout=900)
    r.raise_for_status()
    j = r.json()
    c = j["choices"][0]
    toks = c["logprobs"]["tokens"] if c.get("logprobs") else []
    lp0 = c["logprobs"]["token_logprobs"][0] if c.get("logprobs") else None
    cached = ((j.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return dict(text=c["text"], toks=toks, lp0=lp0, cached=cached)


def compare(name, ids, salt, seed_len, seed=None):
    warm = gen(ids, salt)
    cold = gen(ids, "cold-" + uuid.uuid4().hex)
    seed_same = None if seed is None else seed["toks"] == cold["toks"]
    same = warm["toks"] == cold["toks"]
    div = next((i for i, (a, b) in enumerate(zip(warm["toks"], cold["toks"])) if a != b), None)
    # CORRUPT: the warm state is clearly wrong (first token differs, or a confident cold prediction turns uncertain)
    corrupt = (div == 0) or (cold["lp0"] > -0.05 and warm["lp0"] < -0.3)
    flag = "OK  " if same else ("CORRUPT" if corrupt else ("NOISE?" if div is not None and div >= 8 else "FAIL"))
    row = dict(flag=flag.strip(), case=name, prompt=len(ids), seed=seed_len, seed_eq_cold=seed_same, cached=warm["cached"],
               cold_cached=cold["cached"], identical=same, first_div=div, lp0_cold=cold["lp0"], lp0_warm=warm["lp0"],
               cold=cold["text"][:60], warm=warm["text"][:60])
    print(f"[{flag}] {name:34s} prompt={len(ids):6d} cached={warm['cached']!s:>6} div={div!s:>4} "
          f"lp0 cold={cold['lp0']:.4f} warm={warm['lp0']:.4f} seed==cold={seed_same} | warm={warm['text'][:40]!r} cold={cold['text'][:40]!r}",
          flush=True)
    return row


rows = []
case_id = 0
# exact + extend, lengths around 256-token blocks and the 2048-token chunk
QUICK = __import__('os').environ.get('QUICK') == '1'
for L in ((513, 2049, 2052, 2058, 2300) if QUICK else (255, 256, 257, 260, 266, 511, 512, 513, 1000, 2047, 2048, 2049, 2052, 2058, 2300, 4101, 5000)):
    case_id += 1
    ids = tok.encode(f"[case {case_id} {uuid.uuid4().hex[:8]}]", add_special_tokens=False) + filler(L, case_id)
    ids = ids[:L]
    salt = "seed-" + uuid.uuid4().hex
    seed = gen(ids, salt)                                   # seeds the cache under `salt`
    rows.append(compare(f"exact L={L}", ids, salt, L, seed))
    turn2 = ids + tok.encode(seed["text"], add_special_tokens=False) + filler(37, 1000 + case_id)
    rows.append(compare(f"extend L={L}(+{len(turn2) - L})", turn2, salt, L))

# forked prefixes off one long seed: fork points inside prefill chunks (never checkpointed) and on boundaries
case_id += 1
base = tok.encode(f"[fork {uuid.uuid4().hex[:8]}]", add_special_tokens=False) + filler(6000, 4242)
base = base[:6000]
salt = "seed-" + uuid.uuid4().hex
gen(base, salt)
for F in ((1000, 1024, 2100, 3000, 5300) if QUICK else (300, 1000, 1024, 2048, 2100, 2560, 3000, 4096, 4400, 5300)):
    fork = base[:F] + filler(50, 7000 + F)
    rows.append(compare(f"fork F={F} of 6000", fork, salt, 6000))

bad = [r for r in rows if not r["identical"]]
print(f"\nCORRUPT={sum(r['flag'] == 'CORRUPT' for r in rows)} FAIL(<8)={sum(r['flag'] == 'FAIL' for r in rows)} NOISE={sum(r['flag'] == 'NOISE?' for r in rows)}")
print(f"{len(rows) - len(bad)}/{len(rows)} identical; {sum(1 for r in bad if r['first_div'] is not None and r['first_div'] < 8)} "
      f"diverge within the first 8 tokens")
if args.out:
    json.dump(rows, open(args.out, "w"), indent=1)
