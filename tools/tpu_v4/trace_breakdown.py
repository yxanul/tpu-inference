"""Per-category device-time breakdown of a JAX/XLA TPU trace (trace.json.gz), for one device's jit_step_fun_impl."""
import collections, gzip, json, re, sys
path = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "/device:TPU:0"
ev = json.load(gzip.open(path, "rt"))["traceEvents"]
procs = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
threads = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
pid = [p for p, n in procs.items() if n == dev][0]
X = [e for e in ev if e.get("pid") == pid and e.get("ph") == "X"]
steps = [e for e in X if threads[(pid, e["tid"])] == "XLA Modules" and e["name"].startswith("jit_step_fun")]
ops = [e for e in X if threads[(pid, e["tid"])] == "XLA Ops"]
in_step = lambda o: any(s["ts"] <= o["ts"] <= s["ts"] + s["dur"] for s in steps)
ops = [o for o in ops if in_step(o)]
step_us = sum(s["dur"] for s in steps); n = len(steps)
def cat(o):
    a = o.get("args", {}); c = a.get("hlo_category", "?"); ln = a.get("long_name", ""); nm = o["name"]
    m = re.search(r'custom_call_target="([^"]+)"', ln)
    if "ragged_paged_attention" in ln or "ragged_paged_attention" in nm: return "attention: RPA kernel (Pallas)"
    if "gdn" in ln.lower() or "gdn" in nm.lower() or "delta" in ln.lower(): return "DeltaNet kernel (Pallas)"
    if "conv1d" in ln.lower(): return "DeltaNet conv1d"
    if c in ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute") or "all-reduce" in nm: return "collective: " + c
    if m: return f"custom-call: {m.group(1)[:40]}"
    if c in ("convolution", "convolution fusion", "output fusion") or "convolution" in c: return "matmul (XLA dot/conv fusion)"
    return "other: " + c
T = collections.defaultdict(float); F = collections.defaultdict(float); B = collections.defaultdict(float); C = collections.Counter(); EX = {}
for o in ops:
    k = cat(o); a = o.get("args", {})
    T[k] += o["dur"]; C[k] += 1; F[k] += float(a.get("model_flops", 0) or 0); B[k] += float(a.get("bytes_accessed", 0) or 0); EX.setdefault(k, o["name"])
busy = sum(T.values())
print(f"{n} steps, step time {step_us / n / 1e3:.1f} ms/step, op-busy {busy / n / 1e3:.1f} ms/step ({100 * busy / step_us:.0f}% of step)")
print(f"{'category':38s} {'ms/step':>8s} {'%':>5s} {'TFLOP/s':>8s} {'GB/s':>7s}  example")
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f"{k[:38]:38s} {v / n / 1e3:8.2f} {100 * v / step_us:5.1f} {F[k] / (v * 1e-6) / 1e12 if v else 0:8.1f} {B[k] / (v * 1e-6) / 1e9 if v else 0:7.0f}  {EX[k][:50]}")
tf = sum(F.values()) / n
print(f"model FLOPs/step (this device): {tf / 1e12:.2f} TFLOP -> {tf / (step_us / n * 1e-6) / 1e12:.1f} TFLOP/s per device")
top = sorted(ops, key=lambda o: -o["dur"])
agg = collections.defaultdict(float)
for o in ops: agg[o["name"]] += o["dur"]
print("\n== top ops by total time (ms/step)")
for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:18]:
    o = next(x for x in ops if x["name"] == k); a = o.get("args", {})
    print(f"{v / n / 1e3:7.2f}  {k[:40]:40s} {a.get('hlo_category', '')[:22]:22s} {a.get('long_name', '')[:110]}")
