"""Per-op device-time diff between two traces (ms per jit_step_fun step on /device:TPU:0), grouped by op name stem."""
import collections, gzip, json, re, sys


def load(path):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    procs = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
    thr = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
    pid = [p for p, n in procs.items() if n == "/device:TPU:0"][0]
    X = [e for e in ev if e.get("pid") == pid and e.get("ph") == "X"]
    steps = [e for e in X if thr[(pid, e["tid"])] == "XLA Modules" and e["name"].startswith("jit_step_fun")]
    ops = [o for o in X if thr[(pid, o["tid"])] == "XLA Ops" and any(s["ts"] <= o["ts"] <= s["ts"] + s["dur"] for s in steps)]
    agg, ex = collections.defaultdict(float), {}
    for o in ops:
        k = o["args"].get("hlo_category", "?") + " | " + re.sub(r"[._]\d+.*$", "", o["name"])
        agg[k] += o["dur"]; ex.setdefault(k, o["args"].get("long_name", "")[:150])
    n = len(steps)
    return {k: v / n / 1e3 for k, v in agg.items()}, ex, sum(s["dur"] for s in steps) / n / 1e3, n


a, exa, sa, na = load(sys.argv[1]); b, exb, sb, nb = load(sys.argv[2])
print(f"A={sys.argv[1].split('/')[-1]}: {sa:.2f} ms/step ({na} steps) | B={sys.argv[2].split('/')[-1]}: {sb:.2f} ms/step ({nb} steps)")
rows = sorted(set(a) | set(b), key=lambda k: -abs(a.get(k, 0) - b.get(k, 0)))
for k in rows[:14]:
    print(f"{a.get(k, 0):7.3f} {b.get(k, 0):7.3f} {a.get(k, 0) - b.get(k, 0):+7.3f}  {k[:60]:60s} {(exa.get(k) or exb.get(k))[:120]}")
