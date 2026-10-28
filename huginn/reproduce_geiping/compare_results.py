"""Compare lm-eval outputs against Geiping et al. (2025) tables.

MATCH: within max(2 SE, 1.5 pts). CLOSE: within max(4 SE, 3 pts). OFF otherwise.

  python compare_results.py outputs/repro [--csv comparison.csv]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGETS = json.loads((HERE / "paper_targets.json").read_text())


def parse_model_args(model_args):
    if isinstance(model_args, dict):
        return {k: str(v) for k, v in model_args.items()}
    out = {}
    for part in str(model_args or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_runs(root):
    runs = []
    for path in sorted(Path(root).rglob("results_*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"skipping {path}: {e}", file=sys.stderr)
            continue
        args = parse_model_args(data.get("config", {}).get("model_args"))
        loops = args.get("mean_recurrence")
        runs.append({
            "path": path,
            "loops": int(float(loops)) if loops else None,
            "sys_prompt": bool(data.get("system_instruction")),
            "results": data.get("results", {}),
            "n_samples": data.get("n-samples", {}),
        })
    return runs


def pct(x):
    return None if x is None else 100.0 * float(x)


def verdict(ours, paper, se, partial):
    if ours is None or paper is None:
        return "?"
    gap = abs(ours - paper)
    se = se or 0.0
    if gap <= max(2 * se, 1.5):
        v = "MATCH"
    elif gap <= max(4 * se, 3.0):
        v = "CLOSE"
    else:
        v = "OFF"
    return v + (" (PARTIAL)" if partial else "")


def sample_counts(run, task):
    n = run["n_samples"].get(task, {})
    eff, orig = n.get("effective"), n.get("original")
    partial = eff is not None and orig is not None and eff < orig
    return eff, orig, partial


def rows_for_run(run):
    rows = []
    r = run["loops"]
    res = run["results"]
    key = str(r)

    # Table 1: zero-shot likelihood tasks
    for task, metric in TARGETS["table1_metric"].items():
        if task not in res:
            continue
        m = res[task]
        ours = pct(m.get(f"{metric},none"))
        se = pct(m.get(f"{metric}_stderr,none"))
        paper = TARGETS["table1"].get(key, {}).get(task)
        eff, orig, partial = sample_counts(run, task)
        rows.append([task, metric, r, eff, orig, ours, paper, se, verdict(ours, paper, se, partial)])

    # GSM8K CoT / GSM8K: strict and flexible
    for task in ("gsm8k_cot", "gsm8k"):
        if task not in res:
            continue
        m = res[task]
        if r == 1:
            block = TARGETS["gsm8k"]["sanity_r1"]
        elif run["sys_prompt"]:
            block = TARGETS["gsm8k"]["with_sys_prompt"]
        else:
            block = TARGETS["gsm8k"]["without_sys_prompt"]
        pair = block.get(key, {}).get(task)
        eff, orig, partial = sample_counts(run, task)
        for i, filt in enumerate(("strict-match", "flexible-extract")):
            ours = pct(m.get(f"exact_match,{filt}"))
            se = pct(m.get(f"exact_match_stderr,{filt}"))
            paper = pair[i] if pair else None
            label = f"{task} ({'sys prompt' if run['sys_prompt'] else 'no sys prompt'})"
            rows.append([label, filt, r, eff, orig, ours, paper, se, verdict(ours, paper, se, partial)])

    # Code
    for task, paper in TARGETS["code"].get(key, {}).items():
        if task not in res:
            continue
        m = res[task]
        metric = next((k for k in m if k.startswith("pass") and "stderr" not in k), None)
        if metric is None:
            continue
        ours = pct(m[metric])
        se_key = metric.replace(",", "_stderr,", 1)
        se = pct(m.get(se_key))
        eff, orig, partial = sample_counts(run, task)
        rows.append([task, metric.split(",")[0], r, eff, orig, ours, paper, se, verdict(ours, paper, se, partial)])
    return rows


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default="outputs/repro")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    runs = load_runs(args.root)
    if not runs:
        print(f"no results_*.json found under {args.root}")
        return
    rows = []
    for run in runs:
        rows.extend(rows_for_run(run))
    rows.sort(key=lambda x: (x[0], x[1], x[2] or 0))

    header = ["benchmark", "metric", "loops", "n", "of", "ours", "paper", "stderr", "verdict"]
    widths = [34, 17, 5, 6, 6, 7, 7, 7, 16]
    print("\n" + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        cells = [row[0], row[1], str(row[2]), str(row[3] or "-"), str(row[4] or "-"),
                 fmt(row[5]), fmt(row[6]), fmt(row[7]), row[8]]
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    # recurrence trend, the paper's headline: accuracy should rise from 4 to 32 loops
    by_task = {}
    for row in rows:
        if row[0] in TARGETS["table1_metric"] and row[5] is not None:
            by_task.setdefault(row[0], {})[row[2]] = row[5]
    trends = {t: v for t, v in by_task.items() if len(v) >= 2}
    if trends:
        print("\nAccuracy by loop count (does more looping help, as the paper claims?):")
        for t, v in sorted(trends.items()):
            seq = " -> ".join(f"r{r}: {v[r]:.1f}" for r in sorted(v))
            print(f"  {t:<15} {seq}")

    counts = {}
    for row in rows:
        k = row[8].split()[0]
        counts[k] = counts.get(k, 0) + 1
    print("\nsummary:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
