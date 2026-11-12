"""Success rates from run.py's result.json files, by mode x task family and by mode x task.

    python benchmarks/openha/score.py ~/runs/openha/LABEL [--suite suites/eval255.json] [--json]

With --suite only that suite's tasks and repeats count, and a suite with `groups` ({task-name
prefix: group}) is also scored by group, each task weighted alike.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(root.glob("*/*/r*/result.json"))]


def table(rows: list[dict], key) -> dict:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        decisions = sum(r["decisions"] for r in rs)
        lat = [r["latency_ms"] for r in rs if r.get("latency_ms") is not None]
        out[k] = {"n": len(rs), "success": round(sum(r["success"] for r in rs) / len(rs), 3),
                  "unparsed": round(sum(r["parse_failures"] for r in rs) / max(1, decisions), 4),
                  "gui_share": round(sum(r["gui_decisions"] for r in rs) / max(1, decisions), 3),
                  "latency_ms": round(sum(lat) / len(lat), 1) if lat else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--suite")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = load(Path(args.root).expanduser())
    res = {}
    if args.suite:
        suite = json.loads(Path(args.suite).read_text())
        tasks, repeats = set(suite["tasks"]), suite.get("repeats", 1)
        rows = [r for r in rows if r["task"] in tasks and r["rep"] < repeats]
        if suite.get("groups"):
            res["by_group"] = table(rows, lambda r: f"{r['mode']} {suite['groups'][r['task'].split(':')[0]]}")
    res.update({"by_family": table(rows, lambda r: f"{r['mode']} {r['family']}"),
                "by_task": table(rows, lambda r: f"{r['mode']} {r['task']}"),
                "overall": table(rows, lambda r: r["mode"])})
    if args.json:
        print(json.dumps(res, indent=1))
        return
    for name in [k for k in ("overall", "by_group", "by_family", "by_task") if k in res]:
        print(f"== {name}")
        for k, v in res[name].items():
            print(f"  {k:48s} n={v['n']:4d} success={v['success']:.3f} unparsed={v['unparsed']:.4f} "
                  f"gui={v['gui_share']:.3f} latency_ms={v['latency_ms']}")


if __name__ == "__main__":
    main()
