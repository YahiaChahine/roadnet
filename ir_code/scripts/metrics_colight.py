#!/usr/bin/env python
import argparse
import glob
import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np


COST_RE = re.compile(r"Cost time of\s+\d+:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)")


def parse_cost_times(output_dir: str) -> Tuple[List[float], Optional[float], Optional[str]]:
    candidates = []
    for pattern in ("*.log", "*.txt", "**/*.log", "**/*.txt"):
        candidates.extend(glob.glob(os.path.join(output_dir, pattern), recursive=True))

    best_file = None
    best_times: List[float] = []
    best_total = None
    for path in sorted(set(candidates), key=lambda p: os.path.getmtime(p), reverse=True):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        matches = COST_RE.findall(text)
        if not matches:
            continue
        times = [float(m[0]) for m in matches]
        total = float(matches[-1][1]) if matches else None
        if len(times) > len(best_times):
            best_times = times
            best_total = total
            best_file = path
    return best_times, best_total, best_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/baseline/CoLight_xuancheng")
    args = parser.parse_args()

    p = args.output_dir
    reward_path = os.path.join(p, "reward.npy")
    loss_path = os.path.join(p, "training_loss.npy")
    print("reward shape", np.load(reward_path, allow_pickle=True).shape if os.path.exists(reward_path) else "missing")
    print("loss shape", np.load(loss_path, allow_pickle=True).shape if os.path.exists(loss_path) else "missing")

    metrics_path = os.path.join(p, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        print("avg travel time", d.get("avg_travel_time"))
        return

    avg_npy_path = os.path.join(p, "avg_travel_time.npy")
    if os.path.exists(avg_npy_path):
        arr = np.load(avg_npy_path, allow_pickle=True)
        print("avg travel time", float(arr[0]) if arr.size else "missing")
        return

    times, total, source = parse_cost_times(p)
    if times:
        print("avg travel time", "missing")
        print("avg episode wall time (from logs)", float(np.mean(times)))
        print("latest cumulative wall time (from logs)", total)
        print("log source", source)
    else:
        print("avg travel time", "missing")


if __name__ == "__main__":
    main()
