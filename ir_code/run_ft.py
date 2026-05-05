#!/usr/bin/env python
import argparse
import json
import os
from datetime import datetime

import numpy as np

from cityflow_env import CityFlowEnv
from agent.ft_agent import FTAgent


def main():
    parser = argparse.ArgumentParser(description="Run fixed-time (FT) agent simulation")
    parser.add_argument("--config", default="./cfg/xuancheng/config_xuancheng_test.json")
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--output-dir", default="./output/ft_agent")
    parser.add_argument("--thread-num", type=int, default=1)
    args = parser.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    env = CityFlowEnv(args.config, thread_num=args.thread_num)
    agent = FTAgent(env, args.decision_interval)

    env.reset()
    dyna = {}
    for _ in range(0, args.duration, args.decision_interval):
        state = agent.observe(env, dyna)
        actions = agent.act(state)
        dyna = env.step(actions, args.decision_interval)

    avg_travel_time = float(env.get_avg_travel_time())
    metrics = {
        "agent": "FTAgent",
        "config": args.config,
        "duration": args.duration,
        "decision_interval": args.decision_interval,
        "avg_travel_time": avg_travel_time,
        "finished_at_utc": datetime.utcnow().isoformat() + "Z",
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(run_dir, "avg_travel_time.npy"), np.array([avg_travel_time], dtype=float))

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {run_dir}")


if __name__ == "__main__":
    main()
