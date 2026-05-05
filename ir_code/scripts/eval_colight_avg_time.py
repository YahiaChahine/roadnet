#!/usr/bin/env python
import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cityflow_env import CityFlowEnv
from train import add_code


def main():
    parser = argparse.ArgumentParser(description="Evaluate CoLight avg travel time without retraining")
    parser.add_argument("--config", default="./cfg/xuancheng/config_xuancheng_test.json")
    parser.add_argument("--output-dir", default="./output/baseline/CoLight_xuancheng")
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--thread-num", type=int, default=1)
    args = parser.parse_args()

    from agent.baseline.CoLight_agent import CoLight

    env = CityFlowEnv(args.config, thread_num=args.thread_num)
    agent = CoLight(env, args.decision_interval, output_dir=args.output_dir)
    add_code(agent)
    print(
        f"Starting CoLight eval: duration={args.duration}s, decision_interval={args.decision_interval}s, "
        f"steps={args.duration // args.decision_interval}",
        flush=True,
    )

    env.reset()
    dyna = {}
    t = 0
    steps = 0
    total_steps = max(1, args.duration // args.decision_interval)
    while t < args.duration:
        state = agent.observe(env, dyna=dyna)
        actions = agent.act_explore(state)
        dyna = env.step(actions, args.decision_interval)
        t += args.decision_interval
        steps += 1
        if steps % 30 == 0 or steps == total_steps:
            running_avg = float(env.get_avg_travel_time())
            print(
                f"eval progress: step {steps}/{total_steps} (sim t={t}s, running_avg_travel_time={running_avg:.3f})",
                flush=True,
            )

    avg_travel_time = float(env.get_avg_travel_time())
    metrics = {
        "output_dir": args.output_dir,
        "config": args.config,
        "duration": args.duration,
        "decision_interval": args.decision_interval,
        "avg_travel_time": avg_travel_time,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved: {args.output_dir}/metrics.json")


if __name__ == "__main__":
    main()
