#!/usr/bin/env python
import argparse
import json
import os
import sys

import numpy as np

from cityflow_env import CityFlowEnv
import train as train_module
from train import add_code
from scripts.eval_avg_time import evaluate_avg_travel_time, resolve_eval_duration


def _validate_mplight_dependencies():
    missing = []
    if not os.path.exists("agent/dqn_agent.py"):
        missing.append("agent/dqn_agent.py")
    if not os.path.exists("model/model_tf/baseline/MPLight_model.py"):
        missing.append("model/model_tf/baseline/MPLight_model.py")
    if missing:
        print("MPLight dependencies are missing from this repository:")
        for m in missing:
            print(f" - {m}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run MPLight training")
    parser.add_argument("--config", default="./cfg/xuancheng/config_xuancheng_test.json")
    parser.add_argument("--output-dir", default="./output/baseline/MPLight_xuancheng")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--parallel-num", type=int, default=0)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--full-flow-duration", action="store_true",
                        help="Use full duration inferred from flowFile in config for evaluation")
    args = parser.parse_args()

    if not _validate_mplight_dependencies():
        sys.exit(2)

    from agent.baseline.MPLight_agent import MPLight

    train_module.Agent = MPLight
    os.makedirs(args.output_dir, exist_ok=True)
    env = CityFlowEnv(args.config, thread_num=args.thread_num)
    agent = MPLight(env, args.decision_interval, output_dir=args.output_dir)
    agent.episodes = args.episodes
    add_code(agent)
    train_module.train(
        agent,
        args.config,
        args.duration,
        parallel_num=args.parallel_num,
        output_dir=args.output_dir,
        gen_flow=True,
        thread_num=args.thread_num,
    )

    eval_duration = resolve_eval_duration(args.config, args.duration, args.full_flow_duration)

    avg_travel_time = evaluate_avg_travel_time(MPLight, args.config, args.output_dir, eval_duration, args.decision_interval, args.thread_num)
    metrics = {
        "output_dir": args.output_dir,
        "episodes": args.episodes,
        "duration": eval_duration,
        "decision_interval": args.decision_interval,
        "avg_travel_time": avg_travel_time,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(args.output_dir, "avg_travel_time.npy"), np.array([avg_travel_time], dtype=float))
    print(f"Saved MPLight metrics: {args.output_dir}/metrics.json")


if __name__ == "__main__":
    main()
