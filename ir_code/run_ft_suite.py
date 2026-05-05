#!/usr/bin/env python
import argparse, json, os, tempfile
from datetime import datetime
from glob import glob

from cityflow_env import CityFlowEnv
from agent.ft_agent import FTAgent


def run_one(base_cfg, flow_file, duration, decision_interval, thread_num):
    cfg = dict(base_cfg)
    cfg["flowFile"] = flow_file
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(cfg, tf)
        tmp = tf.name
    try:
        env = CityFlowEnv(tmp, thread_num=thread_num)
        agent = FTAgent(env, decision_interval)
        env.reset()
        dyna = {}
        for _ in range(0, duration, decision_interval):
            state = agent.observe(env, dyna)
            actions = agent.act(state)
            dyna = env.step(actions, decision_interval)
        return float(env.get_avg_travel_time())
    finally:
        os.unlink(tmp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./cfg/xuancheng/config_xuancheng_test.json")
    p.add_argument("--flows-glob", default="data/xuancheng_mod/xuancheng_2023_04_*_8to9.json")
    p.add_argument("--duration", type=int, default=1200)
    p.add_argument("--decision-interval", type=int, default=10)
    p.add_argument("--thread-num", type=int, default=1)
    p.add_argument("--output-dir", default="./output/ft_agent")
    args = p.parse_args()

    with open(args.config) as f:
        base_cfg = json.load(f)

    flows = sorted(glob(args.flows_glob))
    if not flows:
        raise SystemExit(f"No files matched: {args.flows_glob}")

    results = []
    for flow in flows:
        avg = run_one(base_cfg, flow, args.duration, args.decision_interval, args.thread_num)
        results.append({"flowFile": flow, "avg_travel_time": avg})
        print(flow, avg)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"suite_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics_suite.json"), "w") as f:
        json.dump({"results": results, "mean_avg_travel_time": sum(r['avg_travel_time'] for r in results)/len(results)}, f, indent=2)
    print(f"Saved suite metrics to {out_dir}/metrics_suite.json")


if __name__ == "__main__":
    main()
