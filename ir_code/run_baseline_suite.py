#!/usr/bin/env python
import argparse
import json
import os
import tempfile
from datetime import datetime
from glob import glob

from cityflow_env import CityFlowEnv
import train as train_module
from train import add_code


def evaluate_avg_travel_time(env, agent, duration, decision_interval):
    env.reset()
    dyna = {}
    t = 0
    while t < duration:
        state = agent.observe(env, dyna=dyna)
        actions = agent.act_explore(state)
        dyna = env.step(actions, decision_interval)
        t += decision_interval
    return float(env.get_avg_travel_time())


def main():
    p = argparse.ArgumentParser(description="Run baseline suite over multiple flow files")
    p.add_argument("--agent", choices=["colight", "mplight"], required=True)
    p.add_argument("--config", default="./cfg/xuancheng/config_xuancheng_test.json")
    p.add_argument("--flows-glob", default="data/xuancheng_mod/xuancheng_2023_04_*_8to9.json")
    p.add_argument("--output-dir", default="./output/baseline")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--duration", type=int, default=1200)
    p.add_argument("--decision-interval", type=int, default=10)
    p.add_argument("--thread-num", type=int, default=1)
    args = p.parse_args()

    if args.agent == "colight":
        from agent.baseline.CoLight_agent import CoLight as Agent
        suite_name = "CoLight_suite_xuancheng"
    else:
        from agent.baseline.MPLight_agent import MPLight as Agent
        suite_name = "MPLight_suite_xuancheng"

    train_module.Agent = Agent

    with open(args.config) as f:
        base_cfg = json.load(f)

    flows = sorted(glob(args.flows_glob))
    if not flows:
        raise SystemExit(f"No files matched: {args.flows_glob}")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suite_root = os.path.join(args.output_dir, f"{suite_name}_{ts}")
    os.makedirs(suite_root, exist_ok=True)

    summary = []
    for flow in flows:
        flow_name = os.path.splitext(os.path.basename(flow))[0]
        flow_out = os.path.join(suite_root, flow_name)
        os.makedirs(flow_out, exist_ok=True)

        cfg = dict(base_cfg)
        cfg["flowFile"] = flow
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(cfg, tf)
            cfg_path = tf.name

        try:
            env = CityFlowEnv(cfg_path, thread_num=args.thread_num)
            agent = Agent(env, args.decision_interval, output_dir=flow_out)
            agent.episodes = args.episodes
            add_code(agent)
            train_module.train(
                agent,
                cfg_path,
                args.duration,
                parallel_num=0,
                output_dir=flow_out,
                gen_flow=False,
                thread_num=args.thread_num,
            )
            avg_travel_time = evaluate_avg_travel_time(env, agent, args.duration, args.decision_interval)
            summary.append({"flowFile": flow, "outputDir": flow_out, "avg_travel_time": avg_travel_time, "status": "ok"})
        finally:
            os.unlink(cfg_path)

    with open(os.path.join(suite_root, "metrics_suite.json"), "w") as f:
        json.dump({"agent": args.agent, "episodes": args.episodes, "results": summary, "mean_avg_travel_time": (sum(r.get("avg_travel_time",0.0) for r in summary)/len(summary) if summary else None)}, f, indent=2)
    print(f"Saved suite summary: {suite_root}/metrics_suite.json")


if __name__ == "__main__":
    main()
