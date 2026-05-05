#!/usr/bin/env python
import argparse
import importlib
import json
import os
import re
import sys
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _import_cityflow_env():
    try:
        mod = importlib.import_module("cityflow_env")
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        print(
            "Dependency import failed while loading cityflow_env.py. "
            f"Missing module: '{missing}'.\n"
            "Install required packages first (example):\n"
            "  pip install numpy scipy matplotlib tqdm cityflow",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return mod.CityFlowEnv


def _import_agent_class(agent_name: str):
    try:
        if agent_name == "MPLight":
            from agent.baseline.MPLight_agent import MPLight as AgentClass
        else:
            from agent.baseline.CoLight_agent import CoLight as AgentClass
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        print(
            "Dependency import failed while loading agent modules. "
            f"Missing module: '{missing}'.\n"
            "Install required packages first (example):\n"
            "  pip install numpy scipy matplotlib tqdm cityflow",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return AgentClass


def _latest_model_episode(model_dir: str) -> Optional[int]:
    if not os.path.isdir(model_dir):
        return None
    # Support both checkpoint naming styles used in this repo:
    # - model_<N>.* (older helper scripts)
    # - eps_<N>.pth (baseline agents)
    patterns = [
        re.compile(r"(?:^|_)model_(\d+)\."),
        re.compile(r"(?:^|_)eps_(\d+)\.pth$"),
    ]
    episodes = []
    for name in os.listdir(model_dir):
        for pat in patterns:
            m = pat.search(name)
            if m:
                episodes.append(int(m.group(1)))
                break
    return max(episodes) if episodes else None




def _infer_duration_from_flow(config_path: str) -> Optional[int]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    flow_file = cfg.get("flowFile")
    if not flow_file:
        return None
    if not os.path.isabs(flow_file):
        flow_file = os.path.join(os.path.dirname(config_path), flow_file)

    try:
        with open(flow_file, "r", encoding="utf-8") as f:
            flow_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(flow_data, list) or not flow_data:
        return None

    max_end_time = 0.0
    for item in flow_data:
        if not isinstance(item, dict):
            continue
        start = float(item.get("startTime", 0))
        end = float(item.get("endTime", start))
        max_end_time = max(max_end_time, end)

    return int(max_end_time) if max_end_time > 0 else None


def resolve_eval_duration(config: str, requested_duration: int, full_flow_duration: bool) -> int:
    inferred = _infer_duration_from_flow(config)
    if not full_flow_duration or inferred is None:
        return requested_duration
    if inferred <= requested_duration:
        return requested_duration
    print(
        f"Using inferred full flow duration from flowFile: {inferred}s (requested {requested_duration}s)",
        flush=True,
    )
    return inferred

def evaluate_avg_travel_time(agent_cls, config: str, output_dir: str, duration: int, decision_interval: int, thread_num: int) -> float:
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if decision_interval <= 0:
        raise ValueError("decision-interval must be > 0")

    CityFlowEnv = _import_cityflow_env()
    env = CityFlowEnv(config, thread_num=thread_num)
    agent = agent_cls(env, decision_interval, output_dir=output_dir)

    eps = _latest_model_episode(getattr(agent, "model_dir", ""))
    if eps is not None:
        try:
            agent.load_model(eps)
            print(f"Loaded model episode: {eps}", flush=True)
        except (FileNotFoundError, NotImplementedError) as exc:
            print(
                "Found checkpoint files but this agent implementation cannot "
                f"load them for evaluation ({exc}). "
                "Falling back to current agent weights.",
                flush=True,
            )
    else:
        print("No saved model found; evaluating current agent weights.", flush=True)

    env.reset()
    dyna = {}
    t = 0
    while t < duration:
        state = agent.observe(env, dyna=dyna)
        actions = agent.act(state)
        dyna = env.step(actions, decision_interval)
        t += decision_interval

    return float(env.get_avg_travel_time())


def main():
    parser = argparse.ArgumentParser(description="Evaluate avg travel time for saved RL models")
    parser.add_argument("--agent", choices=["MPLight", "CoLight"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--full-flow-duration", action="store_true",
                        help="Use full duration inferred from flowFile in config")
    args = parser.parse_args()

    AgentClass = _import_agent_class(args.agent)

    duration = resolve_eval_duration(args.config, args.duration, args.full_flow_duration)

    avg_travel_time = evaluate_avg_travel_time(
        AgentClass,
        args.config,
        args.output_dir,
        duration,
        args.decision_interval,
        args.thread_num,
    )
    metrics = {
        "agent": args.agent,
        "output_dir": args.output_dir,
        "config": args.config,
        "duration": duration,
        "decision_interval": args.decision_interval,
        "avg_travel_time": avg_travel_time,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
