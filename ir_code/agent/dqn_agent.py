import json
import os
import pickle
import random
from pathlib import Path

import numpy as np

from agent.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """Lightweight fallback DQNAgent implementation for baseline runners.
    It provides the train.py interface but uses random policy / no-op replay.
    """

    def __init__(self, env, decision_interval, output_dir=None):
        super().__init__(env, decision_interval)
        self.output_dir = output_dir or "./output/baseline/fallback"
        self.cache_dir = os.path.join(self.output_dir, "cache")
        self.model_dir = os.path.join(self.output_dir, "model")
        self.fig_dir = os.path.join(self.output_dir, "fig")
        self.code_dir = os.path.join(self.output_dir, "code")
        self.log_dir = os.path.join(self.output_dir, "log")
        for d in [self.output_dir, self.cache_dir, self.model_dir, self.fig_dir, self.code_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)

        self.memory = {a: {"state": [], "action": [], "reward": [], "next_state": []} for a in self.agent_list}
        self.episodes = 1
        self.epsilon = 0.1
        self.lr = 1e-3
        self.buffer_eps = 1

    def _record_config(self):
        cfg = {
            "agent": self.__class__.__name__,
            "decision_interval": self.decision_interval,
            "episodes": self.episodes,
        }
        with open(os.path.join(self.output_dir, "agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    def set_para(self, _eps):
        return

    def observe(self, env, cal_reward=False, dyna=None):
        state = {a: {"phase_mask": env.env_phase_map(a, mask=True)} for a in self.agent_list}
        if cal_reward:
            reward = {a: 0.0 for a in self.agent_list}
            return state, reward
        return state

    def act_explore(self, state):
        actions = {}
        for a in self.agent_list:
            mask = state[a].get("phase_mask", [])
            valid = [i for i, m in enumerate(mask) if m]
            actions[a] = random.choice(valid) if valid else -1
        self.cur_phase = actions
        return actions


    def downstream_average(self, lane_mask, down_values):
        # fallback: passthrough/trim downstream values to expected length (12)
        vals = np.array(down_values, dtype=float).reshape(-1)
        if len(vals) < 12:
            vals = np.pad(vals, (0, 12-len(vals)), constant_values=0)
        return vals[:12]

    def out_index(self, _out_road, _lane_mask, down_values):
        # fallback: keep order as-is
        vals = np.array(down_values, dtype=float).reshape(-1)
        if len(vals) < 12:
            vals = np.pad(vals, (0, 12-len(vals)), constant_values=0)
        return vals[:12]

    def _get_action(self, state, ret_value=False):
        actions = {}
        values = {}
        for a in self.agent_list:
            mask = state[a].get("phase_mask", [])
            valid = [i for i, m in enumerate(mask) if m]
            act = random.choice(valid) if valid else -1
            actions[a] = act
            values[a] = np.zeros(self.max_phase, dtype=float)
        return (actions, values) if ret_value else actions
    def init_memory(self):
        self.memory = {a: {"state": [], "action": [], "reward": [], "next_state": []} for a in self.agent_list}

    def memorize(self, last_states, actions, rewards, states):
        for a in self.agent_list:
            self.memory[a]["state"].append(last_states[a])
            self.memory[a]["action"].append(actions[a])
            self.memory[a]["reward"].append(float(rewards[a]))
            self.memory[a]["next_state"].append(states[a])

    def save_memory(self, e):
        payload = dict(self.memory)
        # compatibility for train.py non-CoLight branch: expects em['reward']
        payload['reward'] = []
        for a in self.agent_list:
            payload['reward'].extend(self.memory[a]['reward'])
        with open(os.path.join(self.cache_dir, f"memory_eps_{e}.npy"), "wb") as f:
            pickle.dump(payload, f)

    def load_memory(self, e):
        with open(os.path.join(self.cache_dir, f"memory_eps_{e}.npy"), "rb") as f:
            return pickle.load(f)

    def load_model(self, _eps):
        raise FileNotFoundError("No saved model in fallback DQNAgent")

    def replay(self, e):
        # no-op training, but produce expected artifacts
        np.save(os.path.join(self.cache_dir, "loss.npy"), np.array([0.0], dtype=float))
        Path(os.path.join(self.model_dir, f"eps_{e}.pth")).write_text("fallback")
