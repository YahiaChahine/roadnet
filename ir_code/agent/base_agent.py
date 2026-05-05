#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File: base_agent.py
@Time: 2022/03/15 13:39:19
@Author: Cl
@Desc: Basic signal control agent
'''


import os
import random
from pathlib import Path
import numpy as np


class BaseAgent:
    def __init__(self, env, decision_interval):
        self.decision_interval = decision_interval
        self.max_phase = len(env.env_config.phase_expansion)
        self.road_lane_num = env.env_config.road_lane_num

        self.load_roadnet(env.intersections, env.roads, env.agents)
        self.cur_phase = dict.fromkeys(self.agent_list, 0)
        self.cur_phase_itv = dict.fromkeys(self.agent_list, 0)

    def load_roadnet(self, intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agent_list = agents

    def _get_action(self, agent_id, state):
        ''' Return phase index >= 0, -1 is all red '''
        raise NotImplementedError()

    def act(self, state):
        actions = self._get_action(state)

        for agent_id in self.agent_list:
            if actions[agent_id] == self.cur_phase[agent_id]:
                self.cur_phase_itv[agent_id] += 1
            else:
                self.cur_phase_itv[agent_id] = 0

        self.cur_phase = actions
        return actions
