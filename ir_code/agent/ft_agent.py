#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File: ft_agent.py
@Time: 2022/03/15 13:41:39
@Author: Cl
@Desc: Fix-time signal control agent
'''


import numpy as np
from .base_agent import BaseAgent


class FTAgent(BaseAgent):
    ''' fix-time agent '''
    name = 'Fix-time'

    def __init__(self, env, decision_interval):
        super().__init__(env, decision_interval)

    def observe(self, env,dyna):
        return {agent_id: [i for i, p in enumerate(env.env_phase_map(agent_id, mask=True)) if p] for agent_id in env.agents}

    def _get_action(self, state):
        actions = {}

        for a, s in state.items():
            if not s:
                actions[a] = -1
            elif self.cur_phase[a] == -1:
                actions[a] = s[0]
            else:
                try:
                    actions[a] = s[(s.index(self.cur_phase[a]) + 1) % len(s)]
                except:
                    self.cur_phase[a]=s[0]
                    actions[a] = s[(s.index(self.cur_phase[a]) + 1) % len(s)]

        return actions