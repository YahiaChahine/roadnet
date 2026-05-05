#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File: mp_agent.py
@Time: 2022/03/15 13:42:03
@Author: Cl
@Desc: Max-pressure signal control agent
'''


import numpy as np
from .base_agent import BaseAgent
from time import sleep


class MPAgent(BaseAgent):
    ''' max-pressure agent '''
    name = 'Max-pressure'

    def __init__(self, env, decision_interval):
        super().__init__(env, decision_interval)
        self.env_phase_map = env.env_phase_map
        self.num_phases = 8  # 标准相位数

    def observe(self, env, dyna):
        state = {}
        lane_veh_num = env.get_lane_veh_num()

        for agent_id in env.agents:
            if agent_id =='34180200000494':
                sleep(0)
            phase_validity = env.env_phase_map(agent_id, True)
            valid_phase_mask = np.array(phase_validity)
            valid_phase_indices = np.where(valid_phase_mask)[0].tolist()
            
            state[agent_id] = {
                'valid_phase_mask': valid_phase_mask,  # 布尔数组表示哪些相位有效
                'valid_phase_indices': valid_phase_indices,  # 有效相位的索引列表
                'phase_pressure': np.zeros(self.num_phases),  # 初始化所有相位的压力
                'phase_inlane_veh_num': np.zeros(self.num_phases)  # 初始化所有相位的入口车道车辆数
            }

            if np.any(valid_phase_mask):
                # 计算全红相位的基准压力
                agent_phase_lane = env.get_inter_phase_lane(agent_id, -1)
                if agent_phase_lane.size:
                    inlane = list(map(lambda x: lane_veh_num.get(x, -np.inf), agent_phase_lane[:, 0]))
                    outlane = list(map(lambda x: lane_veh_num.get(x, 0), agent_phase_lane[:, 1]))
                    phase_pressure_all_red = np.subtract(inlane, outlane).sum()
                else:
                    phase_pressure_all_red = 0

                # 只为有效相位计算压力
                for phase_no in valid_phase_indices:
                    agent_phase_lane = env.get_inter_phase_lane(agent_id, phase_no)
                    inlane = list(map(lambda x: lane_veh_num[x], agent_phase_lane[:, 0]))
                    outlane = list(map(lambda x: lane_veh_num[x], agent_phase_lane[:, 1]))
                    state[agent_id]['phase_pressure'][phase_no] = np.subtract(inlane, outlane).sum() - phase_pressure_all_red
                    state[agent_id]['phase_inlane_veh_num'][phase_no] = sum(inlane)

        return state

    def _get_action(self, state):
        actions = {}

        for a, s in state.items():
            agent_cur_phase = self.cur_phase[a]
            valid_phase_mask = s['valid_phase_mask']
            valid_phase_indices = s['valid_phase_indices']
            
            if len(valid_phase_indices) == 0:
                actions[a] = -1
                continue
                
            # 创建当前相位的one-hot编码（对所有8个相位）
            last_phase = np.zeros(self.num_phases)
            if agent_cur_phase >= 0 and agent_cur_phase < self.num_phases:
                last_phase[agent_cur_phase] = 1
            
            # 使用lexsort直接对所有相位进行排序，但通过valid_phase_mask只考虑有效相位
            # 为无效相位设置极小值，确保它们排在最前面（不会被选择）
            masked_pressures = s['phase_pressure'] * valid_phase_mask - 1e6 * (~valid_phase_mask)
            masked_inlane_nums = s['phase_inlane_veh_num'] * valid_phase_mask - 1e6 * (~valid_phase_mask)
            masked_last_phase = last_phase * valid_phase_mask
            
            # 使用lexsort进行排序
            best_phase = np.lexsort((masked_last_phase, masked_inlane_nums, masked_pressures))[-1]
            
            # 检查当前相位是否已经使用超过5次
            if (agent_cur_phase == best_phase and self.cur_phase_itv[a] >= 5):
                # 找到次优相位
                sorted_phases = np.lexsort((masked_last_phase, masked_inlane_nums, masked_pressures))
                # 从后往前找第一个不是当前相位的有效相位
                for i in range(len(sorted_phases) - 2, -1, -1):
                    candidate_phase = sorted_phases[i]
                    if valid_phase_mask[candidate_phase] and candidate_phase != agent_cur_phase:
                        best_phase = candidate_phase
                        break
            
            # 维护相位持续时间计数器
            if best_phase == agent_cur_phase:
                self.cur_phase_itv[a] += 1
            else:
                self.cur_phase_itv[a] = 1
                
            actions[a] = best_phase
        return actions
